"""RIFT-style attention-selected AR draft KV compression, with exact verification.

Independent reference of arXiv:2610.31098v1's dense-attention selector. This
bounded implementation recomputes full committed draft history each round.
It is not the paper's amortized dynamic/adaptive refresh or sparse-KD trainer.
Only the active proposal cache is compressed; recompute costs are included.
"""

import math

import torch
from transformers import DynamicCache

from .cast.verify import cache_layers
from .memory import Proposal
from .reference_decode import distribution


def select_positions(scores, budget, recent_fraction):
    """Per-head token selection; stable lower-position tie break, sorted output."""
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("RIFT scores must be finite [KV heads, context] values")
    if type(budget) is not int or budget <= 0 or not 0 < recent_fraction <= 1:
        raise ValueError(
            "RIFT requires a positive budget and recent fraction in (0, 1]"
        )
    heads, length = scores.shape
    if length <= budget:
        return torch.arange(length, device=scores.device).expand(heads, -1)
    recent = math.floor(recent_fraction * budget)
    important = budget - recent
    # The recent region is excluded before importance selection: no duplicates.
    old = scores[:, : length - recent] if recent else scores
    keep = torch.argsort(old, dim=-1, descending=True, stable=True)[:, :important]
    tail = torch.arange(length - recent, length, device=scores.device).expand(heads, -1)
    return torch.cat([keep, tail], -1).sort(-1).values


def attention_scores(attention, kv_heads, window):
    """Sum observed post-softmax queries; mean the actual GQA query-head group."""
    if (
        attention.ndim != 4
        or attention.shape[0] != 1
        or window <= 0
        or kv_heads <= 0
        or attention.shape[1] % kv_heads
    ):
        raise ValueError("RIFT requires batch one and integral GQA groups")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("RIFT attention observations must be finite probabilities")
    observed = attention[0, :, -window:, :].float()
    return (
        observed.reshape(kv_heads, -1, observed.shape[-2], observed.shape[-1])
        .sum(-2)
        .mean(1)
    )


class CompressedARProposer:
    """Full-history recompute -> select -> physically gather -> sparse AR propose."""

    def __init__(self, draft, options, temperature=0.0):
        self.draft, self.options, self.temperature = draft, options, temperature
        self.rounds = []

    @torch.inference_mode()
    def __call__(self, context, remaining, generator):
        if (
            len(context) + min(remaining, self.options.draft_tokens)
            > self.draft.config.max_position_embeddings
        ):
            raise ValueError("RIFT proposal exceeds draft context capacity")
        cache = DynamicCache()
        ids = torch.tensor([context], device=self.draft.device)
        full = self.draft(
            ids, past_key_values=cache, use_cache=True, output_attentions=True
        )
        if not full.attentions or any(a is None for a in full.attentions):
            raise ValueError(
                "RIFT requires real eager post-softmax attention observations"
            )
        count, get, put = cache_layers(cache)
        full_bytes, active_bytes, selected = 0, 0, []
        for layer in range(count):
            keys, values = get(layer)
            full_bytes += (
                keys.numel() * keys.element_size()
                + values.numel() * values.element_size()
            )
            scores = attention_scores(
                full.attentions[layer], keys.shape[1], self.options.observation_window
            )
            keep = select_positions(
                scores, self.options.context_budget, self.options.recent_fraction
            )
            selected.append(keep.tolist())
            key_index = keep[None, :, :, None].expand(1, -1, -1, keys.shape[-1])
            value_index = keep[None, :, :, None].expand(1, -1, -1, values.shape[-1])
            keys, values = keys.gather(2, key_index), values.gather(2, value_index)
            put(layer, keys, values)
            active_bytes += (
                keys.numel() * keys.element_size()
                + values.numel() * values.element_size()
            )
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = min(len(context), self.options.context_budget)
        logits, tokens, probabilities = full.logits[0, -1], [], []
        # Drop the dense attention observations before expanding the active view.
        del full
        for index in range(min(remaining, self.options.draft_tokens)):
            q = distribution(logits, self.temperature)
            token = (
                int(q.argmax())
                if self.temperature == 0
                else int(torch.multinomial(q, 1, generator=generator))
            )
            tokens.append(token)
            probabilities.append(
                {i: float(q[i]) for i in q.nonzero().flatten().tolist()}
            )
            if index + 1 < min(remaining, self.options.draft_tokens):
                # Physical slots are compressed; RoPE retains the logical position.
                out = self.draft(
                    torch.tensor([[token]], device=self.draft.device),
                    position_ids=torch.tensor(
                        [[len(context) + index]], device=self.draft.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                )
                logits = out.logits[0, -1]
        self.rounds.append(
            {
                "full_history_tokens": len(context),
                "selected_absolute_positions": selected,
                "dense_history_kv_bytes_before_gather": full_bytes,
                "active_context_kv_bytes_after_gather": active_bytes,
                "inflight_tokens": max(0, len(tokens) - 1),
                "draft_forwards": max(1, len(tokens)),
                "refresh": "full_committed_history_recompute_every_round",
            }
        )
        return Proposal(tuple(tokens), tuple(probabilities))


def run_rift(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .diagnostics import profile_generate
    from .valuation_runner import token_rows

    cfg = config.model
    models, tokenizers = [], []
    for path, revision in (
        (cfg.target_model_path, cfg.target_revision),
        (cfg.draft_model_path, cfg.draft_revision),
    ):
        model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                revision=revision,
                trust_remote_code=cfg.trust_remote_code,
                dtype=getattr(torch, cfg.torch_dtype),
                attn_implementation="eager",
            )
            .to(config.runtime.device)
            .eval()
            .requires_grad_(False)
        )
        if model.config.model_type not in {"llama", "qwen2", "qwen3"} or any(
            kind != "full_attention"
            for kind in getattr(model.config, "layer_types", ["full_attention"])
        ):
            raise ValueError(
                "RIFT reference requires dense full-attention AR Llama/Qwen models"
            )
        models.append(model)
        tokenizers.append(
            AutoTokenizer.from_pretrained(
                path, revision=revision, trust_remote_code=cfg.trust_remote_code
            )
        )
    if (
        tokenizers[0].get_vocab() != tokenizers[1].get_vocab()
        or tokenizers[0].all_special_ids != tokenizers[1].all_special_ids
        or models[0].config.vocab_size != models[1].config.vocab_size
    ):
        raise ValueError("RIFT requires identical target/draft token-ID mappings")
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    answers = []
    for identifier, ids in token_rows(config.data, tokenizers[0]):
        output, stats, proposer = profile_generate(
            models[0],
            torch.tensor([ids], device=models[0].device),
            lambda: CompressedARProposer(
                models[1], config.cache, config.generation.temperature
            ),
            config.generation,
            config.diagnostics,
            generator,
        )
        answers.append(
            {
                "id": identifier,
                "token_ids": output[0, len(ids) :].tolist(),
                "stats": stats,
                "draft_cache_rounds": proposer.rounds,
            }
        )
    return {
        "answers": answers,
        "implementation": "rift_full_history_recompute_reference",
        "production_engine_speed": False,
        "sparse_kd_training": False,
        "active_budget_scope": "context_only_plus_inflight_draft_tokens",
    }
