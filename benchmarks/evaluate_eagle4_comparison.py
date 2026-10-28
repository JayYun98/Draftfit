"""Evaluate actual comparison exports with common target-verified generation.

This deliberately slow full-prefix reference decoder checks greedy token
equality and proposal acceptance. It does not implement the external EAGLE-4
tree/cost policy and cannot establish optimized engine throughput.
"""

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from train_eagle4_comparison import (
    EAGLE_LAYERS,
    LAYERS,
    NAMES,
    digest,
    memory_guard,
    write_json,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDSparkModel,
)
from draftfit.modeling.auto import AutoDraftModel
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.training.joint_verification import _generate, _propose


def eagle4_propose(bundle, features, anchor, count):
    """Greedy chain using the candidate head and recurrent normalized features.

    Recompute the single layer over the committed context and appended draft
    ancestors. The committed window stays fixed during this proposal round.
    This avoids depending on engine-specific tree/cache implementations.
    """
    draft = bundle.draft_model
    hidden = features.hidden_states.reshape(
        *features.hidden_states.shape[:-1], len(LAYERS), -1
    )
    hidden = hidden[..., [LAYERS.index(i) for i in EAGLE_LAYERS], :].flatten(-2)
    hidden = draft.project_hidden_states(hidden)
    ids = torch.cat((features.input_ids[:, 1:], anchor[:, None]), dim=1)
    committed = ids.shape[1]
    selected = []
    for _ in range(count):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        allowed = positions[None, :] <= positions[:, None]
        if draft.sliding_window is not None:
            # Ancestors append to the last committed query's window.
            lower = positions.clamp_max(committed - 1) - draft.sliding_window + 1
            allowed &= positions[None, :] >= lower[:, None]
        mask = torch.zeros(
            (1, 1, length, length), device=hidden.device, dtype=hidden.dtype
        )
        mask.masked_fill_(~allowed[None, None], torch.finfo(hidden.dtype).min)
        output = draft.backbone(
            input_embeds=draft.embed_input_ids(ids),
            hidden_states=hidden,
            cache_hidden=None,
            attention_mask=mask,
            position_ids=positions[None],
        )[:, -1:]
        scores, candidates = draft.candidate_logits(output)
        token = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        selected.append(token)
        hidden = torch.cat((hidden, output), dim=1)
        ids = torch.cat((ids, token), dim=1)
    return torch.cat(selected, dim=1)


@torch.inference_mode()
def candidate_coverage(draft, rows):
    """Measure dense-argmax retention on held-out teacher-forced root states."""
    positions = retained = agreed = 0
    for path in rows:
        row = torch.load(path, weights_only=True)
        hidden = (
            row["hidden_states"]
            .cuda()
            .reshape(*row["hidden_states"].shape[:-1], len(LAYERS), -1)
        )
        hidden = hidden[..., [LAYERS.index(i) for i in EAGLE_LAYERS], :].flatten(-2)
        ids = row["input_ids"].cuda()
        output = draft(hidden[:, :-1], draft.embed_input_ids(ids[:, 1:]))
        valid = row["loss_mask"][:, 1:].cuda().bool()
        output = output[valid]
        for chunk in output.split(16):
            dense_ids = draft.compute_logits(chunk).argmax(-1)
            scores, ids = draft.candidate_logits(chunk)
            retained += int((ids == dense_ids[:, None]).any(-1).sum())
            selected = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
            agreed += int((selected == dense_ids).sum())
            positions += len(chunk)
    return dict(
        scope="teacher-forced one-step supervised positions; not recurrent generation",
        positions=positions,
        candidates=draft.config.efficient_lm_head_config["num_candidates"],
        dense_argmax_retained=retained,
        dense_argmax_retention=retained / positions if positions else None,
        rescored_argmax_equal=agreed,
        rescored_argmax_agreement=agreed / positions if positions else None,
    )


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    memory_guard(args)
    metadata = json.loads((args.output / "capture.json").read_text())
    rows = sorted((args.output / "test_sft").glob("*.pt"))[: args.samples]
    if len(rows) != args.samples:
        raise ValueError("Not enough held-out captured samples")
    for path, identity in zip(rows, metadata["splits"]["test_sft"]):
        if digest(path) != identity["sha256"]:
            raise ValueError("Held-out capture identity changed")
    prompts = [torch.load(path, weights_only=True)["prompt_ids"][0] for path in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    verifier_dtype = (
        torch.float32 if args.verifier_dtype == "float32" else torch.bfloat16
    )
    torch.set_float32_matmul_precision("highest")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target,
            dtype=verifier_dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    teacher = OfflineTransformersCapture(target)
    teacher.set_capture_layers(LAYERS, capture_method="dspark")
    # The reference verifier always uses the frozen TARGET head, never a draft head.
    projection = SimpleNamespace(
        parameters=target.parameters, _compute_logits=target.lm_head
    )
    baseline_bundle = SimpleNamespace(model=projection)
    eos = target.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stops = [] if eos is None else eos if isinstance(eos, list) else [eos]
    config = dict(
        stop_token_ids=stops,
        max_new_tokens=args.max_new_tokens,
        draft_tokens=6,
        deadline=None,
    )
    destination = args.output / args.result_dir
    destination.mkdir(exist_ok=True)
    baseline = []
    for index, prompt in enumerate(prompts):
        result = _generate(teacher, baseline_bundle, prompt, config, speculative=False)
        baseline.append(result)
        print(
            json.dumps(
                {
                    "algorithm": "target",
                    "sample": index,
                    "tokens": result["generated_tokens"],
                }
            ),
            flush=True,
        )
    write_json(destination / "target.json", baseline)
    summaries = {}
    for name in NAMES:
        export = args.output / name / "export"
        draft = (
            AutoDraftModel.from_pretrained(export, torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        if name == "eagle4":
            bundle = SimpleNamespace(draft_model=draft, model=projection)
        else:
            cls = OnlineDSparkModel if name == "dspark" else OnlineDFlash2Model
            proposal_head = torch.nn.Linear(
                target.config.hidden_size, target.config.vocab_size, bias=False
            )
            proposal_head.weight = torch.nn.Parameter(
                target.lm_head.weight.detach().to(torch.bfloat16), requires_grad=False
            )
            proposal_embedding = torch.nn.Embedding.from_pretrained(
                target.get_input_embeddings().weight.detach().to(torch.bfloat16),
                freeze=True,
            )
            model = cls(
                draft_model=draft,
                target_lm_head=proposal_head,
                target_embed_tokens=proposal_embedding,
                mask_token_id=draft.mask_token_id,
                block_size=draft.block_size,
                attention_backend="sdpa",
            )
            draft_projection = model._compute_logits
            # The verifier uses the canonical target precision; proposal scoring stays BF16.
            model._compute_logits = lambda hidden: (
                target.lm_head(hidden)
                if hidden.dtype == verifier_dtype
                else draft_projection(hidden)
            )
            bundle = SimpleNamespace(draft_model=draft, model=model)

        def propose(current, features, anchor, count):
            features = replace(
                features, hidden_states=features.hidden_states.to(torch.bfloat16)
            )
            return (eagle4_propose if name == "eagle4" else _propose)(
                current, features, anchor, count
            )

        records = []
        for index, prompt in enumerate(prompts):
            result = _generate(
                teacher,
                bundle,
                prompt,
                config,
                speculative=True,
                proposal_fn=propose,
            )
            result["greedy_equal"] = result["token_ids"] == baseline[index]["token_ids"]
            records.append(result)
            print(
                json.dumps(
                    {
                        "algorithm": name,
                        "sample": index,
                        "accepted": result["accepted_tokens"],
                        "proposed": result["proposed_tokens"],
                        "greedy_equal": result["greedy_equal"],
                    }
                ),
                flush=True,
            )
        write_json(destination / f"{name}.json", records)
        accepted = sum(r["accepted_tokens"] for r in records)
        proposed = sum(r["proposed_tokens"] for r in records)
        tokens = sum(r["generated_tokens"] for r in records)
        rounds = sum(r["rounds"] for r in records)
        seconds = sum(r["seconds"] for r in records)
        summaries[name] = dict(
            samples=len(records),
            greedy_equal_samples=sum(r["greedy_equal"] for r in records),
            accepted_tokens=accepted,
            proposed_tokens=proposed,
            acceptance_rate=accepted / proposed if proposed else 0,
            committed_tokens_per_round=tokens / rounds,
            generated_tokens=tokens,
            rounds=rounds,
            reference_seconds=seconds,
            reference_tok_s=tokens / seconds,
            export_sha256={p.name: digest(p) for p in export.glob("*.safetensors")},
        )
        if name == "eagle4":
            summaries[name]["candidate_head"] = candidate_coverage(draft, rows)
        print(json.dumps(summaries[name]), flush=True)
        del bundle, draft
        if name != "eagle4":
            del model, proposal_head, proposal_embedding, draft_projection
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        destination / "summary.json",
        dict(
            verifier_dtype=args.verifier_dtype,
            decoder="full-prefix greedy chain reference; no optimized engine/tree policy",
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stops,
            draft_tokens=6,
            arms=summaries,
            target_reference_seconds=sum(r["seconds"] for r in baseline),
            heldout_capture_sha256=[digest(p) for p in rows],
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        ),
    )
    if any(r["greedy_equal_samples"] != len(prompts) for r in summaries.values()):
        raise RuntimeError(
            "At least one exported arm diverged from target greedy tokens"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--memory-mib", type=int, default=6500)
    parser.add_argument("--reserve-mib", type=int, default=1536)
    parser.add_argument(
        "--verifier-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--result-dir", default="evaluation")
    evaluate(parser.parse_args())
