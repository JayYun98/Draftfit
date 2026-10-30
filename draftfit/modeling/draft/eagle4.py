"""Dense EAGLE-4 training model and exact candidate rescoring.

Implements the public EAGLE-4 model contract using Draftfit's existing EAGLE
backbone. No eagle-harness source is vendored. The optimized tree scheduler and
engine kernels remain the responsibility of the external eagle-harness runtime.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .llama3_eagle import LlamaForCausalLMEagle3
from .registry import register_draft


@register_draft
class LlamaForCausalLMEagle4(LlamaForCausalLMEagle3):
    """One recurrent dense layer, three target features and a candidate head."""

    def __init__(self, config, quant_config=None, attention_backend="sdpa"):
        if attention_backend != "sdpa":
            raise ValueError(
                "EAGLE4 training currently requires attention_backend=sdpa"
            )
        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE4 requires exactly one draft layer")
        if getattr(config, "draft_vocab_size", config.vocab_size) != config.vocab_size:
            raise ValueError("EAGLE4 requires the full target vocabulary")
        config.draft_vocab_size = config.vocab_size
        head = getattr(config, "efficient_lm_head_config", None)
        if not isinstance(head, dict) or head.get("type") != "low_rank":
            raise ValueError("EAGLE4 requires efficient_lm_head_config.type=low_rank")
        rank, candidates = head.get("rank"), head.get("num_candidates")
        for name, value, upper in (
            ("rank", rank, min(config.hidden_size, config.vocab_size)),
            ("num_candidates", candidates, config.vocab_size),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"EAGLE4 {name} must be an integer in [1, {upper}]")
        if getattr(config, "enable_moe", False) or getattr(config, "attn_norm", False):
            raise ValueError("EAGLE4 training supports dense concatenation fusion only")
        if getattr(config, "use_rope", True) is not True:
            raise ValueError("EAGLE4 training requires rotary positions")
        if any(
            getattr(config, name, False)
            for name in (
                "attention_bias",
                "mlp_bias",
                "bias",
                "enable_post_mlp_norm",
                "enable_2hs",
                "enable_defa",
            )
        ):
            raise ValueError(
                "EAGLE4 training does not implement biased or alternate draft blocks"
            )
        window = getattr(config, "sliding_window", None)
        enabled = getattr(config, "use_sliding_window", window is not None)
        if type(enabled) is not bool:
            raise ValueError("EAGLE4 use_sliding_window must be a boolean")
        if enabled and (type(window) is not int or window < 1):
            raise ValueError("EAGLE4 sliding_window must be a positive integer")
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_types != [
            "sliding_attention" if enabled else "full_attention"
        ]:
            raise ValueError("EAGLE4 layer_types conflicts with the attention window")
        layers = (getattr(config, "eagle_config", None) or {}).get(
            "eagle_aux_hidden_state_layer_ids"
        )
        if layers is not None and (len(layers) != 3 or len(set(layers)) != 3):
            raise ValueError("EAGLE4 requires three distinct target feature layers")
        self.sliding_window = window if enabled else None
        super().__init__(config, quant_config, attention_backend)
        # Identity vocabulary maps are training helpers, not EAGLE4 serving tensors.
        self._non_persistent_buffers_set.update({"t2d", "d2t"})
        self.efficient_lm_head_down_proj = nn.Linear(
            config.hidden_size, rank, bias=False
        )
        self.efficient_lm_head = nn.Linear(rank, config.vocab_size, bias=False)
        self._init_weights(self.efficient_lm_head_down_proj)
        self._init_weights(self.efficient_lm_head)
        self.set_training_stage("draft")

    def set_training_stage(self, stage):
        if stage not in {"draft", "candidates"}:
            raise ValueError("EAGLE4 stage must be draft or candidates")
        self.training_stage = stage
        for name, parameter in self.named_parameters():
            # A stage transition must not leave frozen gradients in clipping.
            parameter.grad = None
            candidate = name.startswith("efficient_lm_head")
            parameter.requires_grad_(
                candidate
                if stage == "candidates"
                else not candidate and name != "embed_tokens.weight"
            )

    def prepare_decoder_attention_mask(
        self,
        attention_mask,
        hidden_states,
        batch_size,
        seq_length,
        past_key_values_length,
    ):
        mask = super().prepare_decoder_attention_mask(
            attention_mask,
            hidden_states,
            batch_size,
            seq_length,
            past_key_values_length,
        )
        if self.sliding_window is not None:
            q = (
                torch.arange(seq_length, device=hidden_states.device)
                + past_key_values_length
            )
            k = torch.arange(
                seq_length + past_key_values_length, device=hidden_states.device
            )
            outside = k[None, :] < q[:, None] - self.sliding_window + 1
            mask = mask.masked_fill(
                outside[None, None], torch.finfo(hidden_states.dtype).min
            )
        return mask

    def backbone(self, *args, **kwargs):
        hidden = super().backbone(*args, **kwargs)
        return self.norm(hidden) if self.norm_output else hidden

    def forward(self, hidden_states, inputs_embeds, attention_mask=None, ttt_length=1):
        batch, length, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch, length), device=hidden_states.device)
        mask = self.prepare_decoder_attention_mask(
            attention_mask, hidden_states, batch, length, 0
        )
        return self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=self.project_hidden_states(hidden_states),
            cache_hidden=None if ttt_length == 1 else [[], []],
            attention_mask=mask,
            position_ids=torch.arange(length, device=hidden_states.device)[None],
        )

    def head_features(self, hidden):
        return hidden if self.norm_output else self.norm(hidden)

    def compute_logits(self, hidden_states):
        return self.lm_head(self.head_features(hidden_states))

    def candidate_scores(self, hidden_states):
        features = self.head_features(hidden_states)
        return self.efficient_lm_head(self.efficient_lm_head_down_proj(features))

    def candidate_logits(self, hidden_states):
        """Select with low-rank scores; rescore only selected full-head rows."""
        features = self.head_features(hidden_states)
        ids = (
            self.candidate_scores(hidden_states)
            .topk(self.config.efficient_lm_head_config["num_candidates"], dim=-1)
            .indices
        )
        logits = (
            (
                F.embedding(ids, self.lm_head.weight).float()
                * features.float().unsqueeze(-2)
            )
            .sum(-1)
            .to(features.dtype)
        )
        return logits, ids

    @torch.no_grad()
    def initialize_candidate_head(self, *, randomized=False):
        """SVD initialization; call after dense training, before candidate training."""
        rank = self.efficient_lm_head_down_proj.out_features
        weight = self.lm_head.weight.float()
        if randomized:
            u, s, v = torch.svd_lowrank(weight, q=rank, niter=4)
            vh = v.T
        else:
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        self.efficient_lm_head_down_proj.weight.copy_(vh[:rank])
        self.efficient_lm_head.weight.copy_(u[:, :rank] * s[:rank])

    def candidate_loss(self, hidden, dense_logits, position_mask):
        """Forward KL from the frozen dense draft, averaged on valid positions."""
        candidate = self.candidate_scores(hidden.detach()).float()
        target = dense_logits.detach().float().softmax(-1)
        per_token = F.kl_div(candidate.log_softmax(-1), target, reduction="none").sum(
            -1
        )
        mask = position_mask.squeeze(-1).to(per_token)
        return (per_token * mask).sum() / mask.sum().clamp_min(1)
