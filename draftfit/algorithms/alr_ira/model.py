"""DFlash-compatible distillation over prepared ALR/IRA rollout features."""

import torch

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_sdpa_mask,
)

from .loss import harvest_target_topk, targetkd_loss


class ALRTrainingModel(OnlineDFlashModel):
    def __init__(self, *, topk=8, hard_label_mix=0.0, tail_bucket=True, **kwargs):
        super().__init__(**kwargs)
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError(
                "prepared ALR/IRA currently supports sdpa or eager attention"
            )
        if self.draft_model.sliding_window is not None:
            raise ValueError("prepared ALR/IRA currently requires full draft attention")
        self.topk = topk
        self.hard_label_mix = hard_label_mix
        self.tail_bucket = tail_bucket

    def forward(
        self, input_ids, hidden_states, loss_mask, target_last_hidden_states, alr_anchor
    ):
        batch, length = input_ids.shape
        anchors = alr_anchor.reshape(batch, 1).long()
        if torch.any(anchors < 0) or torch.any(anchors >= length - 1):
            raise ValueError("ALR anchor must precede a supervised rollout token")
        keep = torch.ones_like(anchors, dtype=torch.bool)
        noise = self._create_noise_embed(input_ids, anchors, keep)
        positions = torch.cat(
            [
                torch.arange(length, device=input_ids.device).expand(batch, -1),
                self._create_position_ids(anchors),
            ],
            dim=1,
        )
        mask = create_dflash_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=length,
            block_size=self.block_size,
            device=input_ids.device,
        )
        hidden = self.draft_model(
            position_ids=positions,
            noise_embedding=noise,
            target_hidden=hidden_states,
            attention_mask=mask,
        )
        logits = self._compute_logits(hidden).reshape(-1, self.lm_head.weight.shape[0])
        indices = anchors.unsqueeze(-1) + torch.arange(
            self.block_size, device=input_ids.device
        )
        valid = (indices < length) & (
            torch.arange(self.block_size, device=input_ids.device) > 0
        )
        safe = indices.clamp_max(length - 1)
        weights = (
            loss_mask.gather(1, safe.flatten(1)).reshape_as(indices).float() * valid
        )
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            offsets = torch.arange(self.block_size, device=input_ids.device)
            weights = weights * torch.exp(
                -(offsets - 1).clamp_min(0).float() / self.loss_decay_gamma
            )
        if not weights.any():
            raise ValueError("ALR batch has no supervised rollout slots")
        top_idx, top_prob, tail = harvest_target_topk(
            target_last_hidden_states,
            anchors,
            self.block_size,
            self.lm_head,
            min(self.topk, logits.shape[-1]),
        )
        loss = targetkd_loss(
            logits,
            top_idx,
            top_prob,
            tail,
            weights.flatten(),
            hard_label_mix=self.hard_label_mix,
            tail_bucket=self.tail_bucket,
        )
        denom = weights.sum() + 1e-6
        correct = ((logits.argmax(-1) == top_idx[:, 0]) * (weights.flatten() > 0)).sum()
        accuracy_den = (weights > 0).sum()
        return (
            loss,
            correct / accuracy_den.clamp_min(1),
            {
                "loss_terms": (loss * denom, denom.detach()),
                "ratio_metrics": {
                    "accuracy": (correct.detach(), accuracy_den.detach())
                },
            },
        )
