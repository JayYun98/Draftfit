"""Target-sampled REAL branch loss on proposal-only blocks.

Adapted from King-ty/speculators, Apache-2.0, commit
3dff4579c5f93f692108df8b5178c9e9200b8536, models/dflash/metrics.py and
losses/eager.py. Local changes define a standalone logits boundary, stopped
masks, precision validation, and a CPU recipe adapter. See NOTICE.md and
_vendor/REAL_LICENSE. This does not reproduce the full author training wrapper.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class REALResult:
    loss: torch.Tensor
    real_loss: torch.Tensor
    ce_loss: torch.Tensor
    acceptance_estimate: torch.Tensor
    overlap: torch.Tensor
    path_weights: torch.Tensor
    prefix_survival: torch.Tensor
    branch_terms: torch.Tensor
    active_mask: torch.Tensor


def _validate_logits(logits, name):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or logits.dtype
        not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or logits.ndim != 3
        or logits.numel() == 0
        or logits.device.type == "meta"
    ):
        raise ValueError(
            f"{name} must be nonempty floating [blocks, gamma, vocabulary] logits"
        )
    if not torch.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite logits")


def real_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    prediction_mask: torch.Tensor | None = None,
    real_weight: float = 0.9,
) -> REALResult:
    """Return REAL and teacher-argmax CE without running backward.

    Logits are aligned [blocks, gamma, vocabulary] tensors. IDs are torch.long
    [blocks, gamma]; an optional Boolean mask stops each block at its first
    false slot. Inactive IDs may be vocabulary indices or -100. The EAL loss
    averages nonempty blocks with their fixed gamma; CE averages retained slots
    without position decay. Targets are detached and prefix draft gradients are
    retained. Tiny survival weights can underflow in finite precision.
    """
    if isinstance(real_weight, bool) or not isinstance(real_weight, Real):
        raise ValueError("real_weight must be a finite real scalar in [0, 1]")
    try:
        real_weight = float(real_weight)
    except OverflowError as exc:
        raise ValueError("real_weight must be a finite real scalar in [0, 1]") from exc
    if not math.isfinite(real_weight) or not 0 <= real_weight <= 1:
        raise ValueError("real_weight must be a finite real scalar in [0, 1]")
    _validate_logits(draft_logits, "draft_logits")
    _validate_logits(target_logits, "target_logits")
    if (
        draft_logits.shape != target_logits.shape
        or draft_logits.device != target_logits.device
    ):
        raise ValueError("draft and target logits must have the same shape and device")
    if (
        not isinstance(target_token_ids, torch.Tensor)
        or target_token_ids.layout != torch.strided
        or target_token_ids.dtype != torch.long
        or target_token_ids.shape != draft_logits.shape[:-1]
        or target_token_ids.device != draft_logits.device
    ):
        raise ValueError(
            "target_token_ids must be torch.long with the logits' leading shape and device"
        )
    if prediction_mask is None:
        active = torch.ones_like(target_token_ids, dtype=torch.bool)
    else:
        if (
            not isinstance(prediction_mask, torch.Tensor)
            or prediction_mask.layout != torch.strided
            or prediction_mask.dtype != torch.bool
            or prediction_mask.shape != target_token_ids.shape
            or prediction_mask.device != draft_logits.device
        ):
            raise ValueError(
                "prediction_mask must be torch.bool with the IDs' shape and device"
            )
        active = prediction_mask.cumprod(dim=-1).bool()
    if not active.any():
        raise ValueError("REAL requires at least one active prediction")
    valid_ids = (target_token_ids >= 0) & (target_token_ids < draft_logits.shape[-1])
    if not (valid_ids | (target_token_ids.eq(-100) & ~active)).all():
        raise ValueError(
            "target_token_ids must index the vocabulary; only inactive suffixes allow -100"
        )
    safe_ids = target_token_ids.masked_fill(~active, 0).unsqueeze(-1)
    dtype = (
        torch.float64
        if torch.float64 in (draft_logits.dtype, target_logits.dtype)
        else torch.float32
    )
    log_q = F.log_softmax(draft_logits.to(dtype), dim=-1)
    log_p = F.log_softmax(target_logits.detach().to(dtype), dim=-1)
    if not torch.isfinite(log_q).all() or not torch.isfinite(log_p).all():
        raise ValueError("logit normalization must be representable and finite")
    log_ratio = (log_q.gather(-1, safe_ids) - log_p.gather(-1, safe_ids)).squeeze(-1)
    if not torch.isfinite(log_ratio).all():
        raise ValueError("log path ratios must be representable and finite")
    log_b = torch.where(active, log_ratio.clamp(max=0), 0.0)
    log_prev = torch.cat(
        (torch.zeros_like(log_b[:, :1]), log_b[:, :-1].cumsum(dim=-1)), dim=-1
    )
    if not torch.isfinite(log_prev).all():
        raise ValueError("log prefix survival must be representable and finite")
    overlap = torch.where(active, torch.minimum(log_q.exp(), log_p.exp()).sum(-1), 0.0)
    path_weights = log_b.exp()
    prefix_survival = torch.where(active, log_prev.exp(), 0.0)
    branch_terms = prefix_survival * overlap
    acceptance_estimate = branch_terms.sum(-1)
    eal_loss = (1 - acceptance_estimate[active.any(-1)] / draft_logits.shape[1]).mean()
    hard_labels = target_logits.detach().argmax(dim=-1, keepdim=True)
    ce = -log_q.gather(-1, hard_labels).squeeze(-1)[active].mean()
    loss = real_weight * eal_loss + (1 - real_weight) * ce
    result = REALResult(
        loss,
        eal_loss,
        ce,
        acceptance_estimate,
        overlap,
        path_weights,
        prefix_survival,
        branch_terms,
        active,
    )
    if any(
        not torch.isfinite(value).all()
        for value in (
            result.loss,
            result.real_loss,
            result.ce_loss,
            result.acceptance_estimate,
            result.overlap,
            result.path_weights,
            result.prefix_survival,
            result.branch_terms,
        )
    ):
        raise ValueError("REAL results must be representable and finite")
    return result


def _json_array(value, name, scalar_types):
    if not isinstance(value, list):
        raise ValueError(f"JSON {name} must be an array")
    pending = list(value)
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif type(item) not in scalar_types:
            raise ValueError(f"JSON {name} has an invalid scalar type")
    return value


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def run_real_loss(config):
    """Evaluate CPU float64 JSON-lines blocks and return a finite JSON report."""
    answers = []
    digest = hashlib.sha256()
    with open(config.data.input_path, "rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            if index >= config.data.max_samples:
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(row, dict):
                    raise ValueError("each JSON line must be an object")
                identifier = row.get("id", index)
                json.dumps(identifier, allow_nan=False)
                values = {
                    name: _json_array(row[name], name, (int, float))
                    for name in ("draft_logits", "target_logits")
                }
                ids = _json_array(row["target_token_ids"], "target_token_ids", (int,))
                mask = (
                    _json_array(row["prediction_mask"], "prediction_mask", (bool,))
                    if "prediction_mask" in row
                    else None
                )
                with torch.enable_grad():
                    draft = torch.tensor(
                        values["draft_logits"],
                        dtype=torch.float64,
                        device="cpu",
                        requires_grad=True,
                    )
                    target = torch.tensor(
                        values["target_logits"], dtype=torch.float64, device="cpu"
                    )
                    result = real_loss(
                        draft,
                        target,
                        torch.tensor(ids, dtype=torch.long, device="cpu"),
                        prediction_mask=(
                            torch.tensor(mask, dtype=torch.bool, device="cpu")
                            if mask is not None
                            else None
                        ),
                        real_weight=config.loss.real_weight,
                    )
                    result.loss.backward()
                norm = draft.grad.norm()
                if not torch.isfinite(draft.grad).all() or not torch.isfinite(norm):
                    raise ValueError("draft gradient and its norm must be finite")
                answers.append(
                    {
                        "id": identifier,
                        "loss": float(result.loss.detach()),
                        "real_loss": float(result.real_loss.detach()),
                        "ce_loss": float(result.ce_loss.detach()),
                        "acceptance_estimate": result.acceptance_estimate.detach().tolist(),
                        "mean_acceptance_estimate": float(
                            result.acceptance_estimate[result.active_mask.any(-1)]
                            .detach()
                            .mean()
                        ),
                        "overlap": result.overlap.detach().tolist(),
                        "path_weights": result.path_weights.detach().tolist(),
                        "prefix_survival": result.prefix_survival.detach().tolist(),
                        "branch_terms": result.branch_terms.detach().tolist(),
                        "active_mask": result.active_mask.tolist(),
                        "active_positions": int(result.active_mask.sum()),
                        "active_blocks": int(result.active_mask.any(-1).sum()),
                        "gamma": draft.shape[1],
                        "draft_gradient_norm": float(norm),
                    }
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
                OverflowError,
            ) as exc:
                raise ValueError(
                    f"invalid REAL input at line {index + 1}: {exc}"
                ) from exc
    if not answers:
        raise ValueError("REAL input is empty")
    return {
        "answers": answers,
        "implementation": "adapted_author_real_branch_component",
        "provenance": {
            "paper": "https://arxiv.org/abs/2611.14399v1",
            "source": (
                "https://github.com/King-ty/speculators/tree/"
                "3dff4579c5f93f692108df8b5178c9e9200b8536"
            ),
            "source_revision": "3dff4579c5f93f692108df8b5178c9e9200b8536",
            "license": "Apache-2.0",
            "input_sha256": digest.hexdigest(),
        },
        "input_kind": config.input_kind,
        "sampling": config.sampling,
        "sampling_provenance_verified": False,
        "device": "cpu",
        "dtype": "float64",
        "scope": "target_sampled_real_branch_component_only",
        "ce_target": "target_logits_argmax",
        "ce_position_decay": False,
        "author_wrapper_parity": False,
        "model_training": "not_implemented",
        "draft_sampled_d_real": "not_implemented",
        "gpu_parity": "not-run",
        "real_model_quality": "not-run",
        "export": "not-run",
        "serving_correctness": "not-run",
        "throughput": "not-run",
    }
