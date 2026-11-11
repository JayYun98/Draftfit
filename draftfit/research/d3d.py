# Adapted from D3D training/pruning/losses.py, MIT.
# Source revision cd7762436217fb51d7e7a310b310b72766bb816f.
# Copyright (c) 2026 Fusion Lab: Generative Vision Lab of Fudan University and SII.
# Copyright (c) 2026 D3D contributors. License retained in _vendor/D3D_LICENSE.

import hashlib
import json
import math
from numbers import Real

import torch
import torch.nn.functional as F


def _parameters(temperature, probability_margin):
    values = []
    for name, value in (
        ("temperature", temperature),
        ("probability_margin", probability_margin),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a finite real scalar") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        values.append(value)
    if values[1] >= 1:
        raise ValueError("probability_margin must be strictly between zero and one")
    return tuple(values)


def _work_dtype(*tensors):
    return (
        torch.float64
        if any(t.dtype == torch.float64 for t in tensors)
        else torch.float32
    )


def _validate_logits(logits, name):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.layout != torch.strided
        or not logits.is_floating_point()
        or logits.ndim not in (2, 3)
        or logits.numel() == 0
        or logits.shape[-1] < 2
        or logits.device.type == "meta"
    ):
        raise ValueError(
            f"{name} must be a nonempty floating 2D or 3D tensor with vocabulary >= 2"
        )
    if not torch.isfinite(logits.to(_work_dtype(logits))).all():
        raise ValueError(f"{name} must contain only finite logits")


def _validate_labels(logits, labels, active=None):
    if (
        not isinstance(labels, torch.Tensor)
        or labels.layout != torch.strided
        or labels.dtype != torch.long
        or labels.shape != logits.shape[:-1]
        or labels.device != logits.device
    ):
        raise ValueError(
            "labels must be torch.long with the logits' leading shape and device"
        )
    valid = (labels >= 0) & (labels < logits.shape[-1])
    if active is not None:
        valid = valid | (labels.eq(-100) & ~active)
    if not valid.all():
        raise ValueError(
            "labels must be vocabulary indices; only zero weights allow -100"
        )


def _scaled_logits(logits, temperature):
    scaled = logits / temperature
    if not torch.isfinite(scaled).all():
        raise ValueError("temperature-scaled logits must be representable and finite")
    return scaled


def _correct_probs(probs, labels, probability_margin):
    target = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    top_two = probs.topk(2, dim=-1)
    best_other = torch.where(
        top_two.indices[..., 0].eq(labels),
        top_two.values[..., 1],
        top_two.values[..., 0],
    )
    margin = target - best_other
    needs_correction = margin < probability_margin
    denominator = torch.where(needs_correction, 1 - margin, 1.0)
    correction = torch.where(
        needs_correction, (probability_margin - margin) / denominator, 0.0
    )
    corrected = probs * (1 - correction.unsqueeze(-1))
    return corrected.scatter_add(-1, labels.unsqueeze(-1), correction.unsqueeze(-1))


def corrected_teacher_probs(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    probability_margin: float = 0.01,
) -> torch.Tensor:
    """Return frozen teacher targets with the minimal label-margin correction.

    Logits have shape [positions, vocabulary] or [batch, positions, vocabulary].
    Every label must be a valid vocabulary index, including unused positions.
    """
    temperature, probability_margin = _parameters(temperature, probability_margin)
    _validate_logits(teacher_logits, "teacher_logits")
    _validate_labels(teacher_logits, labels)
    teacher = teacher_logits.detach().to(_work_dtype(teacher_logits))
    probs = _scaled_logits(teacher, temperature).softmax(dim=-1)
    return _correct_probs(probs, labels, probability_margin)


def corrected_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    probability_margin: float = 0.01,
) -> torch.Tensor:
    """Compute the weighted mean KL(corrected_teacher || student), scaled by T**2.

    Teacher logits and weights are fixed metadata. Zero-weight positions get no
    student gradient and may use the ignored label -100.
    """
    temperature, probability_margin = _parameters(temperature, probability_margin)
    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if (
        student_logits.shape != teacher_logits.shape
        or student_logits.device != teacher_logits.device
    ):
        raise ValueError(
            "student and teacher logits must have the same shape and device"
        )
    if (
        not isinstance(weights, torch.Tensor)
        or weights.layout != torch.strided
        or weights.is_complex()
        or weights.is_quantized
        or weights.shape != student_logits.shape[:-1]
        or weights.device != student_logits.device
    ):
        raise ValueError(
            "weights must be real with the logits' leading shape and device"
        )
    dtype = _work_dtype(student_logits, teacher_logits, weights)
    fixed_weights = weights.detach().to(dtype)
    if not torch.isfinite(fixed_weights).all() or (fixed_weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    weight_sum = fixed_weights.sum()
    if not torch.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a representable positive finite sum")
    active = fixed_weights > 0
    _validate_labels(student_logits, labels, active)
    student = _scaled_logits(student_logits.to(dtype), temperature)[active]
    teacher = _scaled_logits(teacher_logits.detach().to(dtype), temperature)[active]
    target = _correct_probs(teacher.softmax(dim=-1), labels[active], probability_margin)
    student_log_probs = F.log_softmax(student, dim=-1)
    student_log_probs = torch.where(target > 0, student_log_probs, 0.0)
    token_kl = (torch.special.xlogy(target, target) - target * student_log_probs).sum(
        dim=-1
    )
    loss = (token_kl * (fixed_weights[active] / weight_sum)).sum()
    loss = loss * temperature * temperature
    if not torch.isfinite(loss):
        raise ValueError("corrected KL loss must be representable and finite")
    return loss


def run_d3d_corrected_kl(config):
    """Evaluate JSON-lines logits on CPU without writing recipe artifacts."""
    answers = []
    digest = hashlib.sha256()
    options = {
        "temperature": config.loss.temperature,
        "probability_margin": config.loss.probability_margin,
    }
    with open(config.data.input_path, "rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            if index >= config.data.max_samples:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each JSON line must be an object")
                label_values = [row["labels"]]
                while label_values:
                    value = label_values.pop()
                    if isinstance(value, list):
                        label_values.extend(value)
                    elif type(value) is not int:
                        raise ValueError("JSON labels must contain only integers")
                with torch.enable_grad():
                    student = torch.tensor(
                        row["student_logits"],
                        dtype=torch.float64,
                        device="cpu",
                        requires_grad=True,
                    )
                    teacher = torch.tensor(
                        row["teacher_logits"], dtype=torch.float64, device="cpu"
                    )
                    labels = torch.tensor(row["labels"], device="cpu")
                    weights = torch.tensor(
                        row["weights"], dtype=torch.float64, device="cpu"
                    )
                    loss = corrected_kl_loss(
                        student, teacher, labels, weights, **options
                    )
                    loss.backward()
                gradient_norm = student.grad.norm()
                if not torch.isfinite(student.grad).all() or not torch.isfinite(
                    gradient_norm
                ):
                    raise ValueError("student gradient and its norm must be finite")
                ignored = labels.eq(-100)
                corrected = corrected_teacher_probs(
                    teacher, labels.masked_fill(ignored, 0), **options
                )
                corrected = torch.where(
                    ignored.unsqueeze(-1),
                    _scaled_logits(teacher, options["temperature"]).softmax(dim=-1),
                    corrected,
                )
                answers.append(
                    {
                        "id": row.get("id", index),
                        "kl_loss": float(loss.detach()),
                        "corrected_teacher_probs": corrected.tolist(),
                        "active_positions": int((weights > 0).sum()),
                        "student_gradient_norm": float(gradient_norm),
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
                    f"invalid D3D input at line {index + 1}: {exc}"
                ) from exc
    if not answers:
        raise ValueError("D3D corrected KL input is empty")
    return {
        "answers": answers,
        "implementation": "adapted_author_corrected_kl_component",
        "provenance": {
            "paper": "https://arxiv.org/html/2611.06425v1",
            "paper_component": "section 3.3 and Appendix A corrected forward KL",
            "source": (
                "https://github.com/fudan-generative-vision/D3D/blob/"
                "cd7762436217fb51d7e7a310b310b72766bb816f/training/pruning/losses.py"
            ),
            "source_revision": "cd7762436217fb51d7e7a310b310b72766bb816f",
            "license": "MIT",
            "input_sha256": digest.hexdigest(),
        },
        "input_kind": config.input_kind,
        "device": "cpu",
        "teacher_role": "fixed_full_depth_drafter",
        "scope": "corrected_forward_kl_component_only",
        "model_training": "not_implemented",
        "pruning": "not_implemented",
        "asynchronous_serving": "not_implemented",
        "throughput": "not-run",
        "real_model_quality": "not-run",
    }
