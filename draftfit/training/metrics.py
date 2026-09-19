# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Stateless training telemetry reduction and host materialization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def _materialize_metrics(values: Dict[str, Any]) -> Dict[str, float]:
    """Materialize scalar metrics with at most one device-to-host transfer."""
    host_values: Dict[str, float] = {}
    tensor_names = []
    tensors = []
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"metric {name!r} must be scalar before materialization"
                )
            tensor_names.append(name)
            tensors.append(value.detach().float().reshape(()))
        else:
            host_values[name] = float(value)

    if tensors:
        device = tensors[0].device
        packed = torch.stack([tensor.to(device) for tensor in tensors])
        materialized = packed.cpu().tolist()
        host_values.update(
            {name: float(value) for name, value in zip(tensor_names, materialized)}
        )
    return host_values


def _dp_mean_scalars(
    values: Dict[str, Any],
    *,
    device: torch.device,
    process_group: Any = None,
) -> Dict[str, Any]:
    """Average scalar metrics across DP ranks with one collective.

    Uses the established DFlash metric convention (DP mean):
    without it the disagg consumer logs a single rank's local-batch accuracy
    (~1 rank x batch x anchors), which is ~sqrt(world) noisier and spikes because
    each rank's few round-robin refs can be all-easy or all-hard. Reducing across
    ranks recovers the ~world x larger effective sample the stock path logs.
    The caller invokes this only at an optimizer boundary: intermediate
    micro-step metrics are not logged, so synchronizing them only adds latency.
    """
    import torch.distributed as dist

    normalized = {
        name: (
            value.detach().float().reshape(())
            if isinstance(value, torch.Tensor)
            else float(value)
        )
        for name, value in values.items()
    }
    if not normalized or not (dist.is_available() and dist.is_initialized()):
        return normalized
    world = (
        dist.get_world_size()
        if process_group is None
        else dist.get_world_size(group=process_group)
    )
    if world <= 1:
        return normalized
    names = list(normalized)
    packed = torch.stack(
        [
            (
                value.to(device)
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=torch.float32, device=device)
            )
            for value in normalized.values()
        ]
    )
    if process_group is None:
        dist.all_reduce(packed)
    else:
        dist.all_reduce(packed, group=process_group)
    packed /= world
    return {name: packed[index] for index, name in enumerate(names)}


def _reduce_ratio_metrics(
    values: Dict[str, Any],
    *,
    device: torch.device,
    process_group: Any,
    reduce: bool,
) -> Dict[str, torch.Tensor]:
    """Form telemetry ratios only after summing their numerators and counts."""

    if not values:
        return {}
    normalized = []
    for name in sorted(values):
        pair = values[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(
                f"ratio metric {name!r} must be a (numerator, denominator) pair"
            )
        numerator = torch.as_tensor(pair[0]).detach().float().flatten().to(device)
        denominator = torch.as_tensor(pair[1]).detach().float().flatten().to(device)
        if numerator.shape != denominator.shape:
            raise ValueError(
                f"ratio metric {name!r} shape mismatch: "
                f"{tuple(numerator.shape)} vs {tuple(denominator.shape)}"
            )
        normalized.append((name, numerator, denominator))

    packed = torch.cat(
        [
            tensor
            for _, numerator, denominator in normalized
            for tensor in (numerator, denominator)
        ]
    )
    if reduce:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world = (
                dist.get_world_size()
                if process_group is None
                else dist.get_world_size(group=process_group)
            )
            if world > 1:
                if process_group is None:
                    dist.all_reduce(packed)
                else:
                    dist.all_reduce(packed, group=process_group)

    output: Dict[str, torch.Tensor] = {}
    cursor = 0
    for name, numerator, _denominator in normalized:
        width = numerator.numel()
        summed_numerator = packed[cursor : cursor + width]
        cursor += width
        summed_denominator = packed[cursor : cursor + width]
        cursor += width
        ratios = summed_numerator / summed_denominator.clamp_min(1e-12)
        if width == 1:
            output[name] = ratios.reshape(())
        else:
            output.update({f"{name}_{index}": ratios[index] for index in range(width)})
    return output


_EAGLE3_STRUCTURED_METRIC_KEYS = frozenset(
    {
        "acces",
        "acceptance_rates",
        "plosses",
        "acc_corrects",
        "acc_denoms",
        "metric_losses",
        "metric_loss_denoms",
    }
)


def _metric_vector(values: Any, *, device: torch.device, name: str) -> torch.Tensor:
    """Normalize one per-TTT metric sequence without losing its positions."""
    if isinstance(values, torch.Tensor):
        vector = values.detach().flatten()
    elif isinstance(values, (list, tuple)):
        vector = torch.stack(
            [torch.as_tensor(value).detach().reshape(()) for value in values]
        )
    else:
        raise TypeError(f"{name} must be a tensor or sequence, got {type(values)!r}")
    if vector.numel() == 0:
        raise ValueError(f"{name} must contain at least one TTT position")
    return vector.to(device=device, dtype=torch.float32)


def _reduce_eagle3_metrics(
    raw: Dict[str, Any],
    *,
    device: torch.device,
    process_group: Any,
    ploss_decay: float,
    reduce: bool,
) -> Optional[Dict[str, torch.Tensor]]:
    """Reduce EAGLE3's per-position training telemetry as numerators/counts.

    Accuracy and p-loss are ratios, so averaging rank-local ratios biases the
    result whenever ranks carry different token counts.  Pack every numerator
    and denominator into one collective and form ratios only after the global
    SUM.  Acceptance rate has no separate count in the model contract; weight
    it by the corresponding p-loss token count, matching the evaluator's
    batch-size-invariant convention.
    """
    required = {
        "acc_corrects",
        "acc_denoms",
        "metric_losses",
        "metric_loss_denoms",
    }
    if not required.issubset(raw):
        return None

    corrects = _metric_vector(raw["acc_corrects"], device=device, name="acc_corrects")
    acc_denoms = _metric_vector(raw["acc_denoms"], device=device, name="acc_denoms")
    losses = _metric_vector(raw["metric_losses"], device=device, name="metric_losses")
    loss_denoms = _metric_vector(
        raw["metric_loss_denoms"], device=device, name="metric_loss_denoms"
    )
    length = corrects.numel()
    vectors = {
        "acc_denoms": acc_denoms,
        "metric_losses": losses,
        "metric_loss_denoms": loss_denoms,
    }
    acceptance_rates = None
    if "acceptance_rates" in raw:
        acceptance_rates = _metric_vector(
            raw["acceptance_rates"], device=device, name="acceptance_rates"
        )
        vectors["acceptance_rates"] = acceptance_rates
    mismatched = {
        name: value.numel()
        for name, value in vectors.items()
        if value.numel() != length
    }
    if mismatched:
        raise ValueError(
            "EAGLE3 structured metric lengths must match acc_corrects "
            f"({length}); got {mismatched}"
        )

    # Rows: accuracy numerator/denominator, p-loss numerator/denominator,
    # acceptance numerator/denominator.  The last two rows stay zero when the
    # strategy does not expose acceptance telemetry.
    packed = torch.stack(
        (
            corrects,
            acc_denoms,
            losses * loss_denoms,
            loss_denoms,
            (
                acceptance_rates * loss_denoms
                if acceptance_rates is not None
                else torch.zeros_like(loss_denoms)
            ),
            (
                loss_denoms
                if acceptance_rates is not None
                else torch.zeros_like(loss_denoms)
            ),
        )
    )

    import torch.distributed as dist

    if reduce and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(process_group)
        if world > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=process_group)

    reduced_acc = packed[0] / packed[1].clamp_min(1e-6)
    reduced_ploss = packed[2] / packed[3].clamp_min(1e-6)
    result: Dict[str, torch.Tensor] = {}
    for index in range(length):
        result[f"acc_{index}"] = reduced_acc[index]
        result[f"ploss_{index}"] = reduced_ploss[index]

    result["acc"] = packed[0].sum().div(packed[1].sum().clamp_min(1e-6))
    weights = torch.tensor(
        [ploss_decay**index for index in range(length)],
        dtype=reduced_ploss.dtype,
        device=reduced_ploss.device,
    )
    result["loss"] = (reduced_ploss * weights).sum()

    if acceptance_rates is not None:
        reduced_acceptance = packed[4] / packed[5].clamp_min(1e-6)
        for index in range(length):
            result[f"acceptance_rate_{index}"] = reduced_acceptance[index]
        result["acceptance_rate"] = reduced_acceptance.mean()
    return result
