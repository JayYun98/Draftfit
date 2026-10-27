"""Conservative runtime admission for staged joint-training campaigns."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

STEP_METRIC = "step_with_capture_seconds"


def conservative_quantile(values, quantile=0.90):
    """Nearest-rank upper quantile for a nonempty set of positive durations."""
    if not values:
        raise ValueError("no measured duration samples")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile)
        or not 0 < quantile <= 1
    ):
        raise ValueError("quantile must be in (0, 1]")
    ordered = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("duration samples must be finite positive numbers")
        ordered.append(float(value))
    ordered.sort()
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def read_capture_profile(path, *, through_step=None):
    """Read one monitor run's end-to-end online-capture update timings.

    Optimizer-only ``joint_step_seconds`` is deliberately ignored: it omits
    target feature capture and is unsafe for forecasting online joint runs.
    """
    rows = []
    run_ids = set()
    steps = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid monitor JSON at line {line_number}") from exc
            if row.get("kind") != "capture":
                continue
            run_id = row.get("run_id")
            step = row.get("step")
            seconds = row.get(STEP_METRIC)
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"capture row {line_number} has no run_id")
            if type(step) is not int or step < 1:
                raise ValueError(f"capture row {line_number} has invalid step")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError(f"capture row {line_number} has invalid {STEP_METRIC}")
            if through_step is not None and step > through_step:
                raise ValueError(
                    f"capture step {step} exceeds completed profile step {through_step}"
                )
            if step in steps:
                raise ValueError(f"duplicate capture timing for step {step}")
            steps.add(step)
            run_ids.add(run_id)
            rows.append((step, float(seconds)))
    if len(run_ids) > 1:
        raise ValueError("runtime profile mixes monitor runs")
    return [seconds for _, seconds in sorted(rows)]


def read_joint_profile(path, *, through_step=None):
    """Read end-to-end update and milestone overhead observations from one run."""
    rows = []
    run_ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid monitor JSON at line {line_number}") from exc
            if row.get("kind") not in {
                "capture",
                "checkpoint",
                "evaluation_capture",
                "generation_verification",
            }:
                continue
            run_id = row.get("run_id")
            step = row.get("step")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"monitor row {line_number} has no run_id")
            if type(step) is not int or step < 0:
                raise ValueError(f"monitor row {line_number} has invalid step")
            if through_step is not None and step > through_step:
                raise ValueError(
                    f"monitor step {step} exceeds completed profile step {through_step}"
                )
            run_ids.add(run_id)
            rows.append(row)
    if len(run_ids) > 1:
        raise ValueError("runtime profile mixes monitor runs")

    captures = {}
    for row in rows:
        if row["kind"] == "capture":
            step = row["step"]
            if step in captures:
                raise ValueError(f"duplicate capture timing for step {step}")
            seconds = row.get(STEP_METRIC)
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError(f"capture row has invalid {STEP_METRIC}")
            captures[step] = (float(seconds), elapsed_seconds(row, "capture"))
    if not captures:
        raise ValueError("runtime profile has no capture-inclusive update timings")

    def grouped_elapsed(kind):
        result = {}
        for row in rows:
            if row["kind"] == kind:
                result.setdefault(row["step"], []).append(elapsed_seconds(row, kind))
        return result

    checkpoint_ends = grouped_elapsed("checkpoint")
    evaluation_ends = grouped_elapsed("evaluation_capture")
    evaluations = []
    for step, ends in evaluation_ends.items():
        starts = checkpoint_ends.get(step)
        if starts:
            duration = min(ends) - max(starts)
            if duration <= 0:
                # Initial evaluation is performed before the first checkpoint,
                # so those same-step events cannot be paired to measure its cost.
                # A normal update boundary must retain checkpoint-before-eval.
                if step in captures:
                    raise ValueError(
                        f"evaluation timing is not positive at step {step}"
                    )
                continue
            evaluations.append(duration)
    verifications = {}
    for row in rows:
        if row["kind"] != "generation_verification":
            continue
        seconds = row.get("verification_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("generation verification has invalid duration")
        verifications.setdefault(row["step"], []).append(float(seconds))
    return {
        "step_seconds": [value for _, (value, _) in sorted(captures.items())],
        "checkpoint_seconds": [
            max(values)
            for _, values in sorted(_group_checkpoints(rows, captures).items())
        ],
        "evaluation_seconds": evaluations,
        "verification_seconds": [
            max(values) for _, values in sorted(verifications.items())
        ],
    }


def _group_checkpoints(rows, captures):
    grouped = {}
    for row in rows:
        if row["kind"] == "checkpoint" and row["step"] in captures:
            grouped.setdefault(row["step"], []).append(
                elapsed_seconds(row, "checkpoint") - captures[row["step"]][1]
            )
    for step, durations in grouped.items():
        if any(value <= 0 for value in durations):
            raise ValueError(f"checkpoint timing is not positive at step {step}")
    return grouped


def elapsed_seconds(row, kind):
    value = row.get("elapsed_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{kind} row has invalid elapsed_seconds")
    return float(value)


def _event_steps(current_step, target_step, interval, *, include_final):
    steps = set()
    if interval:
        step = (current_step // interval + 1) * interval
        while step <= target_step:
            steps.add(step)
            step += interval
    if include_final and target_step > current_step:
        steps.add(target_step)
    return steps


def _checkpoint_steps(
    current_step, target_step, save_interval, evaluation_interval, verification_interval
):
    steps = (
        _event_steps(current_step, target_step, save_interval, include_final=False)
        | _event_steps(
            current_step, target_step, evaluation_interval, include_final=False
        )
        | _event_steps(
            current_step, target_step, verification_interval, include_final=False
        )
    )
    if target_step > current_step:
        steps.add(target_step)
    return steps


def estimate_runtime(
    *,
    samples,
    current_step,
    target_step,
    available_seconds,
    overhead_seconds,
    safety_factor=1.25,
    minimum_samples=5,
    checkpoint_interval=0,
    checkpoint_seconds=0,
    evaluation_interval=0,
    evaluation_seconds=0,
    verification_interval=0,
    verification_seconds=0,
):
    """Estimate feasibility from capture-inclusive update and milestone timings.

    ``overhead_seconds`` covers one-time work such as model reload, final export
    and recovery. Periodic/final checkpoint, evaluation and generation
    verification are counted separately from measured monitor events. The p90
    update quantile uses nearest rank; the safety factor adds margin to recurring work.
    """
    for name, value in (
        ("available_seconds", available_seconds),
        ("overhead_seconds", overhead_seconds),
        ("safety_factor", safety_factor),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    if type(current_step) is not int or current_step < 0:
        raise ValueError("current_step must be a nonnegative integer")
    if type(target_step) is not int or target_step < 0:
        raise ValueError("target_step must be a nonnegative integer")
    if type(minimum_samples) is not int or minimum_samples < 1:
        raise ValueError("minimum_samples must be a positive integer")
    intervals = (checkpoint_interval, evaluation_interval, verification_interval)
    if any(type(value) is not int or value < 0 for value in intervals):
        raise ValueError("milestone intervals must be nonnegative integers")
    operation_seconds = (checkpoint_seconds, evaluation_seconds, verification_seconds)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in operation_seconds
    ):
        raise ValueError("milestone operation times must be finite and nonnegative")
    values = []
    for value in samples:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("profile samples must be finite positive seconds")
        values.append(float(value))

    remaining = max(0, target_step - current_step)
    base = {
        "current_step": current_step,
        "target_step": target_step,
        "remaining_updates": remaining,
        "profile_samples": len(values),
        "metric": STEP_METRIC,
        "available_seconds": float(available_seconds),
        "overhead_seconds": float(overhead_seconds),
        "safety_factor": float(safety_factor),
        "minimum_samples": minimum_samples,
    }
    if remaining == 0:
        return {
            **base,
            "feasible": True,
            "reason": "target_already_reached",
            "p50_step_seconds": None,
            "p90_step_seconds": None,
            "checkpoint_events": 0,
            "evaluation_events": 0,
            "verification_events": 0,
            "measured_overhead_seconds": 0.0,
            "median_estimated_seconds": float(overhead_seconds),
            "guard_estimated_seconds": float(overhead_seconds),
            "max_feasible_step": current_step,
        }
    if len(values) < minimum_samples:
        return {
            **base,
            "feasible": False,
            "reason": "insufficient_profile",
            "p50_step_seconds": None,
            "p90_step_seconds": None,
            "median_estimated_seconds": None,
            "guard_estimated_seconds": None,
            "max_feasible_step": None,
        }

    ordered = sorted(values)
    p90 = conservative_quantile(ordered)
    p50 = statistics.median(ordered)
    checkpoint_count = len(
        _checkpoint_steps(
            current_step,
            target_step,
            checkpoint_interval,
            evaluation_interval,
            verification_interval,
        )
    )
    evaluation_steps = _event_steps(
        current_step, target_step, evaluation_interval, include_final=False
    ) | _event_steps(
        current_step, target_step, verification_interval, include_final=False
    )
    if remaining:
        evaluation_steps.add(target_step)
    verification_count = len(
        _event_steps(
            current_step,
            target_step,
            verification_interval,
            include_final=verification_interval > 0,
        )
    )
    evaluation_count = len(evaluation_steps)
    checkpoint_p50 = float(checkpoint_seconds)
    evaluation_p50 = float(evaluation_seconds)
    verification_p50 = float(verification_seconds)
    measured_overhead = (
        checkpoint_count * checkpoint_p50
        + evaluation_count * evaluation_p50
        + verification_count * verification_p50
    )
    median_seconds = overhead_seconds + remaining * p50 + measured_overhead
    guard_seconds = (
        overhead_seconds
        + remaining * p90 * safety_factor
        + measured_overhead * safety_factor
    )

    def guard_for(updates):
        candidate = current_step + updates
        checkpoint_events = len(
            _checkpoint_steps(
                current_step,
                candidate,
                checkpoint_interval,
                evaluation_interval,
                verification_interval,
            )
        )
        eval_steps = _event_steps(
            current_step, candidate, evaluation_interval, include_final=False
        ) | _event_steps(
            current_step, candidate, verification_interval, include_final=False
        )
        if updates:
            eval_steps.add(candidate)
        verify_events = len(
            _event_steps(
                current_step,
                candidate,
                verification_interval,
                include_final=verification_interval > 0,
            )
        )
        return (
            updates * p90 * safety_factor
            + checkpoint_events * checkpoint_p50 * safety_factor
            + len(eval_steps) * evaluation_p50 * safety_factor
            + verify_events * verification_p50 * safety_factor
        )

    low, high = 0, remaining
    while low < high:
        middle = (low + high + 1) // 2
        if overhead_seconds + guard_for(middle) <= available_seconds:
            low = middle
        else:
            high = middle - 1
    step_capacity = low
    feasible = guard_seconds <= available_seconds
    return {
        **base,
        "feasible": feasible,
        "reason": "within_runtime_budget" if feasible else "runtime_infeasible",
        "p50_step_seconds": p50,
        "p90_step_seconds": p90,
        "checkpoint_events": checkpoint_count,
        "evaluation_events": evaluation_count,
        "verification_events": verification_count,
        "measured_overhead_seconds": measured_overhead,
        "median_estimated_seconds": median_seconds,
        "guard_estimated_seconds": guard_seconds,
        "max_feasible_step": current_step + step_capacity,
    }
