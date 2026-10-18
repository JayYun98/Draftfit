"""Deterministic resource advice; no execution, optimizer access or cost prediction.

``decide(config, observations, current=..., admission=..., state=...)`` consumes
JSON mappings. Config registers ordered ``presets`` with id, training (batch_size,
accumulation_steps, learning_rate), preflight_passed, pilot_steps, pilot_cost_usd
and pilot_seconds. Costs/times must be caller-supplied upper bounds, including
phase initialization/export. A smaller microbatch is the only downward candidate;
larger microbatches are exploratory candidates, never a promised speedup.

Current binds phase_id/preset/step and an optional boolean oom event. Observations
repeat that binding and contain peak_vram_bytes, capacity_bytes, utilization
(fraction, not percent), throughput (consistent units), loss. Steps are positive,
monotonic CAMPAIGN steps, including across weights-only phases. The caller passes
the recent observation window; windows never mix phases or capacities.
Capacity must be positive; allocator peak, utilization, throughput and loss may
be zero. Zero throughput is a valid no-progress measurement, never evidence for
batch growth; memory-pressure reduction can still proceed. Relative loss growth
from a zero baseline permits only zero loss.

Admission is external: allowed, phase_id, step, preset_ids, max_steps,
max_cost_usd, max_seconds. It must bind the current decision boundary. No fallback
authorizes measurement work: measure_only has zero steps/cost/time. The runner
must reserve and recheck admission immediately before starting any new phase.

State is REQUIRED: {transitions: 0, last_transition_step: None} initially.
Persist it with the campaign journal. Only after committing a new phase, atomically
replace it with returned transition_state; retain it on an unexecuted advisory.
Do not reset this state on exact resume or weights-only restart. Concurrent
dispatch/idempotency and mapping campaign steps to local trainer steps belong to
the runner. This module cannot enforce spend by an external executor.
"""

from __future__ import annotations

import math
from copy import deepcopy

SCHEMA = "draftfit-resource-policy-v1"
_DEFAULTS = dict(
    window_size=3,
    max_lag_steps=10,
    cooldown_steps=20,
    max_transitions=3,
    low_memory_ratio=0.65,
    high_memory_ratio=0.90,
    low_utilization=0.70,
    max_loss_growth=0.05,
    max_throughput_drop=0.10,
)
_METRICS = ("peak_vram_bytes", "capacity_bytes", "utilization", "throughput", "loss")


def _number(value, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0):
        raise ValueError("expected finite positive/nonnegative number")
    return value


def _integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("expected positive/nonnegative integer")
    return value


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected nonempty identifier")
    return value


def _config(config):
    cfg = {**_DEFAULTS, **deepcopy(config)}
    for key in ("window_size", "max_lag_steps", "cooldown_steps", "max_transitions"):
        _integer(cfg[key], positive=key == "window_size")
    if cfg["window_size"] < 2:
        raise ValueError("window_size must be at least two")
    for key in (
        "low_memory_ratio",
        "high_memory_ratio",
        "low_utilization",
        "max_loss_growth",
        "max_throughput_drop",
    ):
        if _number(cfg[key]) >= 1:
            raise ValueError(f"{key} must be below one")
    if not 0 < cfg["low_memory_ratio"] < cfg["high_memory_ratio"]:
        raise ValueError("memory thresholds require a nonempty hysteresis band")
    presets = cfg.get("presets", [])
    if not isinstance(presets, list):
        raise ValueError("presets must be a list")
    seen = set()
    for preset in presets:
        name = _name(preset["id"])
        if name in seen:
            raise ValueError("duplicate preset")
        seen.add(name)
        if type(preset["preflight_passed"]) is not bool:
            raise ValueError("preflight_passed must be boolean")
        if preset.get("optimizer_mode", "weights_only") != "weights_only":
            raise ValueError("preset must use weights_only")
        training = preset["training"]
        # This sidecar changes only the explicitly supported training fields.
        if set(training) != {"batch_size", "accumulation_steps", "learning_rate"}:
            raise ValueError(
                "preset needs batch_size, accumulation_steps, learning_rate only"
            )
        _integer(training["batch_size"], positive=True)
        _integer(training["accumulation_steps"], positive=True)
        _number(training["learning_rate"], positive=True)
        _integer(preset["pilot_steps"], positive=True)
        _number(preset["pilot_cost_usd"], positive=True)
        _number(preset["pilot_seconds"], positive=True)
    cfg["presets"] = presets
    return cfg


def decide(config, observations, *, current, admission, state):
    """Return measure_only or new_phase with a registered, bounded phase plan.

    Invalid policy/current/state raises ValueError (missing fields: KeyError).
    Bad observations are reported/rejected; any bad row within the active window
    prevents normal tuning. OOM is a separate current-bound event and may bypass
    the window and cooldown, but never admission or the transition limit.
    """
    cfg = _config(config)
    step = _integer(current["step"], positive=True)
    phase = _name(current["phase_id"])
    preset_id = _name(current["preset"])
    oom = current.get("oom", False)
    if type(oom) is not bool:
        raise ValueError("oom must be boolean")
    transitions = _integer(state["transitions"])
    last = state["last_transition_step"]
    if (transitions == 0 and last is not None) or (transitions > 0 and last is None):
        raise ValueError("inconsistent transition state")
    if last is not None and _integer(last, positive=True) > step:
        raise ValueError("state is ahead of current campaign step")
    result = dict(
        schema=SCHEMA,
        action="measure_only",
        reason=None,
        executed=False,
        optimizer_mode="preserve",
        optimizer_migration_allowed=False,
        new_phase_required=False,
        current=deepcopy(current),
        next_preset=None,
        selected_preset=None,
        next_steps=0,
        next_cost_usd=0.0,
        next_seconds=0.0,
        rejected=[],
        state=deepcopy(state),
        transition_state=None,
    )

    def finish(reason):
        result["reason"] = reason
        return result

    active = next((p for p in cfg["presets"] if p["id"] == preset_id), None)
    if active is None:
        return finish("current_preset_not_registered")
    if transitions >= cfg["max_transitions"]:
        return finish("transition_limit")

    # Fail closed on missing, stale, malformed or exhausted external admission.
    try:
        if (
            admission.get("allowed") is not True
            or admission["phase_id"] != phase
            or _integer(admission["step"], positive=True) != step
        ):
            raise ValueError("unbound admission")
        ids = admission["preset_ids"]
        if not isinstance(ids, list) or any(
            not isinstance(p, str) or not p for p in ids
        ):
            raise ValueError("invalid admitted presets")
        max_steps = _integer(admission["max_steps"])
        max_cost = _number(admission["max_cost_usd"])
        max_seconds = _number(admission["max_seconds"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return finish("admission_required")

    rows = {}
    invalid_window = False
    for item in observations:
        try:
            observed_step = _integer(item["step"], positive=True)
            if item["phase_id"] != phase or item["preset"] != preset_id:
                raise ValueError("phase_or_preset_mismatch")
            if observed_step > step or step - observed_step > cfg["max_lag_steps"]:
                raise ValueError("stale_or_future_observation")
            if last is not None and observed_step <= last:
                raise ValueError("observation_precedes_transition")
            row = {
                key: _number(item[key], positive=key == "capacity_bytes")
                for key in _METRICS
            }
            if row["utilization"] > 1 or row["peak_vram_bytes"] > row["capacity_bytes"]:
                raise ValueError("invalid_resource_ratio")
            row["step"] = observed_step
            if observed_step in rows and rows[observed_step] != row:
                raise ValueError("conflicting_observation")
            rows[observed_step] = row
        except (KeyError, TypeError, ValueError) as exc:
            result["rejected"].append(str(exc))
            invalid_window = True
    result["rejected"].sort()
    window = sorted(rows.values(), key=lambda row: row["step"])[-cfg["window_size"] :]
    if oom:
        direction, reason = "down", "oom"
    else:
        if invalid_window:
            return finish("invalid_observations")
        if len(window) < cfg["window_size"]:
            return finish("insufficient_window")
        if len({row["capacity_bytes"] for row in window}) != 1:
            return finish("capacity_changed")
        if last is not None and step - last < cfg["cooldown_steps"]:
            return finish("cooldown")
        ratios = [r["peak_vram_bytes"] / r["capacity_bytes"] for r in window]
        if all(r >= cfg["high_memory_ratio"] for r in ratios):
            direction, reason = "down", "memory_pressure"
        elif all(r <= cfg["low_memory_ratio"] for r in ratios) and all(
            r["utilization"] <= cfg["low_utilization"] for r in window
        ):
            # No inferred speedup. A stable observed workload only permits a pilot.
            if any(r["throughput"] == 0 for r in window):
                return finish("zero_throughput")
            first = window[0]
            if any(
                r["loss"] > first["loss"] * (1 + cfg["max_loss_growth"])
                or r["throughput"]
                < first["throughput"] * (1 - cfg["max_throughput_drop"])
                for r in window[1:]
            ):
                return finish("unstable_loss_or_throughput")
            direction, reason = "up", "underutilized_window"
        else:
            return finish("hysteresis_hold")

    batch = active["training"]["batch_size"]
    candidates = [
        p
        for p in cfg["presets"]
        if p["id"] != preset_id
        and p["preflight_passed"]
        and p["id"] in ids
        and (
            p["training"]["batch_size"] < batch
            if direction == "down"
            else p["training"]["batch_size"] > batch
        )
        and p["pilot_steps"] <= max_steps
        and p["pilot_cost_usd"] <= max_cost
        and p["pilot_seconds"] <= max_seconds
    ]
    if not candidates:
        return finish("no_admitted_candidate")
    # Closest registered microbatch, declaration order breaks ties deterministically.
    selected = min(candidates, key=lambda p: abs(p["training"]["batch_size"] - batch))
    result.update(
        action="new_phase",
        new_phase_required=True,
        optimizer_mode="weights_only",
        next_preset=selected["id"],
        selected_preset=deepcopy(selected),
        next_steps=selected["pilot_steps"],
        next_cost_usd=selected["pilot_cost_usd"],
        next_seconds=selected["pilot_seconds"],
        transition_state=dict(transitions=transitions + 1, last_transition_step=step),
    )
    return finish(reason)
