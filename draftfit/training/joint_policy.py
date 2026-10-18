"""Pure milestone policy; returns plans, never runs jobs or migrates optimizers.

An observation bundles a caller-verified immutable ``artifact`` binding, its
held-out ``loss``, and an optional ``serving`` report. Artifact fields are
artifact_sha256/family/preset/step and training/objective/capture fingerprints.
The report repeats the artifact key and binds a validation comparison
fingerprint covering target, runtime/precision, workload and serving settings.
Artifact hashing and report authenticity belong to the caller/verifier.

``config`` declares domains, comparison_fingerprint, revision, next_steps,
next_cost_usd, limits and ordered presets. ``budget`` supplies cumulative spent,
reserved costs and consumed steps/pilots. Persist returned evidence with the
caller-owned checkpoint/decision journal; no policy state or wall clock is hidden.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "draftfit-joint-policy-v1"
_CONTRACT = ("family", "preset", "training_fingerprint", "objective_fingerprint", "capture_fingerprint")
_TRIGGERS = {"loss_plateau", "loss_improving_acceptance_plateau",
             "acceptance_improving_speed_plateau", "below_serving_threshold"}


def _config(config):
    return {"patience": 3, "min_delta": .001, "acceptance_min_delta": .01,
            "speed_min_relative_gain": .01, "max_report_lag_steps": 0,
            "min_requests": 32, "min_domain_requests": 1,
            "min_speedup": 0., "min_domain_speedup": 0.,
            "require_serving": False, "presets": [], **config}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _observation_fingerprint(value):
    # Nonfinite incoming metrics may be rejected, but still need an audit hash.
    # They are never copied into normalized evidence or a decision's metrics.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=True).encode()).hexdigest()


def _number(value, *, positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < 0
            or (positive and value == 0)):
        raise ValueError("expected finite nonnegative number" if not positive else "expected positive number")
    return value


def _integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("expected nonnegative integer" if not positive else "expected positive integer")
    return value


def observation_key(artifact) -> tuple:
    """Validate and return the immutable artifact/family/preset/step key."""
    sha = artifact["artifact_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid artifact_sha256")
    for name in _CONTRACT:
        if not isinstance(artifact[name], str) or not artifact[name]:
            raise ValueError(f"missing {name}")
    return sha, artifact["family"], artifact["preset"], _integer(artifact["step"])


def _trend(values, delta, *, minimize=False, relative=False):
    baseline, stale, improved = values[0], 0, True
    for value in values[1:]:
        threshold = delta * baseline if relative else delta
        gain = baseline - value if minimize else value - baseline
        improved = gain > threshold
        if improved:
            baseline, stale = value, 0
        else:
            stale += 1
    return {"meaningful_best": baseline, "stale": stale, "improved": improved,
            "best": min(values) if minimize else max(values)}


def _validate(config, budget):
    if config.get("schema") != SCHEMA or not isinstance(config.get("revision"), str) or not config["revision"]:
        raise ValueError("invalid policy schema/revision")
    domains = config["domains"]
    if (not isinstance(domains, list) or not domains or any(not isinstance(d, str) or not d for d in domains)
            or len(set(domains)) != len(domains)
            or not isinstance(config["comparison_fingerprint"], str) or not config["comparison_fingerprint"]):
        raise ValueError("declare unique domains and comparison_fingerprint")
    for name in ("patience", "next_steps", "min_requests", "min_domain_requests"):
        _integer(config[name], positive=True)
    _integer(config["max_report_lag_steps"])
    for name in ("min_delta", "acceptance_min_delta", "speed_min_relative_gain", "min_speedup", "min_domain_speedup"):
        _number(config[name])
    _number(config["next_cost_usd"], positive=True)
    if type(config["require_serving"]) is not bool:
        raise ValueError("require_serving must be boolean")
    limits = config["limits"]
    for name in ("max_usd", "reserve_usd", "max_pilot_usd", "max_exploration_usd"):
        _number(limits[name], positive=name == "max_usd")
    for name in ("max_steps", "max_pilot_steps", "max_pilots", "max_unvalidated_steps"):
        _integer(limits[name])
    if limits["reserve_usd"] >= limits["max_usd"]:
        raise ValueError("no budget after reserve")
    for name in ("spent_usd", "reserved_usd", "exploration_spent_usd"):
        _number(budget.get(name, 0))
    for name in ("steps_used", "pilots_used", "unvalidated_steps"):
        _integer(budget.get(name, 0))
    seen = set()
    for preset in config["presets"]:
        for name in ("id", "family", "training_fingerprint", "objective_fingerprint", "capture_fingerprint", "reason"):
            if not isinstance(preset.get(name), str) or not preset[name]:
                raise ValueError(f"preset needs {name}")
        if preset["id"] in seen:
            raise ValueError("duplicate preset")
        seen.add(preset["id"])
        if (not preset.get("when") or set(preset["when"]) - _TRIGGERS
                or type(preset.get("preflight_passed")) is not bool):
            raise ValueError("preset needs declared triggers and preflight result")
        if preset.get("optimizer_mode", "weights_only") != "weights_only":
            raise ValueError("branch optimizer migration is unsupported; use weights_only")
        _integer(preset["pilot_steps"], positive=True)
        _number(preset["pilot_cost_usd"], positive=True)


def _serving(report, artifact, config):
    # Same matched workload/parity/count criteria as campaign.evaluate_report,
    # without that scheduler's filesystem/result.json dependency.
    if tuple(report.get(k) for k in ("artifact_sha256", "family", "preset", "step")) != observation_key(artifact):
        raise ValueError("artifact_binding_mismatch")
    _integer(report["step"])
    if (report.get("comparison_fingerprint") != config["comparison_fingerprint"]
            or report.get("split") != "validation"):
        raise ValueError("comparison_fingerprint_or_split_mismatch")
    if report.get("exact_token_parity") is not True:
        raise ValueError("parity_failure")
    if type(report.get("speed_decisions_allowed", True)) is not bool:
        raise ValueError("invalid_speed_decisions_allowed")
    domains = report["domains"]
    names = [d["name"] for d in domains]
    if len(names) != len(set(names)) or set(names) != set(config["domains"]):
        raise ValueError("domain_completeness")
    total_seconds = base_seconds = tokens = requests = accepted = proposed = 0
    ratios = {}
    for domain in domains:
        n = _integer(domain["requests"], positive=True)
        if (type(domain.get("matched_requests")) is not int or domain["matched_requests"] != n
                or domain.get("exact_token_parity") is not True):
            raise ValueError("unmatched_requests_or_domain_parity")
        if n < config["min_domain_requests"]:
            raise ValueError("insufficient_domain_requests")
        count = _integer(domain["output_tokens"], positive=True)
        if type(domain.get("baseline_output_tokens")) is not int or count != domain["baseline_output_tokens"]:
            raise ValueError("output_token_mismatch")
        seconds = _number(domain["seconds"], positive=True)
        baseline = _number(domain["baseline_seconds"], positive=True)
        a, p = _integer(domain["accepted_tokens"]), _integer(domain["proposed_tokens"], positive=True)
        if a > p:
            raise ValueError("invalid_acceptance_counts")
        accepted, proposed = accepted + a, proposed + p
        requests, tokens = requests + n, tokens + count
        total_seconds, base_seconds = total_seconds + seconds, base_seconds + baseline
        ratios[domain["name"]] = baseline / seconds
    if requests < config["min_requests"]:
        raise ValueError("insufficient_requests")
    _number(total_seconds, positive=True)
    _number(base_seconds, positive=True)
    speedup = _number(base_seconds / total_seconds, positive=True)
    tok_s = _number(tokens / total_seconds, positive=True)
    for value in ratios.values():
        _number(value, positive=True)
    return {"speedup": speedup, "tok_s": tok_s,
            "speed_decisions_allowed": report.get("speed_decisions_allowed", True),
            "acceptance": accepted / proposed, "accepted_tokens": accepted,
            "proposed_tokens": proposed, "requests": requests,
            "output_tokens": tokens, "seconds": total_seconds, "baseline_seconds": base_seconds,
            "domains": dict(sorted(ratios.items())), "report_sha256": fingerprint(report)}


def decide(config, observations, *, current, budget, tried_presets=()):
    """Return continue/stop/branch, best bindings and bounded, unexecuted options.

    The caller supplies verified artifact manifests in each observation and the
    current manifest. Reports never identify the current weights by step alone.
    Only the current contract's history contributes to patience. Delayed valid
    reports remain historical evidence and cannot promote the current artifact.
    Family/objective/freeze/batch changes require a new weights-only phase.
    """
    config = _config(config)
    _validate(config, budget)
    key = observation_key(current)
    if len(set(tried_presets)) != len(tried_presets) or any(not isinstance(p, str) for p in tried_presets):
        raise ValueError("tried_presets must be unique preset IDs")
    records, rejected, seen = [], [], {}
    fatal_parity = False
    for item in observations:
        binding = None
        try:
            artifact = item["artifact"]
            binding = observation_key(artifact)
            if tuple(artifact[k] for k in _CONTRACT) != tuple(current[k] for k in _CONTRACT):
                raise ValueError("training_objective_capture_or_preset_fingerprint_mismatch")
            if artifact["step"] > current["step"]:
                raise ValueError("future_observation")
            if artifact["step"] == current["step"] and binding != key:
                raise ValueError("current_artifact_mismatch")
            loss = _number(item["loss"])
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"key": list(binding) if binding else None, "reason": str(exc)})
            continue
        step = artifact["step"]
        if step in seen:
            if _observation_fingerprint(item) != seen[step]:
                raise ValueError("conflicting observations for the same contract/step")
            continue
        seen[step] = _observation_fingerprint(item)
        record = {"artifact": dict(artifact), "loss": loss, "serving": None}
        if item.get("serving") is not None:
            try:
                record["serving"] = _serving(item["serving"], artifact, config)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                reason = str(exc)
                rejected.append({"key": list(binding), "reason": reason})
                fatal_parity |= binding == key and reason in {"parity_failure", "unmatched_requests_or_domain_parity"}
        records.append(record)
    records.sort(key=lambda row: row["artifact"]["step"])
    rejected.sort(key=lambda row: fingerprint(row))
    evidence = {"config": config, "current": current, "budget": budget,
                "observations": sorted(observations, key=_observation_fingerprint), "tried_presets": sorted(tried_presets)}
    result = {"schema": SCHEMA, "policy_revision": config["revision"],
              "policy_fingerprint": fingerprint(config), "input_fingerprint": _observation_fingerprint(evidence),
              "action": "stop", "reason": None, "executed": False, "current": dict(current),
              "optimizer_migration_allowed": False, "optimizer_mode": "preserve",
              "next_preset": None, "allowed_presets": [], "excluded_presets": [],
              "rejected": rejected, "best": {}, "plateau": {}, "next_steps": 0,
              "next_cost_usd": 0., "current_serving_validated": False}

    def finish(reason, action="stop"):
        result.update(action=action, reason=reason)
        return result

    remaining = config["limits"]["max_usd"] - config["limits"]["reserve_usd"] - budget.get("spent_usd", 0) - budget.get("reserved_usd", 0)
    remaining_steps = config["limits"]["max_steps"] - budget.get("steps_used", 0)
    if not records or observation_key(records[-1]["artifact"]) != key:
        return finish("current_loss_required")
    loss_trend = _trend([r["loss"] for r in records], config["min_delta"], minimize=True)
    result["plateau"]["loss"] = loss_trend
    best_loss = min(records, key=lambda r: (r["loss"], r["artifact"]["step"]))
    result["best"]["loss"] = {"artifact": best_loss["artifact"], "value": best_loss["loss"]}
    serving = [r for r in records if r["serving"] is not None]
    fresh = bool(serving and current["step"] - serving[-1]["artifact"]["step"] <= config["max_report_lag_steps"])
    result["current_serving_validated"] = bool(serving and observation_key(serving[-1]["artifact"]) == key)
    for metric, delta, relative in (("acceptance", config["acceptance_min_delta"], False),
                                    ("speedup", config["speed_min_relative_gain"], True)):
        measured = [r for r in serving if metric != "speedup" or r["serving"]["speed_decisions_allowed"]]
        if measured:
            name = "speed" if metric == "speedup" else metric
            result["plateau"][name] = _trend([r["serving"][metric] for r in measured], delta, relative=relative)
            best = max(measured, key=lambda r: (r["serving"][metric], -r["artifact"]["step"]))
            result["best"][name] = {"artifact": best["artifact"], "value": best["serving"][metric],
                                   "report_sha256": best["serving"]["report_sha256"]}
    if serving and not fresh:
        result["rejected"].append({"key": list(observation_key(serving[-1]["artifact"])), "reason": "late_report_for_current"})
    if fatal_parity:
        return finish("current_parity_failure")
    if remaining <= 0 or remaining_steps <= 0:
        return finish("cumulative_budget_exhausted")
    trigger = "loss_plateau" if loss_trend["stale"] >= config["patience"] else None
    if fresh:
        latest = serving[-1]["serving"]
        if latest["speed_decisions_allowed"] and (latest["speedup"] < config["min_speedup"] or min(latest["domains"].values()) < config["min_domain_speedup"]):
            trigger = "below_serving_threshold"
        elif latest["speed_decisions_allowed"] and result["plateau"]["speed"]["stale"] >= config["patience"] and result["plateau"]["acceptance"]["improved"]:
            trigger = "acceptance_improving_speed_plateau"
        elif result["plateau"]["acceptance"]["stale"] >= config["patience"] and loss_trend["improved"]:
            trigger = "loss_improving_acceptance_plateau"
    if trigger:
        limits = config["limits"]
        for preset in config["presets"]:
            reason = None
            if trigger not in preset["when"]:
                reason = "trigger_not_declared"
            elif not preset["preflight_passed"]:
                reason = "preflight_not_passed"
            elif preset["id"] == current["preset"] or preset["id"] in tried_presets:
                reason = "already_selected"
            elif preset["capture_fingerprint"] != current["capture_fingerprint"]:
                reason = "incompatible_capture"
            elif budget.get("pilots_used", 0) >= limits["max_pilots"]:
                reason = "pilot_count_bound"
            elif preset["pilot_steps"] > min(limits["max_pilot_steps"], remaining_steps):
                reason = "pilot_step_bound"
            elif preset["pilot_cost_usd"] > min(limits["max_pilot_usd"], remaining):
                reason = "pilot_cost_bound"
            elif preset["pilot_cost_usd"] + budget.get("exploration_spent_usd", 0) > limits["max_exploration_usd"]:
                reason = "exploration_budget_bound"
            if reason:
                result["excluded_presets"].append({"id": preset["id"], "reason": reason})
            else:
                result["allowed_presets"].append(preset["id"])
        if result["allowed_presets"]:
            selected = next(p for p in config["presets"] if p["id"] == result["allowed_presets"][0])
            # Only a fresh serving-driven trigger selects an older measured artifact.
            source = serving[-1]["artifact"] if fresh and trigger != "loss_plateau" else current
            result.update(next_preset=selected["id"], optimizer_mode="weights_only",
                          source_artifact=dict(source), new_phase_required=True,
                          selection_reason=selected["reason"], selected_preset=dict(selected),
                          next_steps=selected["pilot_steps"], next_cost_usd=selected["pilot_cost_usd"])
            return finish(trigger, "branch")
        return finish(trigger)
    steps = min(config["next_steps"], remaining_steps)
    if config["require_serving"] and not fresh:
        steps = min(steps, config["limits"]["max_unvalidated_steps"] - budget.get("unvalidated_steps", 0))
        if steps <= 0:
            return finish("fresh_serving_required")
    if config["next_cost_usd"] > remaining:
        return finish("next_interval_exceeds_budget")
    result.update(next_steps=steps, next_cost_usd=config["next_cost_usd"])
    return finish("measured_continue" if result["current_serving_validated"] else "loss_only_not_serving_validated", "continue")


def execute_registered_phase(config, decision, *, budget, artifact_sha256,
                             executor, tried_presets=()):
    """Admit one registered weights-only pilot and invoke the caller's executor.

    Call at a durable checkpoint boundary. ``artifact_sha256`` is the verified
    source's hash, not the live trainer's newer step. The executor implements a
    NEW phase with a NEW optimizer, returns actual steps/cost and initialization
    evidence, and owns checkpointing/error recovery. Reserve the returned plan's
    upper cost before dispatch; account actual spend even if the executor fails.
    No cloud/CLI executor is supplied by this module.
    """
    config = _config(config)
    _validate(config, budget)
    if (decision.get("action") != "branch" or decision.get("executed") is not False
            or decision.get("optimizer_mode") != "weights_only"
            or decision.get("optimizer_migration_allowed") is not False
            or decision.get("policy_fingerprint") != fingerprint(config)):
        raise ValueError("not an unexecuted registered weights-only decision")
    preset = next((p for p in config["presets"] if p["id"] == decision["next_preset"]), None)
    if (preset is None or not preset["preflight_passed"]
            or preset["id"] in tried_presets or decision["reason"] not in preset["when"]
            or fingerprint(preset) != fingerprint(decision["selected_preset"])):
        raise ValueError("preset changed, unapproved or already tried")
    source = decision["source_artifact"]
    observation_key(source)
    if artifact_sha256 != source["artifact_sha256"]:
        raise ValueError("source artifact hash changed")
    if preset["capture_fingerprint"] != source["capture_fingerprint"]:
        raise ValueError("incompatible capture")
    current = decision["current"]
    if (tuple(source[k] for k in _CONTRACT) != tuple(current[k] for k in _CONTRACT)
            or not 0 <= current["step"] - source["step"] <= config["max_report_lag_steps"]
            or (decision["reason"] == "loss_plateau" and observation_key(source) != observation_key(current))):
        raise ValueError("phase source no longer binds to the decision")
    limits = config["limits"]
    cost, steps = preset["pilot_cost_usd"], preset["pilot_steps"]
    if (budget.get("pilots_used", 0) >= limits["max_pilots"]
            or steps > min(limits["max_pilot_steps"], limits["max_steps"] - budget.get("steps_used", 0))
            or cost > min(limits["max_pilot_usd"], limits["max_usd"] - limits["reserve_usd"]
                          - budget.get("spent_usd", 0) - budget.get("reserved_usd", 0))
            or cost + budget.get("exploration_spent_usd", 0) > limits["max_exploration_usd"]):
        raise ValueError("pilot no longer fits cumulative/exploration bounds")
    plan = {"preset": dict(preset), "source_artifact": dict(source),
            "optimizer_mode": "weights_only", "new_phase_required": True,
            "max_steps": steps, "max_cost_usd": cost}
    outcome = executor(plan)
    if (outcome.get("initialization") != "weights_only" or outcome.get("optimizer_migrated") is not False
            or _integer(outcome["steps"]) > steps or _number(outcome["cost_usd"]) > cost):
        raise ValueError("phase executor violated weights-only/pilot bounds")
    return {"schema": SCHEMA, "executed": True, "preset": preset["id"],
            "source_artifact": dict(source), "outcome": outcome,
            "policy_fingerprint": decision["policy_fingerprint"]}
