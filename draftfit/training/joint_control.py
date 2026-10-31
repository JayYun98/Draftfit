"""Checkpoint-bound milestone admission using actual paired generation reports.

Stops at a saved joint boundary. A branch is an explicit weights-only handoff;
this observer never mutates family membership or migrates optimizer state.
"""

from __future__ import annotations

import time
from copy import deepcopy

from draftfit.training.joint_policy import decide, fingerprint


def training_fingerprint(training):
    """Keep unrelated objective defaults out of existing policy histories."""
    values = training.model_dump(mode="json")
    if values.get("strategy") != "alr_ira":
        for field in ("alr_topk", "alr_hard_label_mix", "alr_tail_bucket"):
            values.pop(field, None)
    return fingerprint(values)


def admit_initial(settings, budget, start_step, stop_step, interval):
    """Bound the first work interval before any optimizer update."""
    from draftfit.training.joint_policy import _config, _validate

    policy = _config(
        {**settings["policy"], "comparison_fingerprint": "pending-generation-binding"}
    )
    _validate(policy, budget)
    timestamp = budget.get("timestamp_unix")
    if (
        not isinstance(timestamp, (int, float))
        or not 0 <= time.time() - timestamp <= 120
    ):
        raise ValueError("joint_control requires a fresh measured budget observation")
    remaining = max(
        0, policy["limits"]["max_steps"] - max(start_step, budget.get("steps_used", 0))
    )
    bound = min(stop_step, start_step + remaining)
    money = (
        policy["limits"]["max_usd"]
        - policy["limits"]["reserve_usd"]
        - budget.get("spent_usd", 0)
        - budget.get("reserved_usd", 0)
    )
    if money < policy["next_cost_usd"]:
        bound = start_step
    return bound, start_step + min(
        interval, policy["next_steps"], max(0, bound - start_step)
    )


def observe_generation(report, metrics, stages, settings, history, budget):
    if (
        report["production_engine_speed"] is not False
        or report["schema"] != "draftfit-joint-generation-v1"
    ):
        raise ValueError("unsupported generation report contract")
    timestamp = budget.get("timestamp_unix")
    if (
        not isinstance(timestamp, (int, float))
        or not 0 <= time.time() - timestamp <= 120
    ):
        raise ValueError("joint_control requires a fresh measured budget observation")
    state = deepcopy(history)
    bindings = deepcopy(report["bindings"])
    bindings["config"].pop("deadline_unix", None)
    comparison = fingerprint(bindings)
    policy = {**settings["policy"], "comparison_fingerprint": comparison}
    policy_domains = policy.get("domains")
    report_domains = report["domains"]
    if (
        not isinstance(policy_domains, list)
        or not isinstance(report_domains, list)
        or any(not isinstance(d, str) or not d for d in policy_domains + report_domains)
        or len(set(policy_domains)) != len(policy_domains)
        or len(set(report_domains)) != len(report_domains)
        or sorted(policy_domains) != sorted(report_domains)
    ):
        raise ValueError("policy domains differ from held-out generation domains")
    decisions = {}
    baseline = {(r["trial"], r["row_index"]): r for r in report["baseline"]["runs"]}
    candidates = {c["name"]: c for c in report["candidates"]}
    if set(candidates) != {s.name for s in stages}:
        raise ValueError("generation family set differs from current joint phase")
    for stage in stages:
        candidate = candidates[stage.name]
        artifact = dict(
            artifact_sha256=report["artifact_sha256"],
            step=report["step"],
            family=stage.config.training.strategy,
            preset=settings.get("preset", "default"),
            training_fingerprint=training_fingerprint(stage.config.training),
            objective_fingerprint=fingerprint(
                {
                    "strategy": stage.config.training.strategy,
                    "draft": stage.draft_config.to_dict(),
                }
            ),
            capture_fingerprint=report["capture_fingerprint"],
        )
        domains = []
        rows = candidate["runs"]
        if len(rows) != len(baseline) or len(
            {(r["trial"], r["row_index"]) for r in rows}
        ) != len(rows):
            raise ValueError("unmatched or duplicate generation requests")
        for name in report["domains"]:
            group = [
                r
                for r in rows
                if report["bindings"]["prompts"][r["row_index"]]["domain"] == name
            ]
            paired = [baseline[(r["trial"], r["row_index"])] for r in group]
            domains.append(
                dict(
                    name=name,
                    requests=len(group),
                    matched_requests=len(group),
                    exact_token_parity=all(
                        r["token_ids"] == b["token_ids"] for r, b in zip(group, paired)
                    ),
                    output_tokens=sum(r["generated_tokens"] for r in group),
                    baseline_output_tokens=sum(r["generated_tokens"] for r in paired),
                    seconds=sum(r["seconds"] for r in group),
                    baseline_seconds=sum(r["seconds"] for r in paired),
                    accepted_tokens=sum(r["accepted_tokens"] for r in group),
                    proposed_tokens=sum(r["proposed_tokens"] for r in group),
                )
            )
        serving = {
            **{k: artifact[k] for k in ("artifact_sha256", "family", "preset", "step")},
            "comparison_fingerprint": comparison,
            "split": "validation",
            "exact_token_parity": candidate["tokenparity"],
            "domains": domains,
            "speed_decisions_allowed": settings.get("allow_reference_timings", False),
        }
        loss = metrics[stage.name]["eval/avg_loss"]
        observations = state.setdefault(stage.name, [])
        observations.append(dict(artifact=artifact, loss=loss, serving=serving))
        decisions[stage.name] = decide(
            policy, observations, current=artifact, budget=budget
        )
        decisions[stage.name]["timing_scope"] = report["timing_scope"]
        decisions[stage.name]["production_engine_speed"] = False
    return dict(
        schema="draftfit-joint-control-v1",
        history=state,
        decisions=decisions,
        stop=any(d["action"] != "continue" for d in decisions.values()),
        branch_execution="explicit_new_phase_required",
    )
