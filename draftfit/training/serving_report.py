"""Validate external production measurements into ``joint_policy`` reports.

``normalize_serving_report(report, *, artifacts, bindings, domains, ...)`` is a
pure, fail-closed adapter. It neither launches a server nor reads checkpoints.
The caller must hash the actual saved checkpoint/export, inspect the running
engine, and supply those independent expectations; copying them from an
untrusted report defeats validation. Hashes establish consistency, not report
authenticity or proof that CUDA executed. All failures raise ``ValueError``.

External command JSON contract (``SCHEMA``), required fields::

    {"schema": SCHEMA, "split": "validation", "status": "complete",
     "production_engine_speed": true, "speed_decisions_allowed": false,
     "timing_scope": "end_to_end_generation",
     "acceptance_scope": "committed_draft_suffix_excludes_target_anchor",
     "bindings": B, "comparison_fingerprint": joint_policy.fingerprint(B),
     "baseline": {"name": "target-only", "status": "complete",
                  "comparison_fingerprint": ..., "runs": [R, ...]},
     "candidates": [{"name": "dspark", "status": "complete",
                     "artifact": A, "comparison_fingerprint": ...,
                     "runs": [R, ...]}, ...],
     "existing_draft": null}

``artifacts`` maps the exact expected candidate names to ``A``. ``A`` is the
existing joint_policy artifact (artifact_sha256/family/preset/step plus
training/objective/capture_fingerprint), extended with checkpoint_sha256 and
export_sha256. The checkpoint may be a family checkpoint distinct from the joint
artifact. Export SHA is the caller's deterministic manifest digest covering all
loaded files, including config and weights; a weights-only digest is insufficient.
An optional existing_draft has the same candidate structure; its independent
artifact must also be supplied as the ``existing_draft`` function argument.

``B`` has these required fields (additional JSON fields are fingerprinted)::

    {"target": {"model": ..., "revision": ..., "weights_sha256": ...,
                "tokenizer_sha256": ...},
     "runtime": {"engine": "sglang", "version": ..., "build_sha256": ...,
                 "device": ..., "dtype": ...},
     "validation_data_sha256": ...,
     "config": {"temperature": 0, "max_new_tokens": 8, "trials": 1,
                "concurrency": 1, "warmup_requests": 1, "cache_policy": ...},
     "prompts": [{"row_index": 0, "domain": "chat", "prompt_sha256": ...}, ...]}

Runtime engine is sglang, vllm or tensorrt-llm; record hardware, dependency
versions, TP, cache/graph options and serving settings in B as applicable.
Prompt SHA covers the exact rendered input (UTF-8 text or the caller's canonical
token serialization), after template application. Hash expectations come from
the fixed held-out dataset. Each prompt must occur in every trial in every arm.

``R`` reuses joint_generation's row_index/trial/token_ids/generated_tokens/
seconds/accepted_tokens/proposed_tokens, adding domain and prompt_sha256.
Tokens are nonnegative integer IDs; output must be nonempty, within the token
limit and exactly equal to target-only for the same (trial, row_index). Baseline
acceptance counts are zero; draft counts are mandatory, accepted <= proposed,
and proposed totals must be positive per domain (as required by joint_policy).
Seconds are measured client wall time for full generation including prefill,
excluding server startup and warmup. Rates and parity flags cannot replace raw
counts, IDs or timings. No reference timing report is upgraded to production.

Result ``candidates`` contains named, checkpoint/export-bound ``serving`` dicts
ready for ``joint_policy.decide``; ``existing_draft`` is separate and never
eligible as a trained candidate. Aggregate validation uses joint_policy._serving
itself. The returned comparison_fingerprint must be used in the policy config.
Set speed_decisions_allowed=false for smoke evidence: production execution alone
does not establish benchmark quality. Defaults require 32 matched requests.
Only the caller can authorize a speed decision after its benchmark gates pass,
by passing ``allow_speed_decisions=True``. Both caller and report must opt in.
"""

from __future__ import annotations

import math
from copy import deepcopy

from draftfit.training import joint_policy

SCHEMA = "draftfit-production-serving-v1"
TIMING_SCOPE = "end_to_end_generation"
ACCEPTANCE_SCOPE = "committed_draft_suffix_excludes_target_anchor"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing/empty {label}")
    return value


def _sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid {label}")
    return value


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("seconds must be finite and positive")
    return value


def _production(value):
    # Explicit contradictory provenance must not be erased by normalization.
    if value.get("production_engine_speed", True) is not True:
        raise ValueError("reference timings are not production measurements")
    if "reference" in str(value.get("preset", "")).lower():
        raise ValueError("reference preset is not production serving")
    if value.get("timing_scope", TIMING_SCOPE) != TIMING_SCOPE:
        raise ValueError("unsupported production timing_scope")
    if value.get("acceptance_scope", ACCEPTANCE_SCOPE) != ACCEPTANCE_SCOPE:
        raise ValueError("unsupported acceptance_scope")
    if value.get("status", "complete") != "complete":
        raise ValueError("incomplete production measurement")
    if (
        value.get("tokenparity", True) is not True
        or value.get("exact_token_parity", True) is not True
    ):
        raise ValueError("reported token parity failure")


def _artifact(value):
    joint_policy.observation_key(value)
    _sha(value["checkpoint_sha256"], "checkpoint_sha256")
    _sha(value["export_sha256"], "export_sha256")
    return value


def _binding(bindings, domains):
    if (
        not isinstance(domains, list)
        or not domains
        or any(not isinstance(d, str) or not d.strip() for d in domains)
        or len(set(domains)) != len(domains)
    ):
        raise ValueError("declare nonempty unique domains")
    target, runtime, config = (bindings[k] for k in ("target", "runtime", "config"))
    for name in ("model", "revision"):
        _text(target[name], f"target.{name}")
    for name in ("weights_sha256", "tokenizer_sha256"):
        _sha(target[name], f"target.{name}")
    if runtime["engine"] not in ("sglang", "vllm", "tensorrt-llm"):
        raise ValueError("unsupported production engine")
    for name in ("version", "device", "dtype"):
        _text(runtime[name], f"runtime.{name}")
    _sha(runtime["build_sha256"], "runtime.build_sha256")
    _production(runtime)
    if (
        "reference" in str(runtime.get("decode_state", "")).lower()
        or runtime.get("decode_state") == "native_full_reset_replay"
    ):
        raise ValueError("reference decode_state is not production serving")
    _sha(bindings["validation_data_sha256"], "validation_data_sha256")
    if type(config["temperature"]) not in (int, float) or config["temperature"] != 0:
        raise ValueError("exact parity requires greedy temperature=0")
    for name in ("max_new_tokens", "trials", "concurrency"):
        _integer(config[name], name, minimum=1)
    _integer(config["warmup_requests"], "warmup_requests")
    _text(config["cache_policy"], "cache_policy")
    prompts = bindings["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("empty prompts/domains")
    indexed = {}
    for prompt in prompts:
        index = _integer(prompt["row_index"], "row_index")
        _text(prompt["domain"], "domain")
        _sha(prompt["prompt_sha256"], "prompt_sha256")
        if index in indexed:
            raise ValueError("duplicate prompt row_index")
        indexed[index] = prompt
    if {p["domain"] for p in prompts} != set(domains):
        raise ValueError("missing/extra/empty domains in prompts")
    return indexed


def _runs(arm, *, comparison, prompts, config, baseline=None):
    _production(arm)
    if arm["status"] != "complete" or arm["comparison_fingerprint"] != comparison:
        raise ValueError("incomplete arm or comparison_fingerprint mismatch")
    rows = arm["runs"]
    expected_count = len(prompts) * config["trials"]
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("missing/extra requests")
    indexed = {}
    for row in rows:
        _production(row)
        trial = _integer(row["trial"], "trial")
        index = _integer(row["row_index"], "row_index")
        key = trial, index
        if trial >= config["trials"] or index not in prompts or key in indexed:
            raise ValueError("unmatched or duplicate request")
        prompt = prompts[index]
        if (
            row["domain"] != prompt["domain"]
            or row["prompt_sha256"] != prompt["prompt_sha256"]
        ):
            raise ValueError("request domain/prompt hash mismatch")
        ids = row["token_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(t) is not int or t < 0 for t in ids)
        ):
            raise ValueError("missing/invalid output token IDs")
        count = _integer(row["generated_tokens"], "generated_tokens", minimum=1)
        if len(ids) != count or count > config["max_new_tokens"]:
            raise ValueError("output token count/limit mismatch")
        _seconds(row["seconds"])
        accepted = _integer(row["accepted_tokens"], "accepted_tokens")
        proposed = _integer(row["proposed_tokens"], "proposed_tokens")
        if accepted > proposed or accepted > count:
            raise ValueError("invalid acceptance counts")
        if baseline is None:
            if accepted or proposed:
                raise ValueError("target-only baseline has draft acceptance counts")
        elif ids != baseline[key]["token_ids"]:
            raise ValueError("exact token parity failure")
        indexed[key] = row
    return indexed


def normalize_serving_report(
    report,
    *,
    artifacts,
    bindings,
    domains,
    existing_draft=None,
    min_requests=32,
    min_domain_requests=1,
    allow_speed_decisions=False,
):
    """Validate a JSON mapping; return detached policy reports or raise ValueError.

    ``artifacts`` and ``bindings`` are independently verified caller expectations,
    never defaults extracted from ``report``. Candidate sets must match exactly.
    ``existing_draft`` is an artifact mapping, not a candidate name or a flag.
    No filesystem, network, subprocess, GPU or global state is touched.
    """
    try:
        return _normalize(
            report,
            artifacts=artifacts,
            bindings=bindings,
            domains=domains,
            existing_draft=existing_draft,
            min_requests=min_requests,
            min_domain_requests=min_domain_requests,
            allow_speed_decisions=allow_speed_decisions,
        )
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed production serving report: {exc}") from exc


def _normalize(
    report,
    *,
    artifacts,
    bindings,
    domains,
    existing_draft,
    min_requests,
    min_domain_requests,
    allow_speed_decisions,
):
    _integer(min_requests, "min_requests", minimum=1)
    _integer(min_domain_requests, "min_domain_requests", minimum=1)
    prompts = _binding(bindings, domains)
    comparison = joint_policy.fingerprint(bindings)
    if (
        report["schema"] != SCHEMA
        or report["split"] != "validation"
        or report["status"] != "complete"
        or report["production_engine_speed"] is not True
        or report["timing_scope"] != TIMING_SCOPE
        or report["acceptance_scope"] != ACCEPTANCE_SCOPE
    ):
        raise ValueError("unsupported production serving report contract")
    _production(report)
    if type(report["speed_decisions_allowed"]) is not bool:
        raise ValueError("speed_decisions_allowed must be boolean")
    if type(allow_speed_decisions) is not bool:
        raise ValueError("allow_speed_decisions must be boolean")
    speed_allowed = allow_speed_decisions and report["speed_decisions_allowed"]
    if (
        joint_policy.fingerprint(report["bindings"]) != comparison
        or report["comparison_fingerprint"] != comparison
    ):
        raise ValueError("target/runtime/data/settings comparison mismatch")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("empty expected candidates")
    for name, artifact in artifacts.items():
        _text(name, "candidate name")
        _artifact(artifact)
    candidates = report["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(artifacts):
        raise ValueError("missing/extra candidates")
    names = [_text(c["name"], "candidate name") for c in candidates]
    if len(set(names)) != len(names) or set(names) != set(artifacts):
        raise ValueError("missing/duplicate/substituted candidates")
    base = report["baseline"]
    if base["name"] != "target-only":
        raise ValueError("target-only baseline required")
    baseline = _runs(
        base, comparison=comparison, prompts=prompts, config=bindings["config"]
    )
    source_sha = joint_policy.fingerprint(report)
    policy_config = dict(
        domains=domains,
        comparison_fingerprint=comparison,
        min_requests=min_requests,
        min_domain_requests=min_domain_requests,
    )

    def candidate(arm, expected):
        _artifact(arm["artifact"])
        if joint_policy.fingerprint(arm["artifact"]) != joint_policy.fingerprint(
            expected
        ):
            raise ValueError("artifact/checkpoint/export binding mismatch")
        rows = _runs(
            arm,
            comparison=comparison,
            prompts=prompts,
            config=bindings["config"],
            baseline=baseline,
        )
        groups = []
        for domain in sorted(domains):
            keys = sorted(k for k, r in rows.items() if r["domain"] == domain)
            group, paired = [rows[k] for k in keys], [baseline[k] for k in keys]
            groups.append(
                dict(
                    name=domain,
                    requests=len(keys),
                    matched_requests=len(keys),
                    exact_token_parity=True,
                    output_tokens=sum(r["generated_tokens"] for r in group),
                    baseline_output_tokens=sum(r["generated_tokens"] for r in paired),
                    seconds=sum(r["seconds"] for r in group),
                    baseline_seconds=sum(r["seconds"] for r in paired),
                    accepted_tokens=sum(r["accepted_tokens"] for r in group),
                    proposed_tokens=sum(r["proposed_tokens"] for r in group),
                )
            )
        normalized = dict(
            expected,
            name=arm["name"],
            comparison_fingerprint=comparison,
            split="validation",
            exact_token_parity=True,
            domains=groups,
            production_engine_speed=True,
            timing_scope=TIMING_SCOPE,
            acceptance_scope=ACCEPTANCE_SCOPE,
            speed_decisions_allowed=speed_allowed,
            source_report_sha256=source_sha,
        )
        joint_policy._serving(normalized, expected, policy_config)
        return normalized

    normalized = [
        candidate(c, artifacts[c["name"]])
        for c in sorted(candidates, key=lambda c: c["name"])
    ]
    existing = report.get("existing_draft")
    if (existing is None) != (existing_draft is None):
        raise ValueError("existing_draft expectation/report mismatch")
    if existing is not None:
        name = _text(existing["name"], "existing_draft name")
        if name in artifacts or name == "target-only":
            raise ValueError("existing_draft name collides with another arm")
        existing = candidate(existing, _artifact(existing_draft))
    return deepcopy(
        dict(
            schema=SCHEMA,
            comparison_fingerprint=comparison,
            production_engine_speed=True,
            speed_decisions_allowed=speed_allowed,
            timing_scope=TIMING_SCOPE,
            source_report_sha256=source_sha,
            baseline=base,
            candidates=normalized,
            existing_draft=existing,
        )
    )


__all__ = ["SCHEMA", "TIMING_SCOPE", "ACCEPTANCE_SCOPE", "normalize_serving_report"]
