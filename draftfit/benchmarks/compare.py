"""Compare compatible SGLang measurements; this is not a release certificate."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def _load_report(path):
    with path.open("rb") as handle:
        payload = handle.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("benchmark report exceeds 1 MiB")
    return json.loads(payload)


def _positive(value, name, integer=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        or (integer and type(value) is not int)
    ):
        raise ValueError(
            f"{name} must be a finite positive {'integer' if integer else 'number'}"
        )
    return value


def validate_runtime(runtime):
    """Validate operator-recorded identity, not attest the remote server."""
    required = {
        "target_revision",
        "tokenizer_revision",
        "backend_version",
        "hardware",
        "common_server_config_sha256",
        "draft_revision",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise ValueError(f"runtime must contain exactly {sorted(required)}")
    for key, value in runtime.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"runtime.{key} must be a nonempty string")
        if key.endswith("revision"):
            if key == "draft_revision" and value == "none":
                continue
            if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
                raise ValueError(
                    f"runtime.{key} requires a full commit or content hash"
                )
            if key == "tokenizer_revision" and len(value) != 40:
                raise ValueError(
                    "runtime.tokenizer_revision requires a full 40-hex commit"
                )
    if not re.fullmatch(r"[0-9a-f]{64}", runtime["common_server_config_sha256"]):
        raise ValueError("runtime.common_server_config_sha256 requires SHA256")
    return runtime


def _validate(report):
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    try:
        if report["backend"] != "sglang":
            raise ValueError("requires sglang reports with a workload hash")
        if not isinstance(report["dataset"], str) or not report["dataset"].strip():
            raise ValueError("missing dataset identity")
        for name in ("samples", "output_tokens"):
            _positive(report[name], name, integer=True)
        for name in ("latency_seconds", "throughput_tokens_per_second"):
            _positive(report[name], name)
        if not math.isclose(
            report["throughput_tokens_per_second"],
            report["output_tokens"] / report["latency_seconds"],
            rel_tol=1e-6,
        ):
            raise ValueError(
                "throughput disagrees with output_tokens / latency_seconds"
            )
        metadata = report["metadata"]
        source = metadata["source"]
        if source["kind"] not in ("messages_jsonl", "builtin") or not re.fullmatch(
            "[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError("requires a SHA256 workload identity")
        if source["kind"] == "builtin" and (
            source.get("name") != report["dataset"]
            or "rendered_prompts_sha256" not in metadata
        ):
            raise ValueError(
                "builtin reports require dataset and rendered prompt identities"
            )
        if source["kind"] == "messages_jsonl" and report["dataset"] != "messages-jsonl":
            raise ValueError("messages_jsonl source requires messages-jsonl dataset")
        _positive(source["sample_count"], "source.sample_count", integer=True)
        for key in ("model", "tokenizer"):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise ValueError(f"missing {key} identity")
        for key in ("concurrency", "warmup_prompts", "num_prompts_requested"):
            _positive(metadata[key], key, integer=True)
        if metadata["num_prompts_requested"] != report["samples"]:
            raise ValueError("completed samples disagree with requested prompts")
        if (
            type(metadata["seed"]) is not int
            or type(metadata["trust_remote_code"]) is not bool
        ):
            raise ValueError("invalid seed or trust_remote_code")
        if metadata["max_samples"] is not None:
            _positive(metadata["max_samples"], "max_samples", integer=True)
        _positive(metadata["timeout_seconds"], "timeout_seconds")
        generation = metadata["generation"]
        _positive(generation["max_new_tokens"], "max_new_tokens", integer=True)
        for key in ("temperature", "top_p"):
            value = generation[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid {key}")
        if generation["temperature"] < 0 or not 0 < generation["top_p"] <= 1:
            raise ValueError("invalid temperature or top_p")
        if type(generation["top_k"]) is not int or (
            generation["top_k"] != -1 and generation["top_k"] <= 0
        ):
            raise ValueError("invalid top_k")
        if type(generation["enable_thinking"]) is not bool:
            raise ValueError("invalid enable_thinking")
        for key in ("average_acceptance_length", "spec_verify_count"):
            if report.get(key) is not None:
                _positive(report[key], key, integer=key == "spec_verify_count")
        # Paths and endpoints may differ across machines; compare recorded settings.
        identity = {
            key: metadata[key]
            for key in (
                "model",
                "tokenizer",
                "seed",
                "concurrency",
                "warmup_prompts",
                "num_prompts_requested",
                "max_samples",
                "trust_remote_code",
                "timeout_seconds",
                "generation",
            )
        }
        identity["source"] = {
            key: source[key] for key in ("kind", "sha256", "sample_count")
        }
        identity["dataset"] = report["dataset"]
        if "rendered_prompts_sha256" in metadata:
            rendered = metadata["rendered_prompts_sha256"]
            if not isinstance(rendered, str) or not re.fullmatch(
                "[0-9a-f]{64}", rendered
            ):
                raise ValueError("invalid rendered prompt hash")
            identity["rendered_prompts_sha256"] = rendered
        if "runtime" in metadata:
            runtime = validate_runtime(metadata["runtime"])
            identity["runtime"] = {
                k: v for k, v in runtime.items() if k != "draft_revision"
            }
        return identity
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"missing or malformed benchmark evidence: {exc}") from exc


def compare(baseline, candidate):
    """Return median aggregate throughput ratio for one or more reports per side."""
    if not baseline or not candidate:
        raise ValueError("at least one report per side is required")
    identity = _validate(baseline[0])
    for report in [*baseline, *candidate]:
        if _validate(report) != identity:
            raise ValueError("incompatible workload or benchmark settings")
    runtime_recorded = "runtime" in identity
    rendered_recorded = "rendered_prompts_sha256" in identity
    if runtime_recorded:
        for side in (baseline, candidate):
            if len({r["metadata"]["runtime"]["draft_revision"] for r in side}) != 1:
                raise ValueError("draft revision must be consistent within each side")
    baseline_rate = statistics.median(
        r["throughput_tokens_per_second"] for r in baseline
    )
    candidate_rate = statistics.median(
        r["throughput_tokens_per_second"] for r in candidate
    )
    ratio = _positive(candidate_rate / baseline_rate, "throughput_ratio")
    return {
        "status": "throughput_comparison_only",
        "baseline_runs": len(baseline),
        "candidate_runs": len(candidate),
        "baseline_median_tokens_per_second": baseline_rate,
        "candidate_median_tokens_per_second": candidate_rate,
        "throughput_ratio": ratio,
        "runtime_identity": (
            "operator_recorded_match" if runtime_recorded else "not_recorded"
        ),
        "rendered_prompt_identity": (
            "recorded_match" if rendered_recorded else "not_recorded"
        ),
        "evidence_strength": (
            "operator_recorded" if runtime_recorded and rendered_recorded else "weak"
        ),
        "production_gate": "not_evaluated",
        "limitations": [
            "Report provenance and independent real runs are not verified.",
            (
                "Runtime identity is operator supplied, not independently verified."
                if runtime_recorded
                else "Weak evidence: hardware, server configuration and immutable model revisions are not recorded."
            ),
            *(
                []
                if rendered_recorded
                else [
                    "Weak evidence: rendered measured-prompt hashes are not recorded."
                ]
            ),
            "Output correctness, latency percentiles and statistical significance are not evaluated.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        paths = args.baseline + args.candidate
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("each measurement must use a distinct report path")
        reports = [_load_report(p) for p in paths]
        result = compare(reports[: len(args.baseline)], reports[len(args.baseline) :])
    except (OSError, ValueError) as exc:
        parser.exit(2, f"comparison rejected: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
