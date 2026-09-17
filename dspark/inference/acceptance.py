"""Dependency-free acceptance-summary validation for serving gates."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _number(value: Any, field: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field} must be a number")
        return None
    value = float(value)
    if not math.isfinite(value):
        errors.append(f"{field} must be finite")
        return None
    return value


def _threshold(value: Any, field: str, errors: list[str]) -> float | None:
    if value is None:
        return None
    parsed = _number(value, field, errors)
    if parsed is not None and parsed < 0:
        errors.append(f"{field} must be non-negative")
        return None
    return parsed


def validate_acceptance_summary(
    summary: Mapping[str, Any],
    *,
    min_train_acc_len: float | None = None,
    min_holdout_acc_len: float | None = None,
    min_holdout_pos2: float | None = None,
    expected_train_requests: int | None = None,
    expected_holdout_requests: int | None = None,
) -> dict[str, Any]:
    """Validate an aggregated serving-gate summary.

    Thresholds are optional so the command can validate an existing report's
    recorded status, or re-evaluate the same artifact for a different model.
    ``true_position_acceptance[1]`` is the second proposed-token position,
    matching the historical pre-10K gate.
    """
    errors: list[str] = []
    if not isinstance(summary, Mapping):
        return {"passed": False, "errors": ["summary must be a JSON object"]}

    train_min = _threshold(min_train_acc_len, "min_train_acc_len", errors)
    holdout_min = _threshold(min_holdout_acc_len, "min_holdout_acc_len", errors)
    pos2_min = _threshold(min_holdout_pos2, "min_holdout_pos2", errors)

    rows: dict[str, Any] = {}
    for split in ("train", "holdout"):
        row = summary.get(split)
        if not isinstance(row, Mapping):
            errors.append(f"{split} must be a JSON object")
            continue
        rows[split] = dict(row)
        for field in ("requests", "weighted_acc_len", "true_position_acceptance"):
            if field not in row:
                errors.append(f"{split}.{field} is required")
        if "requests" in row:
            requests = row["requests"]
            if (
                isinstance(requests, bool)
                or not isinstance(requests, int)
                or requests < 0
            ):
                errors.append(f"{split}.requests must be a non-negative integer")
        if "weighted_acc_len" in row:
            _number(row["weighted_acc_len"], f"{split}.weighted_acc_len", errors)
        positions = row.get("true_position_acceptance")
        if positions is not None:
            if not isinstance(positions, list):
                errors.append(f"{split}.true_position_acceptance must be a list")
            else:
                for index, value in enumerate(positions):
                    _number(value, f"{split}.true_position_acceptance[{index}]", errors)

    checks: list[dict[str, Any]] = []

    def check(name: str, value: Any, minimum: float, label: str) -> None:
        actual = _number(value, label, errors)
        if actual is not None:
            checks.append(
                {
                    "name": name,
                    "value": actual,
                    "minimum": minimum,
                    "passed": actual >= minimum,
                }
            )
            if actual < minimum:
                errors.append(f"{label} {actual:.4f} < {minimum}")

    if train_min is not None:
        check(
            "train_acc_len",
            rows.get("train", {}).get("weighted_acc_len"),
            train_min,
            "train acc_len",
        )
    if holdout_min is not None:
        check(
            "holdout_acc_len",
            rows.get("holdout", {}).get("weighted_acc_len"),
            holdout_min,
            "holdout acc_len",
        )
    if pos2_min is not None:
        positions = rows.get("holdout", {}).get("true_position_acceptance", [])
        value = (
            positions[1] if isinstance(positions, list) and len(positions) > 1 else None
        )
        check("holdout_pos2", value, pos2_min, "holdout pos2")

    for split, expected in (
        ("train", expected_train_requests),
        ("holdout", expected_holdout_requests),
    ):
        if expected is None:
            continue
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            errors.append(f"expected_{split}_requests must be a non-negative integer")
            continue
        actual = rows.get(split, {}).get("requests")
        if actual != expected:
            errors.append(f"expected {expected} {split} requests, found {actual}")
        checks.append(
            {
                "name": f"{split}_requests",
                "value": actual,
                "expected": expected,
                "passed": actual == expected,
            }
        )

    for split in ("train", "holdout"):
        invalid = rows.get(split, {}).get("invalid_requests", [])
        if invalid:
            errors.append(f"{split} contains invalid requests")

    thresholds_given = any(
        value is not None
        for value in (
            min_train_acc_len,
            min_holdout_acc_len,
            min_holdout_pos2,
            expected_train_requests,
            expected_holdout_requests,
        )
    )
    source_passed = summary.get("passed")
    if not thresholds_given and source_passed is False:
        errors.append("summary reports passed=false")

    return {
        "passed": not errors,
        "source_passed": source_passed,
        "checks": checks,
        "train": rows.get("train"),
        "holdout": rows.get("holdout"),
        "errors": errors,
    }


__all__ = ["validate_acceptance_summary"]
