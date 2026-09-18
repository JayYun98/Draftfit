"""Dependency-free parity checks for offline/online capture and serving."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": bytes(value).hex()}
    return value


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_manifest(
    specs: Mapping[str, Any],
    *,
    schema_version: Any = None,
    target_model_version: Any = None,
    aux_layer_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Normalize FeatureSpec-like objects into a comparable JSON manifest."""
    if not isinstance(specs, Mapping):
        raise TypeError("specs must be a mapping of feature names to metadata")
    features = {}
    for name, spec in sorted(specs.items()):
        shape = getattr(spec, "shape", None)
        dtype = getattr(spec, "dtype", None)
        if shape is None and isinstance(spec, Mapping):
            shape, dtype = spec.get("shape"), spec.get("dtype")
        if shape is None:
            raise ValueError(f"feature {name!r} has no shape")
        if isinstance(shape, int):
            shape = (shape,)
        try:
            shape = [int(dim) for dim in shape]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature {name!r} shape must be a sequence of integers"
            ) from exc
        if any(dim < 0 for dim in shape):
            raise ValueError(
                f"feature {name!r} shape cannot contain negative dimensions"
            )
        if dtype is None:
            raise ValueError(f"feature {name!r} has no dtype")
        target_repr = getattr(spec, "target_repr", None)
        target_meta = getattr(spec, "target_meta", None)
        required = getattr(spec, "required", None)
        if isinstance(spec, Mapping):
            target_repr = spec.get("target_repr", target_repr)
            target_meta = spec.get("target_meta", target_meta)
            required = spec.get("required", required)
        feature = {"shape": shape, "dtype": str(dtype)}
        if target_repr is not None:
            feature["target_repr"] = str(target_repr)
        if target_meta is not None:
            feature["target_meta"] = _canonical(target_meta)
        if required is not None:
            feature["required"] = bool(required)
        features[str(name)] = feature
    return {
        "schema_version": schema_version,
        "target_model_version": target_model_version,
        "aux_layer_ids": [int(layer) for layer in aux_layer_ids],
        "features": features,
    }


def compare_feature_manifests(
    offline: Mapping[str, Any], online: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare feature names, shape, dtype, layer IDs and optional hashes."""
    if not isinstance(offline, Mapping) or not isinstance(online, Mapping):
        return {
            "passed": False,
            "differences": [
                {"field": "manifest", "reason": "manifests must be objects"}
            ],
            "offline_hash": manifest_hash({"value": _canonical(offline)}),
            "online_hash": manifest_hash({"value": _canonical(online)}),
        }
    differences: list[dict[str, Any]] = []
    # Provenance fields are optional for legacy manifests, but once present
    # they are part of the gate: a tensor from a different sample, target
    # revision, or capture manifest is never interchangeable.
    for field in (
        "schema_version",
        "target_model_version",
        "sample_id",
        "manifest_hash",
        "aux_layer_ids",
    ):
        if offline.get(field) != online.get(field):
            differences.append(
                {
                    "field": field,
                    "offline": offline.get(field),
                    "online": online.get(field),
                }
            )
    left = offline.get("features", {})
    right = online.get("features", {})
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        differences.append({"field": "features", "reason": "features must be objects"})
    else:
        for name in sorted(set(left) | set(right)):
            if name not in left or name not in right:
                differences.append(
                    {
                        "feature": name,
                        "reason": "missing",
                        "offline": name in left,
                        "online": name in right,
                    }
                )
                continue
            if not isinstance(left[name], Mapping) or not isinstance(
                right[name], Mapping
            ):
                differences.append(
                    {"feature": name, "reason": "feature metadata must be objects"}
                )
                continue
            for field in (
                "shape",
                "dtype",
                "sha256",
                "target_repr",
                "target_meta",
                "required",
            ):
                if field in left[name] or field in right[name]:
                    if left[name].get(field) != right[name].get(field):
                        differences.append(
                            {
                                "feature": name,
                                "field": field,
                                "offline": left[name].get(field),
                                "online": right[name].get(field),
                            }
                        )
    return {
        "passed": not differences,
        "differences": differences,
        "offline_hash": manifest_hash(offline),
        "online_hash": manifest_hash(online),
    }


def _strict_token_ids(value: Sequence[int]) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("token IDs must be a sequence, not text or bytes")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError("token IDs must be a sequence") from exc
    if any(
        isinstance(token, bool) or not isinstance(token, Integral) for token in items
    ):
        raise ValueError("token IDs must contain only integers")
    result = [int(token) for token in items]
    if any(token < 0 for token in result):
        raise ValueError("token IDs must be non-negative")
    return result


def compare_token_ids(expected: Sequence[int], actual: Sequence[int]) -> dict[str, Any]:
    """Compare token IDs without lossy coercion; malformed inputs fail closed."""
    try:
        expected = _strict_token_ids(expected)
        actual = _strict_token_ids(actual)
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "expected_length": None,
            "actual_length": None,
            "first_mismatch": None,
            "error": str(exc),
        }
    first_mismatch = next(
        (i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None
    )
    if first_mismatch is None and len(expected) != len(actual):
        first_mismatch = min(len(expected), len(actual))
    return {
        "passed": first_mismatch is None,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "first_mismatch": first_mismatch,
    }


def compare_state_snapshots(expected: Any, actual: Any) -> dict[str, Any]:
    """Compare snapshots and reject incomplete/tampered state artifacts."""
    expected = (
        expected.to_dict() if callable(getattr(expected, "to_dict", None)) else expected
    )
    actual = actual.to_dict() if callable(getattr(actual, "to_dict", None)) else actual
    errors: list[str] = []
    if expected is None or actual is None:
        errors.append("state snapshots are required")
    snapshot_fields = {
        "schema_version",
        "target_model_version",
        "state_kind",
        "sequence_position",
        "token_ids",
        "state",
        "metadata",
        "state_hash",
    }
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        for label, value in (("expected", expected), ("actual", actual)):
            if not isinstance(value, Mapping):
                errors.append(f"{label} snapshot must be an object")
                continue
            # A snapshot is distinguished by any state-contract field. Once a
            # caller supplies one, silently treating a partial object as a
            # generic state would make a replay gate pass incorrectly.
            snapshot_markers = snapshot_fields - {"state"}
            if snapshot_markers & set(value):
                missing = sorted(snapshot_fields - set(value))
                if missing:
                    errors.append(f"{label} snapshot missing fields: {missing}")
                if value.get("target_model_version") in (None, "", "unknown"):
                    errors.append(f"{label} snapshot has no pinned target revision")
                if "state" in value and "state_hash" in value:
                    try:
                        computed = manifest_hash({"state": value["state"]})
                        if computed != value["state_hash"]:
                            errors.append(f"{label} snapshot state_hash mismatch")
                    except (TypeError, ValueError):
                        errors.append(f"{label} snapshot state is not serializable")
    try:
        expected_hash = manifest_hash(expected)
        actual_hash = manifest_hash(actual)
        equal = _canonical(expected) == _canonical(actual)
    except (TypeError, ValueError) as exc:
        expected_hash = actual_hash = None
        equal = False
        errors.append(f"state is not serializable: {exc}")
    return {
        "passed": equal and not errors,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "errors": errors,
    }


def compare_state_replay(
    expected: Any,
    actual: Any,
    *,
    expected_token_ids: Sequence[int] | None = None,
    actual_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Atomically gate state equality and the token prefix used to reach it."""

    def _snapshot_value(value: Any) -> Any:
        return value.to_dict() if callable(getattr(value, "to_dict", None)) else value

    left = _snapshot_value(expected)
    right = _snapshot_value(actual)
    if expected_token_ids is None and isinstance(left, Mapping):
        expected_token_ids = left.get("token_ids")
    if actual_token_ids is None and isinstance(right, Mapping):
        actual_token_ids = right.get("token_ids")
    errors: list[str] = []
    if expected_token_ids is None or actual_token_ids is None:
        errors.append("both snapshots must include token_ids")
        token_result = {
            "passed": False,
            "expected_length": None,
            "actual_length": None,
            "first_mismatch": None,
            "error": "token_ids are required for exact replay",
        }
    else:
        token_result = compare_token_ids(expected_token_ids, actual_token_ids)
    state_result = compare_state_snapshots(left, right)
    errors.extend(state_result.get("errors", []))
    if token_result.get("error"):
        errors.append(str(token_result["error"]))
    return {
        "passed": bool(
            state_result["passed"] and token_result["passed"] and not errors
        ),
        "state": state_result,
        "tokens": token_result,
        "errors": errors,
    }


__all__ = [
    "compare_feature_manifests",
    "compare_state_replay",
    "compare_state_snapshots",
    "compare_token_ids",
    "feature_manifest",
    "manifest_hash",
]
