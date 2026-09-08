"""Small, weight-free manifest for offline and online target capture.

The manifest is the hand-off between a capture producer and the trainer.  It
records the resolved algorithm layout plus model/tokenizer provenance; tensor
payloads stay in the feature store or capture files.  Keeping this module
stdlib-only makes the dry-run and contract checks usable on a CPU host.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "specforge_capture_v1"


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def build_capture_manifest(
    *,
    strategy: str,
    capture_method: str,
    capture_layers: Sequence[int],
    feature_names: Sequence[str],
    target_model: str,
    target_revision: str,
    target_hidden_size: int | None = None,
    target_vocab_size: int | None = None,
    draft_vocab_size: int | None = None,
    target_repr: str | None = None,
    target_feature: str | None = None,
    tokenizer_version: str | None = None,
    chat_template: str | None = None,
    max_length: int | None = None,
    is_preformatted: bool | None = None,
    draft_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable capture contract."""

    strategy = _non_empty(strategy, "strategy")
    capture_method = _non_empty(capture_method, "capture_method")
    target_model = _non_empty(target_model, "target_model")
    target_revision = _non_empty(target_revision, "target_revision")
    layers = tuple(capture_layers)
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("capture_layers must contain non-negative integers")
    if len(set(layers)) != len(layers):
        raise ValueError("capture_layers must be unique")
    names = tuple(_non_empty(name, "feature name") for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    for name, value in (
        ("target_hidden_size", target_hidden_size),
        ("target_vocab_size", target_vocab_size),
        ("draft_vocab_size", draft_vocab_size),
        ("max_length", max_length),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    if tokenizer_version is not None:
        tokenizer_version = _non_empty(tokenizer_version, "tokenizer_version")
    contract: dict[str, Any] = {
        "strategy": strategy,
        "capture_method": capture_method,
        "capture_layers": list(layers),
        "feature_names": list(names),
        "target_repr": target_repr,
        "target_feature": target_feature,
        "target_hidden_size": target_hidden_size,
        "target_vocab_size": target_vocab_size,
        "draft_vocab_size": draft_vocab_size,
        "max_length": max_length,
        "is_preformatted": is_preformatted,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target": {"model": target_model, "revision": target_revision},
        "tokenizer": {
            "version": tokenizer_version or target_model,
            "chat_template": chat_template,
        },
        "draft": {
            "config": dict(draft_config or {}),
            "config_hash": _hash(draft_config or {}),
        },
        "contract": contract,
    }
    manifest["contract_hash"] = _hash(contract)
    manifest["manifest_hash"] = _hash(manifest)
    return manifest


def validate_capture_manifest(
    manifest: Mapping[str, Any],
    *,
    strategy: str | None = None,
    capture_method: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and return a manifest's immutable contract fields.

    This is intentionally strict when a sidecar exists.  Older feature
    directories without a sidecar remain supported by the legacy reader.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("capture manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported capture manifest schema: {manifest.get('schema_version')!r}"
        )
    contract = manifest.get("contract")
    target = manifest.get("target")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(contract, Mapping) or not isinstance(target, Mapping) or not isinstance(tokenizer, Mapping):
        raise ValueError("capture manifest requires target, tokenizer, and contract objects")
    if manifest.get("contract_hash") != _hash(contract):
        raise ValueError("capture manifest contract_hash mismatch")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if manifest.get("manifest_hash") != _hash(unsigned):
        raise ValueError("capture manifest manifest_hash mismatch")
    if strategy is not None and contract.get("strategy") != strategy:
        raise ValueError(
            f"capture manifest strategy {contract.get('strategy')!r} != {strategy!r}"
        )
    if capture_method is not None and contract.get("capture_method") != capture_method:
        raise ValueError(
            f"capture manifest method {contract.get('capture_method')!r} != {capture_method!r}"
        )
    if feature_names is not None:
        expected = sorted(str(name) for name in feature_names)
        actual = sorted(str(name) for name in contract.get("feature_names", ()))
        if actual != expected:
            raise ValueError(f"capture manifest features {actual!r} != {expected!r}")
    return dict(manifest)


def load_capture_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return validate_capture_manifest(json.load(stream))


def write_capture_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    validated = validate_capture_manifest(manifest)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


__all__ = [
    "SCHEMA_VERSION",
    "build_capture_manifest",
    "load_capture_manifest",
    "validate_capture_manifest",
    "write_capture_manifest",
]
