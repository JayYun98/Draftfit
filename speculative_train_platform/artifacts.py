"""Small, deterministic run-artifact manifests.

The manifest records provenance and validation references without hashing large
model/checkpoint trees.  JSON descriptors are hashed; large files are pinned by
size and mtime so a release command never reads multi-GB weights by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .inference.parity import manifest_hash

_MAX_HASH_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_entry(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"artifact source does not exist: {path}")
    stat = resolved.stat()
    entry: dict[str, Any] = {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "bytes": int(stat.st_size) if resolved.is_file() else None,
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if resolved.is_file() and stat.st_size <= _MAX_HASH_BYTES:
        entry["sha256"] = _sha256(resolved)
    elif resolved.is_file():
        entry["hash_skipped"] = "file_larger_than_8MiB"
    return entry


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def build_artifact_manifest(
    *,
    run_id: str,
    target_model: str,
    target_revision: str,
    tokenizer_version: str | None = None,
    backend_revision: str | None = None,
    sources: Mapping[str, str] | None = None,
    validation_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable provenance manifest for one run."""
    for name, value in (
        ("run_id", run_id),
        ("target_model", target_model),
        ("target_revision", target_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    metadata = {
        "run_id": run_id,
        "target_model": target_model,
        "target_revision": target_revision,
    }
    if tokenizer_version is not None:
        metadata["tokenizer_version"] = tokenizer_version
    if backend_revision is not None:
        metadata["backend_revision"] = backend_revision
    manifest: dict[str, Any] = {
        "schema_version": "dspark_artifact_v1",
        "metadata": metadata,
        "sources": {
            str(name): _source_entry(path)
            for name, path in sorted(dict(sources or {}).items())
        },
        "validation": [],
    }
    for path in validation_paths:
        entry = _source_entry(path)
        if entry["kind"] != "file":
            raise ValueError(f"validation artifact must be a file: {path}")
        record: dict[str, Any] = {"path": entry["path"]}
        if "sha256" in entry:
            record["sha256"] = entry["sha256"]
        try:
            payload = _load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping) and "passed" in payload:
            record["passed"] = bool(payload["passed"])
        record["manifest_hash"] = (
            manifest_hash(payload) if payload is not None else None
        )
        manifest["validation"].append(record)
    manifest["artifact_hash"] = manifest_hash(manifest)
    return manifest


def write_artifact_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


__all__ = ["build_artifact_manifest", "write_artifact_manifest"]
