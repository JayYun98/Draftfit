"""Engine-independent target state snapshots for exact replay gates.

The state adapter does not know how SGLang represents KDA/MLA/conv caches.  A
backend supplies a small capture/restore callback when its cache is not already
made of portable mappings and tensors.  The snapshot itself is immutable and
contains the consumed token prefix, model revision, state kind, and a digest;
restoring a snapshot with the wrong target or token prefix fails closed.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

from .parity import manifest_hash

_SCHEMA_VERSION = "target_state_v1"
CaptureState = Callable[[Any], Any]
RestoreState = Callable[[Any, Any], Any]


def _token_ids(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("token_ids must be a sequence of non-negative integers")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError("token_ids must be a sequence of non-negative integers") from exc
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in items):
        raise TypeError("token_ids must contain only integers")
    result = tuple(int(item) for item in items)
    if any(item < 0 for item in result):
        raise ValueError("token_ids must be non-negative")
    return result


def _tensor_bytes(value: Any) -> tuple[str, tuple[int, ...], bytes] | None:
    """Read tensor/array bytes without importing torch or numpy."""
    if not any(hasattr(value, attr) for attr in ("detach", "tobytes", "numpy")):
        return None
    candidate = value
    try:
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        contiguous = getattr(candidate, "contiguous", None)
        if callable(contiguous):
            candidate = contiguous()
        numpy = getattr(candidate, "numpy", None)
        if callable(numpy):
            candidate = numpy()
        tobytes = getattr(candidate, "tobytes", None)
        if not callable(tobytes):
            return None
        raw = tobytes()
        shape = tuple(int(dim) for dim in getattr(candidate, "shape", ()))
        dtype = str(getattr(candidate, "dtype", "unknown"))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"state tensor cannot be serialized: {exc}") from exc
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("state tensor tobytes() must return bytes")
    if any(dim < 0 for dim in shape):
        raise ValueError("state tensor shape cannot contain negative dimensions")
    return dtype, shape, bytes(raw)


def freeze_state(value: Any) -> Any:
    """Convert supported cache values to immutable JSON-compatible data."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("state mapping keys must be strings")
            result[key] = freeze_state(item)
        return result
    if isinstance(value, (list, tuple)):
        return [freeze_state(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": bytes(value).hex()}
    tensor = _tensor_bytes(value)
    if tensor is not None:
        dtype, shape, raw = tensor
        return {
            "__tensor__": True,
            "dtype": dtype,
            "shape": list(shape),
            "data": raw.hex(),
        }
    raise TypeError(
        "unsupported target state value; provide capture_state/restore_state "
        f"callbacks for {type(value).__name__}"
    )


@dataclass(frozen=True)
class TargetStateSnapshot:
    """Portable immutable state plus the exact token prefix it represents."""

    target_model_version: str
    state_kind: str
    sequence_position: int
    token_ids: tuple[int, ...]
    state: Any
    metadata: Mapping[str, Any]
    state_hash: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported state snapshot schema {self.schema_version!r}")
        if (
            not isinstance(self.target_model_version, str)
            or not self.target_model_version.strip()
            or self.target_model_version == "unknown"
        ):
            raise ValueError("target_model_version must be a pinned revision")
        if not isinstance(self.state_kind, str) or not self.state_kind:
            raise ValueError("state_kind must be non-empty")
        if (
            isinstance(self.sequence_position, bool)
            or not isinstance(self.sequence_position, Integral)
            or self.sequence_position < 0
        ):
            raise ValueError("sequence_position must be non-negative")
        ids = _token_ids(self.token_ids)
        if self.sequence_position != len(ids):
            raise ValueError("sequence_position must equal len(token_ids)")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("snapshot metadata must be a mapping")
        frozen_state = freeze_state(self.state)
        frozen_metadata = freeze_state(self.metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise TypeError("snapshot metadata must be a mapping")
        expected = manifest_hash({"state": frozen_state})
        if self.state_hash != expected:
            raise ValueError("state_hash does not match snapshot state")
        object.__setattr__(self, "state", frozen_state)
        object.__setattr__(self, "metadata", frozen_metadata)
        object.__setattr__(self, "token_ids", ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_model_version": self.target_model_version,
            "state_kind": self.state_kind,
            "sequence_position": self.sequence_position,
            "token_ids": list(self.token_ids),
            "state": copy.deepcopy(self.state),
            "metadata": copy.deepcopy(dict(self.metadata)),
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetStateSnapshot":
        if not isinstance(value, Mapping):
            raise TypeError("state snapshot must be an object")
        required = {
            "schema_version", "target_model_version", "state_kind",
            "sequence_position", "token_ids", "state", "metadata", "state_hash",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"state snapshot missing fields: {missing}")
        frozen = freeze_state(value["state"])
        metadata = freeze_state(value["metadata"])
        if not isinstance(metadata, Mapping):
            raise TypeError("snapshot metadata must be a mapping")
        return cls(
            schema_version=value["schema_version"],
            target_model_version=value["target_model_version"],
            state_kind=value["state_kind"],
            sequence_position=value["sequence_position"],
            token_ids=_token_ids(value["token_ids"]),
            state=frozen,
            metadata=metadata,
            state_hash=value["state_hash"],
        )


class TargetStateAdapter:
    """Capture/restore boundary for stateful target engines."""

    def __init__(
        self,
        *,
        target_model_version: str,
        state_kind: str,
        capture_state: CaptureState | None = None,
        restore_state: RestoreState | None = None,
    ) -> None:
        if (
            not isinstance(target_model_version, str)
            or not target_model_version.strip()
            or target_model_version == "unknown"
        ):
            raise ValueError("target_model_version must be a pinned revision")
        if not isinstance(state_kind, str) or not state_kind:
            raise ValueError("state_kind must be non-empty")
        if (capture_state is None) != (restore_state is None):
            raise ValueError("capture_state and restore_state must be provided together")
        self.target_model_version = target_model_version
        self.state_kind = state_kind
        self._capture_state = capture_state
        self._restore_state = restore_state

    def capture(
        self,
        state: Any,
        token_ids: Sequence[int],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TargetStateSnapshot:
        ids = _token_ids(token_ids)
        if state is None:
            raise ValueError("target state is required; refusing an empty snapshot")
        captured = self._capture_state(state) if self._capture_state else state
        frozen = freeze_state(captured)
        frozen_metadata = freeze_state(dict(metadata or {}))
        if not isinstance(frozen_metadata, Mapping):  # pragma: no cover - defensive
            raise TypeError("snapshot metadata must be a mapping")
        return TargetStateSnapshot(
            target_model_version=self.target_model_version,
            state_kind=self.state_kind,
            sequence_position=len(ids),
            token_ids=ids,
            state=frozen,
            metadata=frozen_metadata,
            state_hash=manifest_hash({"state": frozen}),
        )

    def restore(
        self,
        snapshot: TargetStateSnapshot | Mapping[str, Any],
        *,
        token_ids: Sequence[int] | None = None,
        current_state: Any = None,
    ) -> Any:
        if not isinstance(snapshot, TargetStateSnapshot):
            snapshot = TargetStateSnapshot.from_dict(snapshot)
        if snapshot.target_model_version != self.target_model_version:
            raise ValueError(
                "target model revision mismatch: "
                f"snapshot={snapshot.target_model_version!r}, "
                f"adapter={self.target_model_version!r}"
            )
        if snapshot.state_kind != self.state_kind:
            raise ValueError(
                f"target state kind mismatch: snapshot={snapshot.state_kind!r}, "
                f"adapter={self.state_kind!r}"
            )
        if token_ids is None:
            raise ValueError("token_ids are required for an exact state restore")
        if _token_ids(token_ids) != snapshot.token_ids:
            raise ValueError("target token prefix mismatch; refusing state restore")
        frozen = copy.deepcopy(snapshot.state)
        if self._restore_state:
            return self._restore_state(frozen, current_state)
        if current_state is not None:
            raise ValueError("current_state requires a restore_state callback")
        return frozen


__all__ = ["TargetStateAdapter", "TargetStateSnapshot", "freeze_state"]
