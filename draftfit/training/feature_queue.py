"""A byte-bounded CPU feature queue with explicit consumer leases.

Queued and leased tensors both count against the limit. A completed joint
backward/update releases the lease; checkpoint cursors make it recoverable.
The queue does not own token journals or delete source data.
"""

from collections import deque
from dataclasses import dataclass


def tensor_bytes(value):
    """Count unique underlying storages, including views, without importing CUDA."""
    seen = set()

    def count(item):
        if hasattr(item, "untyped_storage"):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(count(x) for x in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(x) for x in item)
        return 0

    return count(value)


@dataclass
class FeatureLease:
    sequence: int
    value: object
    size_bytes: int


class BoundedFeatureQueue:
    """Single-process queue; producer and consumer may be GPU time-sliced."""

    def __init__(self, max_bytes, max_items):
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or type(max_items) is not int
            or max_items < 1
        ):
            raise ValueError("feature queue limits must be positive integers")
        self.max_bytes, self.max_items = max_bytes, max_items
        self._queued, self._leased = deque(), {}
        self._next = 0
        self.resident_bytes = self.peak_bytes = 0
        self.produced = self.consumed = self.cancelled = self.backpressure = 0

    @property
    def resident_items(self):
        return len(self._queued) + len(self._leased)

    @property
    def ready(self):
        return len(self._queued)

    def can_fit(self, size_bytes):
        return (
            self.resident_items < self.max_items
            and self.resident_bytes + size_bytes <= self.max_bytes
        )

    def publish(self, value, size_bytes=None):
        size = (
            tensor_bytes(value.tensors if hasattr(value, "tensors") else value)
            if size_bytes is None
            else size_bytes
        )
        if type(size) is not int or size < 0 or size > self.max_bytes:
            raise ValueError("feature batch exceeds queue byte limit")
        if not self.can_fit(size):
            self.backpressure += 1
            raise BufferError("feature queue is full; producer must yield to consumers")
        lease = FeatureLease(self._next, value, size)
        self._next += 1
        self._queued.append(lease)
        self.resident_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        self.produced += 1
        return lease.sequence

    def acquire(self):
        if not self._queued:
            raise LookupError("feature queue is empty")
        lease = self._queued.popleft()
        self._leased[lease.sequence] = lease
        return lease

    def release(self, lease, *, consumed):
        if self._leased.get(lease.sequence) is not lease:
            raise ValueError("feature lease is unknown or already released")
        del self._leased[lease.sequence]
        self.resident_bytes -= lease.size_bytes
        self.consumed += int(consumed)
        self.cancelled += int(not consumed)
        lease.value = None

    def clear(self):
        self.cancelled += self.resident_items
        for lease in self._queued:
            lease.value = None
        for lease in self._leased.values():
            lease.value = None
        self._queued.clear()
        self._leased.clear()
        self.resident_bytes = 0

    def cancel_ready(self):
        """Discard producer lookahead, retaining consumer leases across epochs."""
        while self._queued:
            lease = self._queued.popleft()
            self.resident_bytes -= lease.size_bytes
            self.cancelled += 1
            lease.value = None

    def metrics(self):
        return {
            key: getattr(self, key)
            for key in (
                "resident_bytes",
                "peak_bytes",
                "resident_items",
                "produced",
                "consumed",
                "cancelled",
                "backpressure",
            )
        }
