"""One capture producer with bounded RAM credit and optimizer acknowledgements."""

from contextlib import contextmanager
from threading import Condition, Thread

from draftfit.training.feature_queue import BoundedFeatureQueue, tensor_bytes


class AsyncFeatureQueue:
    """Reserve credit before capture; retain it until the consumer commits."""

    def __init__(self, max_bytes, max_items):
        self._queue = BoundedFeatureQueue(max_bytes, max_items)
        self._condition = Condition()
        self._capturing = False
        self._thread = None
        self._stopped = True
        self._finished = False
        self._error = None
        self._paused = 0
        self._reserved = 0
        self._waiting = None
        self._peak = 0

    def start(self, items, capture, estimate_bytes):
        if self._thread is not None:
            raise RuntimeError("close the previous feature producer before restarting")
        self._stopped, self._finished, self._error = False, False, None
        self._thread = Thread(
            target=self._produce,
            args=(items, capture, estimate_bytes),
            name="draftfit-feature-producer",
        )
        self._thread.start()

    def _produce(self, items, capture, estimate_bytes):
        try:
            for item in items:
                expected = estimate_bytes(item)
                if (
                    type(expected) is not int
                    or not 0 < expected <= self._queue.max_bytes
                ):
                    raise ValueError("feature batch exceeds async queue byte budget")
                with self._condition:
                    self._waiting = expected
                    if not self._queue.can_fit(expected):
                        self._queue.backpressure += 1
                    while not self._stopped and (
                        self._paused or not self._queue.can_fit(expected)
                    ):
                        self._condition.notify_all()
                        self._condition.wait()
                    self._waiting = None
                    if self._stopped:
                        return
                    self._reserved = expected
                    self._peak = max(self._peak, self._queue.resident_bytes + expected)
                    self._capturing = True
                try:
                    value = capture(item)
                finally:
                    with self._condition:
                        self._capturing = False
                        self._condition.notify_all()
                size = tensor_bytes(
                    value.tensors if hasattr(value, "tensors") else value
                )
                with self._condition:
                    if size > expected:
                        raise ValueError(
                            "captured feature storage exceeds reserved queue bytes"
                        )
                    self._reserved = 0
                    if self._stopped:
                        return
                    self._queue.publish(value, size)
                    self._condition.notify_all()
                del value
        except BaseException as exc:
            with self._condition:
                self._error = exc
        finally:
            with self._condition:
                self._reserved = 0
                self._waiting = None
                self._finished = True
                self._condition.notify_all()

    def acquire(self):
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError(
                        "asynchronous feature capture failed"
                    ) from self._error
                if self._queue.ready:
                    return self._queue.acquire()
                if self._finished or self._stopped:
                    raise StopIteration
                if self._waiting is not None and not self._queue.can_fit(self._waiting):
                    raise ValueError(
                        "async queue cannot hold accumulation window; increase bytes or batches"
                    )
                self._condition.wait()

    def release(self, lease, *, consumed):
        with self._condition:
            self._queue.release(lease, consumed=consumed)
            self._condition.notify_all()

    @contextmanager
    def pause(self):
        """Wait for active capture before evaluation or target memory handoff."""
        with self._condition:
            self._paused += 1
        try:
            with self._condition:
                while self._capturing:
                    self._condition.wait()
            yield
        finally:
            with self._condition:
                self._paused -= 1
                self._condition.notify_all()

    def stop(self):
        """Join active capture before discarding reproducible, unconsumed features."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._condition:
            self._queue.cancel_ready()

    def clear(self):
        self.stop()
        with self._condition:
            self._queue.clear()
            self._error = None

    def metrics(self):
        with self._condition:
            metrics = self._queue.metrics()
            metrics["resident_bytes"] += self._reserved
            metrics["resident_items"] += int(bool(self._reserved))
            metrics["peak_bytes"] = max(metrics["peak_bytes"], self._peak)
            metrics["reserved_bytes"] = self._reserved
            metrics["producer_alive"] = (
                self._thread is not None and self._thread.is_alive()
            )
            return metrics
