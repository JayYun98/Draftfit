"""Real producer/consumer overlap, bounded credit, failures and shutdown."""

import threading
import unittest

import torch

from draftfit.training.async_feature_queue import AsyncFeatureQueue


class AsyncFeatureQueueTests(unittest.TestCase):
    def queue(self, size=24, items=3):
        queue = AsyncFeatureQueue(size, items)
        self.addCleanup(queue.clear)
        return queue

    def test_produces_while_consumer_holds_a_training_lease(self):
        second = threading.Event()
        queue = self.queue()

        def capture(index):
            if index == 1:
                second.set()
            return torch.tensor([index], dtype=torch.int64)

        queue.start(range(4), capture, lambda _: 8)
        first = queue.acquire()
        self.assertTrue(
            second.wait(2), "producer did not run while consumer held a lease"
        )
        self.assertEqual(first.value.item(), 0)
        queue.release(first, consumed=True)
        for expected in range(1, 4):
            lease = queue.acquire()
            self.assertEqual(lease.value.item(), expected)
            queue.release(lease, consumed=True)
        with self.assertRaises(StopIteration):
            queue.acquire()
        queue.stop()
        metrics = queue.metrics()
        self.assertEqual(metrics["consumed"], 4)
        self.assertEqual(metrics["resident_bytes"], 0)
        self.assertLessEqual(metrics["peak_bytes"], 24)
        self.assertFalse(metrics["producer_alive"])

    def test_counts_inflight_capture_and_leases_before_allocating(self):
        entered, finish = threading.Event(), threading.Event()
        queue = self.queue(size=8, items=1)

        def capture(_):
            entered.set()
            if not finish.wait(2):
                raise TimeoutError("test capture was not released")
            return torch.ones(2)

        queue.start(range(2), capture, lambda _: 8)
        self.assertTrue(entered.wait(2))
        self.assertEqual(queue.metrics()["resident_bytes"], 8)
        self.assertEqual(queue.metrics()["resident_items"], 1)
        finish.set()
        lease = queue.acquire()
        with self.assertRaisesRegex(ValueError, "accumulation window"):
            queue.acquire()
        queue.release(lease, consumed=True)
        lease = queue.acquire()
        queue.release(lease, consumed=True)

    def test_capture_failure_reaches_consumer_with_original_cause(self):
        queue = self.queue()

        def capture(_):
            raise LookupError("teacher failed")

        queue.start([0], capture, lambda _: 8)
        with self.assertRaisesRegex(RuntimeError, "capture failed") as raised:
            queue.acquire()
        self.assertIsInstance(raised.exception.__cause__, LookupError)
        queue.clear()
        self.assertFalse(queue.metrics()["producer_alive"])
        self.assertEqual(queue.metrics()["resident_bytes"], 0)

    def test_stop_unblocks_full_producer_and_preserves_uncommitted_lease(self):
        queue = self.queue(size=8, items=1)
        queue.start(range(10), lambda _: torch.ones(2), lambda _: 8)
        lease = queue.acquire()
        queue.stop()
        self.assertFalse(queue.metrics()["producer_alive"])
        self.assertEqual(queue.metrics()["resident_bytes"], 8)
        self.assertEqual(queue.metrics()["consumed"], 0)
        queue.release(lease, consumed=False)
        self.assertEqual(queue.metrics()["resident_bytes"], 0)

    def test_pause_blocks_capture_and_can_be_nested(self):
        entered = threading.Event()
        queue = self.queue()

        def capture(_):
            entered.set()
            return torch.ones(2)

        with queue.pause(), queue.pause():
            queue.start([0], capture, lambda _: 8)
            self.assertFalse(entered.wait(0.05))
        self.assertTrue(entered.wait(2))
        lease = queue.acquire()
        queue.release(lease, consumed=True)

    def test_can_stop_producer_inside_a_paused_memory_handoff(self):
        queue = self.queue()
        queue.start(range(20), lambda _: torch.ones(2), lambda _: 8)
        lease = queue.acquire()
        queue.release(lease, consumed=True)
        with queue.pause():
            queue.stop()
            self.assertFalse(queue.metrics()["producer_alive"])
        self.assertEqual(queue.metrics()["resident_bytes"], 0)

    def test_stop_waits_for_active_capture_before_freeing(self):
        entered, finish, closed = (threading.Event() for _ in range(3))
        queue = self.queue()

        def capture(_):
            entered.set()
            if not finish.wait(2):
                raise TimeoutError("test capture was not released")
            return torch.ones(2)

        queue.start([0], capture, lambda _: 8)
        self.assertTrue(entered.wait(2))
        closer = threading.Thread(target=lambda: (queue.clear(), closed.set()))
        closer.start()
        try:
            self.assertFalse(closed.wait(0.05))
        finally:
            finish.set()
            closer.join(2)
        self.assertTrue(closed.is_set())
        self.assertEqual(queue.metrics()["resident_bytes"], 0)

    def test_rejects_oversize_and_underestimated_storage(self):
        for estimate in (4, 32):
            with self.subTest(estimate=estimate):
                queue = self.queue(size=16)
                queue.start([0], lambda _: torch.ones(2), lambda _: estimate)
                with self.assertRaises(RuntimeError) as raised:
                    queue.acquire()
                self.assertIsInstance(raised.exception.__cause__, ValueError)
                queue.clear()
                self.assertEqual(queue.metrics()["resident_bytes"], 0)

    def test_restart_retains_consumption_and_discards_lookahead(self):
        queue = self.queue()
        queue.start(range(6), lambda i: torch.tensor([i]), lambda _: 8)
        lease = queue.acquire()
        queue.release(lease, consumed=True)
        queue.stop()
        queue.start(range(1, 3), lambda i: torch.tensor([i]), lambda _: 8)
        for expected in (1, 2):
            lease = queue.acquire()
            self.assertEqual(lease.value.item(), expected)
            queue.release(lease, consumed=True)
        queue.stop()
        self.assertEqual(queue.metrics()["consumed"], 3)
