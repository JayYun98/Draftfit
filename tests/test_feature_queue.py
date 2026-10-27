import unittest

import torch

from draftfit.training.feature_queue import BoundedFeatureQueue, tensor_bytes


class FeatureQueueTests(unittest.TestCase):
    def test_leases_keep_credit_until_consumption(self):
        q = BoundedFeatureQueue(16, 2)
        q.publish(torch.ones(2))
        a = q.acquire()
        q.publish(torch.ones(2))
        with self.assertRaises(BufferError):
            q.publish(torch.ones(1))
        self.assertEqual(q.resident_bytes, 16)
        q.release(a, consumed=True)
        self.assertIsNone(a.value)
        q.publish(torch.ones(1))
        self.assertEqual(q.consumed, 1)
        with self.assertRaises(ValueError):
            q.release(a, consumed=True)
        q.clear()
        self.assertEqual(q.resident_bytes, 0)
        self.assertEqual(q.consumed, 1)
        self.assertEqual(q.cancelled, 2)

    def test_oversized_batch_and_storage_views(self):
        x = torch.ones(4)
        self.assertEqual(tensor_bytes({"x": x, "view": x[1:]}), 16)
        q = BoundedFeatureQueue(8, 1)
        with self.assertRaises(ValueError):
            q.publish(x)
        self.assertEqual(q.produced, 0)
