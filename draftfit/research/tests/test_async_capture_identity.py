"""Policy history from synchronous capture must survive an upgrade."""

import unittest

from draftfit.config import TrainingConfig
from draftfit.training.joint_control import training_fingerprint
from draftfit.training.joint_policy import decide
from tests.test_joint_policy import observation, policy

# Origin/dev ccd27e4, DSpark with the bounded online queue below.
SYNC_FINGERPRINT = "7b1726e66105f48004b918fbbbde52b65333ea75f5d47e742ac2756781576969"


class AsyncCaptureIdentityTests(unittest.TestCase):
    def test_historical_plateau_applies_only_to_synchronous_capture(self):
        for capture in (
            {},
            {"joint_async_capture": False},
            {"joint_async_capture": True},
        ):
            with self.subTest(capture=capture):
                training = TrainingConfig(
                    strategy="dspark",
                    joint_online_capture=True,
                    joint_feature_queue_bytes=32768,
                    joint_feature_queue_batches=3,
                    **capture,
                )
                rows = [observation(step, loss=1.0) for step in (1, 2, 3)]
                for row in rows[:-1]:
                    row["artifact"]["training_fingerprint"] = SYNC_FINGERPRINT
                current = rows[-1]["artifact"]
                current["training_fingerprint"] = training_fingerprint(training)
                result = decide(policy(patience=2), rows, current=current, budget={})
                if training.joint_async_capture:
                    self.assertEqual(result["action"], "continue")
                    self.assertEqual(result["plateau"]["loss"]["stale"], 0)
                    self.assertNotEqual(
                        current["training_fingerprint"], SYNC_FINGERPRINT
                    )
                else:
                    self.assertEqual(
                        (result["action"], result["reason"]), ("stop", "loss_plateau")
                    )
                    self.assertEqual(result["rejected"], [])
                    self.assertEqual(current["training_fingerprint"], SYNC_FINGERPRINT)
