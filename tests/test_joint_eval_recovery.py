"""A failed held-out pass must retain the just-consumed optimizer update."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.training import joint_family
from tests.test_joint_deadline import stages_for
from tests.test_joint_online_cpu import online_fixture


class JointEvalRecoveryTest(unittest.TestCase):
    def test_eval_failure_preserves_optimizer_and_consumption_cursor(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with (
                tempfile.TemporaryDirectory() as folder,
                mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}),
            ):
                root = Path(folder)
                online_fixture(root, False)
                output = root / "run"
                stages = stages_for(root, output)
                for stage in stages:
                    stage.config.training.eval_interval = 1
                original = joint_family._evaluate
                calls = 0

                def fail_milestone(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("held-out capture failed")
                    return original(*args, **kwargs)

                with mock.patch.object(
                    joint_family, "_evaluate", side_effect=fail_milestone
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "held-out capture failed"
                    ):
                        joint_family.run_joint_family(output, stages)
                checkpoint = output / "joint-step1.pt"
                self.assertTrue(
                    checkpoint.is_file(), "successful update lost before evaluation"
                )
                state = torch.load(checkpoint, weights_only=True)
                self.assertEqual(
                    (state["step"], state["epoch"], state["next_batch"]), (1, 0, 2)
                )
                self.assertEqual(state["consumed_samples"], 2)
                self.assertTrue(state["optimizer"])
        finally:
            torch.set_num_threads(threads)
