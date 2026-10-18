"""Real CPU controller stop replay, including repeated failure after final save."""
from pathlib import Path
import unittest
from unittest import mock

import torch

from draftfit.optimizer import BF16Optimizer
from draftfit.training import joint_family
from draftfit.unattended import Controller, read, run, sha
from tests import test_unattended_controller as fixtures


class JointStopReplayTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        self.fixture = fixtures.UnattendedControllerTest()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.path = self.fixture.path
        self.state_path = self.fixture.root / "unattended/state.json"

    def assert_exact(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_exact(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_exact(left, right)
        else:
            self.assertEqual(expected, actual)

    def interrupt_committed_stop(self):
        observe = Controller.observe

        def interrupt(controller, **kwargs):
            response = observe(controller, **kwargs)
            if response["stop"]:
                raise RuntimeError("after committed stop")
            return response

        with mock.patch.object(Controller, "observe", interrupt):
            with self.assertRaisesRegex(RuntimeError, "after committed stop"):
                run(self.path)
        state = read(self.state_path)
        self.assertEqual(state["total_steps"], 2)
        self.assertTrue(state["phases"][0]["last_observation"]["response"]["stop"])
        checkpoint = Path(state["phases"][0]["checkpoint"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual((saved["epoch"], saved["next_batch"]), (1, 0))
        return state, saved

    def assert_no_new_decisions(self, before, after):
        for key in ("total_steps", "history", "best", "cost_floor", "exploration_floor",
                    "external_step_floor", "tried", "resource_state", "resource_history",
                    "resource_reports", "publication_jobs"):
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(len(before["phases"]), len(after["phases"]))
        for key in ("admitted_until", "decisions", "branch", "last_observation"):
            self.assertEqual(before["phases"][0][key], after["phases"][0][key], key)

    def assert_checkpoint_unchanged(self, before, saved, path):
        actual = torch.load(path, map_location="cpu", weights_only=True)
        # Includes cursor, consumed samples, FP32 masters, Adam, scheduler, RNG
        # and every shared/private draft tensor, not just the global step.
        self.assert_exact(saved, actual)
        self.assertEqual(sha(path), before["phases"][0]["checkpoint_sha256"])

    def test_stop_replay_preserves_exact_checkpoint_and_decision_identity(self):
        before, saved = self.interrupt_committed_stop()
        with mock.patch.object(BF16Optimizer, "step", side_effect=AssertionError("unexpected update")):
            result = run(self.path, resume=True)
        self.assert_checkpoint_unchanged(before, saved, result["phases"][0]["resume_checkpoint"])
        self.assert_no_new_decisions(before, result)
        self.assertEqual(result["status"], "complete")
        again = run(self.path, resume=True)
        self.assertEqual(again, result)

    def test_repeated_failure_after_trainer_final_checkpoint_is_idempotent(self):
        before, saved = self.interrupt_committed_stop()
        original = joint_family.run_joint_family
        outcomes = []

        def interrupt_return(*args, **kwargs):
            # All real loading, native verification, final checkpoint handling
            # and queue shutdown happen before this controller crash boundary.
            outcomes.append(original(*args, **kwargs))
            raise RuntimeError("after final checkpoint before phase completion")

        with mock.patch.object(BF16Optimizer, "step", side_effect=AssertionError("unexpected update")):
            for attempt in range(2):
                with mock.patch.object(joint_family, "run_joint_family", interrupt_return):
                    with self.assertRaisesRegex(RuntimeError, "after final checkpoint"):
                        run(self.path, resume=True)
                self.assertEqual(len(outcomes), attempt + 1)
                self.assert_checkpoint_unchanged(before, saved, outcomes[-1]["runs"][0]["joint_checkpoint"])
                self.assert_no_new_decisions(before, read(self.state_path))
            result = run(self.path, resume=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["phases"][0]["attempts"]), 4)
        self.assert_checkpoint_unchanged(before, saved, result["phases"][0]["resume_checkpoint"])
        self.assert_no_new_decisions(before, result)


if __name__ == "__main__":
    unittest.main()
