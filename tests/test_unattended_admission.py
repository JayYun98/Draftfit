"""Admission/recovery regressions with real tiny CPU updates and local fixtures.

Resource telemetry alone is synthetic; its adapter and policy are production code.
Reuse the controller fixture by composition, without collecting its test methods.
"""

import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import torch

from draftfit.training import joint_family
from draftfit.unattended import Controller, load_plan, read, run, write
from tests import test_unattended_controller as fixtures


class UnattendedAdmissionTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        fixtures.UnattendedControllerTest.setUp(self)

    def external_budget(self, steps):
        write(
            self.root / "budget.json",
            dict(timestamp_unix=time.time(), spent_usd=0, steps_used=steps),
        )
        # The operational stop must clip a two-step verification interval to one.
        self.config["policy"]["next_steps"] = 2
        write(self.path, self.config)
        base = read(self.root / "base.json")
        base["training"]["joint_verification"]["interval"] = 2
        write(self.root / "base.json", base)

    def test_one_remaining_external_step_runs_exactly_one_update(self):
        self.external_budget(7)
        with mock.patch.object(
            joint_family, "_backward_window", wraps=joint_family._backward_window
        ) as backward:
            result = run(self.path)
        self.assertEqual(backward.call_count, 1)
        self.assertEqual(result["total_steps"], 1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["external_step_floor"], 7)
        checkpoint = torch.load(
            result["phases"][0]["resume_checkpoint"], weights_only=True
        )
        self.assertEqual(checkpoint["step"], 1)
        # Operational clipping must not shorten the optimizer/scheduler horizon.
        self.assertEqual(checkpoint["horizon"], 4)
        for history in result["history"].values():
            self.assertEqual([row["artifact"]["step"] for row in history], [1])
        self.assertEqual(read(self.root / "budget.json")["steps_used"], 7)

        # A lagging external snapshot must not allow the same step a second time.
        with mock.patch.object(
            joint_family, "run_joint_family", wraps=joint_family.run_joint_family
        ) as train:
            resumed = run(self.path, resume=True)
        train.assert_not_called()
        self.assertEqual(resumed["total_steps"], 1)

    def test_exhausted_external_steps_never_enter_training(self):
        self.external_budget(8)
        with mock.patch.object(
            joint_family, "run_joint_family", wraps=joint_family.run_joint_family
        ) as train:
            result = run(self.path)
        train.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["phases"], [])
        self.assertEqual(result["history"], {})
        self.assertEqual(result["cost_floor"], 0)
        self.assertFalse(list((self.root / "unattended").rglob("joint-step*.pt")))

    def assert_tree_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                with self.subTest(key=key):
                    self.assert_tree_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    @staticmethod
    def history_evidence(state):
        # Wall-clock timings and serialized checkpoint hashes differ across runs.
        # Compare the actual policy observations: steps, losses, parity and counts.
        return {
            name: [
                dict(
                    step=row["artifact"]["step"],
                    loss=row["loss"],
                    parity=row["serving"]["exact_token_parity"],
                    domains=[
                        {
                            k: v
                            for k, v in domain.items()
                            if k not in ("seconds", "baseline_seconds")
                        }
                        for domain in row["serving"]["domains"]
                    ],
                )
                for row in history
            ]
            for name, history in state["history"].items()
        }

    def test_step_zero_resource_resume_matches_uninterrupted_training(self):
        self.config["resources"] = dict(policy={}, pilot_seconds={"initial": 2})
        self.config["output_dir"] = "full"
        write(self.path, self.config)
        full = run(self.path)
        self.config["output_dir"] = "interrupted"
        write(self.path, self.config)
        with mock.patch.object(
            joint_family,
            "_backward_window",
            side_effect=RuntimeError("before first backward"),
        ) as backward:
            with self.assertRaisesRegex(RuntimeError, "before first backward"):
                run(self.path)
        backward.assert_called_once()
        failed = read(self.root / "interrupted/state.json")
        self.assertEqual(failed["history"], {})
        seed_path = Path(failed["phases"][0]["attempts"][0]) / "joint-step0.pt"
        seed = torch.load(seed_path, weights_only=True)
        self.assertEqual(seed["step"], 0)
        self.assertEqual(seed["optimizer"]["optimizer_state_dict"]["state"], {})

        resumed = run(self.path, resume=True)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(full["total_steps"], 2)
        self.assertEqual(resumed["total_steps"], full["total_steps"])
        self.assertEqual(self.history_evidence(resumed), self.history_evidence(full))
        for history in resumed["history"].values():
            self.assertEqual([row["artifact"]["step"] for row in history], [1, 2])
        self.assertEqual(
            len(resumed["resource_reports"]), len(full["resource_reports"])
        )
        self.assertTrue(
            all(
                row["status"] == "measurement_unavailable"
                for row in resumed["resource_reports"]
            )
        )
        self.assertEqual(resumed["cost_floor"], full["cost_floor"])
        expected = torch.load(full["phases"][0]["resume_checkpoint"], weights_only=True)
        actual = torch.load(
            resumed["phases"][0]["resume_checkpoint"], weights_only=True
        )
        for key in (
            "drafts",
            "optimizer",
            "rng",
            "step",
            "epoch",
            "next_batch",
            "consumed_samples",
        ):
            with self.subTest(checkpoint_field=key):
                self.assert_tree_equal(expected[key], actual[key])

    def test_resource_adapter_uses_recent_window_after_stale_observation(self):
        self.config["presets"] = [
            dict(
                id="batch-up",
                when=["loss_plateau"],
                reason="pilot",
                overrides=["training.batch_size=2"],
                pilot_steps=2,
                pilot_cost_usd=0.1,
            )
        ]
        self.config["resources"] = dict(
            policy=dict(window_size=2, max_lag_steps=1, cooldown_steps=0),
            pilot_seconds={"initial": 2, "batch-up": 2},
        )
        write(self.path, self.config)
        config, stages, binding = load_plan(self.path)
        controller = Controller(config, stages, binding)
        self.addCleanup(controller.lock.close)
        self.addCleanup(controller.publisher.close)
        controller.state["phases"].append(dict(id="phase0", preset="initial"))
        controller.resource_started = time.monotonic()
        controller.resource_step = 0
        controller.sampler = mock.Mock()
        metrics = {stage.name: {"eval/avg_loss": 1.0} for stage in stages}
        report = dict(artifact_sha256="a" * 64)
        for step in (1, 2, 3):
            observation = dict(
                phase_id="phase0",
                preset="initial",
                step=step,
                throughput=10,
                loss=1,
                peak_vram_bytes=80 if step == 1 else 20,
                capacity_bytes=100,
                utilization=0.8 if step == 1 else 0.2,
            )
            controller.sampler.summarize.return_value = dict(
                status="available", observation=deepcopy(observation)
            )
            controller.state["total_steps"] = step
            branch = controller.observe_resources(step, metrics, report)
            if step < 3:
                self.assertIsNone(branch)
        self.assertIsNotNone(branch)
        self.assertEqual(branch["next_preset"], "batch-up")
        self.assertEqual(branch["reason"], "underutilized_window")
        self.assertEqual(branch["source_artifact"], report)
        self.assertEqual(
            controller.state["phases"][0]["resource_decision"]["rejected"], []
        )
        # Filtering the decision window must preserve the audit history.
        self.assertEqual(
            [r["step"] for r in controller.state["resource_history"]], [1, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
