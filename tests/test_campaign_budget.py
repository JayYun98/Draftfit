import json
import tempfile
import unittest
from pathlib import Path

from draftfit.training.campaign_budget import (
    estimate_runtime,
    read_capture_profile,
    read_joint_profile,
)


class CampaignBudgetTests(unittest.TestCase):
    def test_profile_uses_capture_inclusive_time_not_optimizer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning-curves.jsonl"
            rows = [{"kind": "train", "step": 1, "joint_step_seconds": 0.4}]
            rows.extend(
                {
                    "kind": "capture",
                    "run_id": "one",
                    "step": step,
                    "step_with_capture_seconds": seconds,
                }
                for step, seconds in enumerate((1.0, 1.2, 1.4, 1.6, 1.8), 1)
            )
            path.write_text("\n".join(json.dumps(row) for row in rows))
            samples = read_capture_profile(path, through_step=5)
            self.assertEqual(samples, [1.0, 1.2, 1.4, 1.6, 1.8])
            result = estimate_runtime(
                samples=samples,
                current_step=5,
                target_step=5000,
                available_seconds=600,
                overhead_seconds=60,
                safety_factor=1.25,
                minimum_samples=5,
            )
            self.assertEqual(result["p50_step_seconds"], 1.4)
            self.assertEqual(result["p90_step_seconds"], 1.8)
            self.assertEqual(result["metric"], "step_with_capture_seconds")
            self.assertFalse(result["feasible"])
            self.assertEqual(result["reason"], "runtime_infeasible")
            self.assertLess(result["max_feasible_step"], 5000)

    def test_insufficient_profile_fails_closed_and_reached_target_needs_no_samples(
        self,
    ):
        result = estimate_runtime(
            samples=[1.0],
            current_step=1,
            target_step=100,
            available_seconds=3600,
            overhead_seconds=10,
            minimum_samples=5,
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "insufficient_profile")
        reached = estimate_runtime(
            samples=[],
            current_step=100,
            target_step=100,
            available_seconds=60,
            overhead_seconds=10,
        )
        self.assertTrue(reached["feasible"])
        self.assertEqual(reached["reason"], "target_already_reached")

    def test_invalid_profile_and_parameters_are_rejected(self):
        for sample in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                estimate_runtime(
                    samples=[sample],
                    current_step=0,
                    target_step=10,
                    available_seconds=100,
                    overhead_seconds=0,
                    minimum_samples=1,
                )
        for factor in (0.99, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                estimate_runtime(
                    samples=[1.0],
                    current_step=0,
                    target_step=10,
                    available_seconds=100,
                    overhead_seconds=0,
                    safety_factor=factor,
                    minimum_samples=1,
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.jsonl"
            path.write_text(
                "\n".join(
                    map(
                        json.dumps,
                        [
                            {
                                "kind": "capture",
                                "run_id": "one",
                                "step": 1,
                                "step_with_capture_seconds": 1.0,
                            },
                            {
                                "kind": "capture",
                                "run_id": "two",
                                "step": 2,
                                "step_with_capture_seconds": 1.0,
                            },
                        ],
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "mixes monitor runs"):
                read_capture_profile(path)
            path.write_text(
                json.dumps(
                    {
                        "kind": "capture",
                        "run_id": "one",
                        "step": 3,
                        "step_with_capture_seconds": 1.0,
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "exceeds completed profile step"):
                read_capture_profile(path, through_step=2)

    def test_milestone_events_are_union_counted_and_use_measured_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning-curves.jsonl"
            rows = []
            checkpoint_steps = {4, 5, 6, 8, 10}  # save=6, eval=4, verify=5, final=10
            evaluation_steps = {4, 5, 8, 10}
            verification_steps = {5, 10}
            # The trainer's one-time initial evaluation precedes its seed save.
            rows.extend(
                [
                    {
                        "kind": "evaluation_capture",
                        "run_id": "one",
                        "step": 0,
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "kind": "checkpoint",
                        "run_id": "one",
                        "step": 0,
                        "elapsed_seconds": 2.0,
                    },
                ]
            )
            for step in range(3, 11):
                if step in checkpoint_steps:
                    elapsed = float(step * 2)
                    rows.append(
                        {
                            "kind": "capture",
                            "run_id": "one",
                            "step": step,
                            "elapsed_seconds": elapsed,
                            "step_with_capture_seconds": 2.0,
                        }
                    )
                    rows.append(
                        {
                            "kind": "checkpoint",
                            "run_id": "one",
                            "step": step,
                            "elapsed_seconds": elapsed + 0.2,
                        }
                    )
                    if step in evaluation_steps:
                        rows.append(
                            {
                                "kind": "evaluation_capture",
                                "run_id": "one",
                                "step": step,
                                "elapsed_seconds": elapsed + 0.6,
                            }
                        )
                    if step in verification_steps:
                        rows.append(
                            {
                                "kind": "generation_verification",
                                "run_id": "one",
                                "step": step,
                                "elapsed_seconds": elapsed + 1.3,
                                "verification_seconds": 0.7,
                            }
                        )
            path.write_text("\n".join(json.dumps(row) for row in rows))
            profile = read_joint_profile(path, through_step=10)
            self.assertEqual(profile["step_seconds"], [2.0] * 5)
            self.assertEqual(len(profile["checkpoint_seconds"]), 5)
            for seconds in profile["checkpoint_seconds"]:
                self.assertAlmostEqual(seconds, 0.2)
            self.assertAlmostEqual(min(profile["evaluation_seconds"]), 0.4)
            self.assertEqual(profile["verification_seconds"], [0.7, 0.7])
            result = estimate_runtime(
                samples=profile["step_seconds"],
                current_step=2,
                target_step=10,
                available_seconds=10_000,
                overhead_seconds=100,
                minimum_samples=5,
                safety_factor=1.25,
                checkpoint_interval=6,
                checkpoint_seconds=0.2,
                evaluation_interval=4,
                evaluation_seconds=0.4,
                verification_interval=5,
                verification_seconds=0.7,
            )
            self.assertEqual(result["checkpoint_events"], 5)
            self.assertEqual(result["evaluation_events"], 4)
            self.assertEqual(result["verification_events"], 2)
            self.assertAlmostEqual(result["measured_overhead_seconds"], 4.0)

    def test_quantile_rejects_nonfinite_duration(self):
        from draftfit.training.campaign_budget import conservative_quantile

        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                conservative_quantile([1.0, value])


if __name__ == "__main__":
    unittest.main()
