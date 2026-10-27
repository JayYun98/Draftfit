"""Real CLI batch probes and measured forecast behavior on a tiny frozen target."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from draftfit.calibrate import batch_probe, forecast
from draftfit.training.hardware_tuning import candidates, choose


class CalibrationTests(unittest.TestCase):
    def test_unmeasured_scheduled_overhead_cannot_admit_work(self):
        profile = dict(
            step_seconds=[1.0] * 5,
            checkpoint_seconds=[],
            evaluation_seconds=[],
            verification_seconds=[],
        )
        settings = dict(
            current_step=5,
            target_step=10,
            available_seconds=6.0,
            overhead_seconds=0.0,
            safety_factor=1.0,
            checkpoint_interval=10,
            evaluation_interval=10,
            verification_interval=10,
        )
        result = forecast(profile, **settings)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "unmeasured_recurring_cost")
        self.assertIsNone(result["guard_estimated_seconds"])
        self.assertIsNone(result["max_feasible_step"])
        bounded = forecast(
            profile,
            **settings,
            checkpoint_seconds=2.0,
            evaluation_seconds=2.0,
            verification_seconds=2.0,
        )
        self.assertEqual(bounded["guard_estimated_seconds"], 11.0)
        self.assertFalse(bounded["feasible"])

    def test_queue_incompatible_candidates_do_not_abort_larger_valid_probes(self):
        from tests.test_joint_online_cpu import online_fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = online_fixture(root, seeded=False)
            base = json.loads((root / "base.json").read_text())
            base["training"].update(
                batch_size=4,
                num_epochs=2,
                joint_feature_queue_bytes=1024 * 1024,
                joint_feature_queue_batches=2,
            )
            (root / "base.json").write_text(json.dumps(base))
            (root / "family.json").write_text(json.dumps(manifest))
            with patch.dict(
                os.environ,
                {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            ):
                receipt = batch_probe(
                    root / "family.json",
                    root / "probe",
                    maximum=4,
                    effective_batch=4,
                    probe_steps=2,
                    max_seconds=90,
                )
            self.assertEqual(
                [r["status"] for r in receipt["trials"]], ["incompatible", "ok", "ok"]
            )
            self.assertIn(receipt["selected"]["batch_size"], (2, 4))

    def test_candidates_keep_effective_batch_and_selector_rejects_bad_memory(self):
        self.assertEqual(candidates(16, 12), [1, 2, 4])
        rows = [
            dict(
                status="ok",
                batch_size=2,
                samples_per_second=100.0,
                peak_bytes=10.0,
                memory_budget_bytes=20.0,
            ),
            dict(
                status="ok",
                batch_size=4,
                samples_per_second=102.0,
                peak_bytes=15.0,
                memory_budget_bytes=20.0,
            ),
            dict(
                status="ok",
                batch_size=8,
                samples_per_second=200.0,
                peak_bytes=30.0,
                memory_budget_bytes=20.0,
            ),
        ]
        self.assertEqual(choose(rows)["batch_size"], 2)
        for bad in (float("inf"), float("nan"), -1, True):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                choose([dict(rows[0], peak_bytes=bad)])

    def test_forecast_counts_cost_reserve_and_periodic_work(self):
        profile = dict(
            step_seconds=[2.0] * 5,
            checkpoint_seconds=[1.0],
            evaluation_seconds=[3.0],
            verification_seconds=[4.0],
        )
        result = forecast(
            profile,
            current_step=5,
            target_step=100,
            available_seconds=3600.0,
            overhead_seconds=30.0,
            hourly_usd=1.0,
            remaining_usd=0.05,
            transfer_reserve_usd=0.01,
            checkpoint_interval=10,
            evaluation_interval=10,
            verification_interval=10,
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["available_seconds"], 144.0)
        self.assertLess(result["max_feasible_step"], 100)
        self.assertGreater(result["estimated_cost_usd"], 0.05)

    def test_real_batch_probes_leave_target_and_training_checkpoint_untouched(self):
        import torch

        from tests.test_joint_online_cpu import online_fixture

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = online_fixture(root, seeded=False)
            base = json.loads((root / "base.json").read_text())
            base["training"]["num_epochs"] = 2
            (root / "base.json").write_text(json.dumps(base))
            before = (root / "target/model.safetensors").read_bytes()
            env = {
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "SPECFORGE_DEVICE": "cpu",
            }
            with patch.dict(os.environ, env):
                result = batch_probe(
                    root / "family.json",
                    root / "probes",
                    maximum=2,
                    effective_batch=2,
                    probe_steps=2,
                    max_seconds=60.0,
                )
            self.assertEqual([r["batch_size"] for r in result["trials"]], [1, 2])
            self.assertTrue(all(r["samples_per_second"] > 0 for r in result["trials"]))
            self.assertEqual(result["effective_batch_size"], 2)
            self.assertEqual(before, (root / "target/model.safetensors").read_bytes())
            self.assertFalse(list((root / "probes").rglob("joint-step*.pt")))
            self.assertFalse((root / "outputs").exists())
            proc = subprocess.run(
                [sys.executable, "-m", "draftfit.cli", "calibrate", "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("forecast", proc.stdout)


if __name__ == "__main__":
    unittest.main()
