"""Restart and budget behavior with the real family config/feature planner."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from draftfit.inference.capture_manifest import (
    build_capture_manifest,
    write_capture_manifest,
)
from draftfit.training.model_loading import convert_family_draft_config
from scripts.run_lfm26_spot_family import _last_metrics, _run_train, run
from tests.test_dflash2_integration import tiny_config


class SpotFamilyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = convert_family_draft_config(tiny_config(), "dspark")
        source.dflash_config = {**source.dflash_config, "sample_from_anchor": True}
        (self.root / "official").mkdir()
        source.to_json_file(str(self.root / "official" / "config.json"))
        capture = build_capture_manifest(
            strategy="dspark",
            capture_method="target_hidden",
            capture_layers=(1, 5),
            feature_names=(
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            ),
            target_model="target/model",
            target_revision="0123456789abcdef0123456789abcdef01234567",
            target_feature="target_last_hidden_states",
            target_repr="hidden_state",
        )
        for split in ("train", "holdout"):
            (self.root / split).mkdir()
            write_capture_manifest(
                str(self.root / split / "capture_manifest.json"), capture
            )
        (self.root / "base.json").write_text(
            json.dumps(
                {
                    "model": {
                        "target_model_path": "target/model",
                        "draft_checkpoint_path": str(self.root / "official"),
                    },
                    "data": {
                        "hidden_states_path": str(self.root / "train"),
                        "eval_hidden_states_path": str(self.root / "holdout"),
                    },
                    "training": {
                        "strategy": "dspark",
                        "eval_interval": 1,
                        "save_interval": 1,
                        "max_steps": 2,
                    },
                }
            )
        )
        self.manifest = self.root / "family.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "base_config": "base.json",
                    "output_dir": "results",
                    "runs": [
                        {"name": "warm", "strategy": "dspark"},
                        {"name": "continual", "strategy": "dspark", "from": "warm"},
                        {
                            "name": "dflash",
                            "strategy": "dflash",
                            "from": "warm",
                            "overrides": ["training.retrain_anchor_alignment=true"],
                        },
                        {
                            "name": "dflash2",
                            "strategy": "dflash2",
                            "from": "warm",
                            "overrides": ["training.retrain_anchor_alignment=true"],
                        },
                    ],
                }
            )
        )

    def call(self, **kwargs):
        return run(
            self.manifest,
            hourly_usd=0.2,
            max_usd=0.5,
            lease_start_epoch=time.time(),
            stop_reserve_seconds=0,
            **kwargs,
        )

    def test_plan_is_read_only_and_budget_stop_skips_training(self):
        self.call(plan=True)
        self.assertFalse((self.root / "results").exists())
        with mock.patch("scripts.run_lfm26_spot_family._run_train") as launch:
            result = run(
                self.manifest,
                hourly_usd=0.2,
                max_usd=0.5,
                lease_start_epoch=time.time() - 10000,
                stop_reserve_seconds=0,
            )
        launch.assert_not_called()
        self.assertEqual(result["stages"]["warm"]["status"], "budget_stop")
        self.assertFalse((self.root / "results" / "warm").exists())

    def test_preemption_resumes_then_hands_off_official_warm_weights(self):
        commands = []

        def launch(command, log_path, deadline, lock_fd):
            config = json.loads(Path(command[-1]).read_text())
            name = config["run_id"]
            commands.append(
                (
                    name,
                    config["model"]["draft_checkpoint_path"],
                    config["training"]["resume_from"],
                )
            )
            output = Path(config["output_dir"])
            checkpoint = output / f"{name}-step1"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "training_state.pt").write_bytes(b"fixture")
            latest = output / f"{name}-latest"
            if not latest.exists():
                latest.symlink_to(checkpoint, target_is_directory=True)
            (output / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "step": 1,
                        "eval/avg_loss": 1.0,
                        "eval/avg_acc": 0.5,
                    }
                )
                + "\n"
            )
            if len(commands) == 1:
                return 143, True
            return 0, False

        lease_start = time.time()
        with mock.patch("scripts.run_lfm26_spot_family._run_train", side_effect=launch):
            first = run(
                self.manifest,
                hourly_usd=0.2,
                max_usd=0.5,
                lease_start_epoch=lease_start,
                stop_reserve_seconds=0,
            )
            second = run(
                self.manifest,
                hourly_usd=0.2,
                max_usd=0.5,
                lease_start_epoch=lease_start,
                stop_reserve_seconds=0,
            )
        self.assertEqual(first["stages"]["warm"]["status"], "budget_stop")
        self.assertEqual(
            [name for name, _, _ in commands],
            ["warm", "warm", "continual", "dflash", "dflash2"],
        )
        self.assertEqual(commands[0][1], str(self.root / "official"))
        self.assertIsNone(commands[1][1])
        self.assertTrue(commands[1][2].endswith("warm-latest"))
        expected_parent = str((self.root / "results" / "warm").resolve())
        self.assertEqual(commands[2][1], expected_parent)
        self.assertEqual(commands[3][1], expected_parent)
        self.assertEqual(commands[4][1], expected_parent)
        self.assertEqual(
            {entry["status"] for entry in second["stages"].values()}, {"complete"}
        )
        with mock.patch("scripts.run_lfm26_spot_family._run_train") as again:
            run(
                self.manifest,
                hourly_usd=0.2,
                max_usd=0.5,
                lease_start_epoch=lease_start,
                stop_reserve_seconds=0,
            )
        again.assert_not_called()

    def test_changed_cost_contract_rejected_before_training(self):
        lease_start = time.time() - 10000
        run(
            self.manifest,
            hourly_usd=0.2,
            max_usd=0.5,
            lease_start_epoch=lease_start,
            stop_reserve_seconds=0,
        )
        with self.assertRaisesRegex(ValueError, "cost contract changed"):
            run(
                self.manifest,
                hourly_usd=0.2,
                max_usd=0.6,
                lease_start_epoch=lease_start,
                stop_reserve_seconds=0,
            )

    def test_existing_output_is_untouched_and_torn_metric_row_ignored(self):
        output = self.root / "results"
        output.mkdir()
        sentinel = output / "user.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.call()
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual([path.name for path in output.iterdir()], ["user.txt"])
        metrics = self.root / "metrics.jsonl"
        metrics.write_bytes(
            b'{"step": 1, "eval/avg_loss": 1.0, "eval/avg_acc": 0.5}\n{"step":'
        )
        self.assertEqual(_last_metrics(metrics)["eval/avg_acc"], 0.5)

    def test_real_child_is_stopped_at_deadline_and_log_survives(self):
        lock = self.root / "lock"
        with lock.open("a") as stream:
            code, timed_out = _run_train(
                ["/bin/sh", "-c", "echo started; sleep 5"],
                self.root / "stage.log",
                time.time() + 0.1,
                stream.fileno(),
            )
        self.assertTrue(timed_out)
        self.assertNotEqual(code, 0)
        self.assertIn("started", (self.root / "stage.log").read_text())
        self.assertIn("=== stopped", (self.root / "stage.log").read_text())


if __name__ == "__main__":
    unittest.main()
