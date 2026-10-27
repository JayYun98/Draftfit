"""Real tiny Gemma orchestration plus receipt and deadline boundaries."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_humanizer_campaign as runner
from tests.test_joint_online_cpu import online_fixture

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_humanizer_campaign.py"


class HumanizerCampaignRunner(unittest.TestCase):
    def test_generation_receives_only_remaining_cumulative_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                root=root,
                train_tokens=None,
                train_input=root / "train.inputs.jsonl",
                train_reserve=[],
                stop_at=None,
                deadline_unix=None,
                generation_batch_size=16,
                hourly_usd=3600.0,
                generation_budget_usd=1.1,
                prior_spend_usd=2.8,
                budget_usd=3.0,
            )
            with (
                patch.object(runner.time, "monotonic", return_value=100.1),
                patch.object(runner.time, "time", return_value=1000.0),
                patch.object(
                    runner.subprocess,
                    "run",
                    return_value=argparse.Namespace(returncode=0),
                ) as launched,
                patch.object(runner, "count_tokens"),
            ):
                runner.token_source(args, "train", 30000, 100.0)
            command = launched.call_args.args[0]
            self.assertAlmostEqual(
                float(command[command.index("--max-projected-usd") + 1]), 0.1
            )
            self.assertAlmostEqual(
                float(command[command.index("--deadline-unix") + 1]), 1000.1
            )
            with (
                patch.object(runner.time, "monotonic", return_value=100.2),
                patch.object(runner.subprocess, "run") as blocked,
            ):
                with self.assertRaises(TimeoutError):
                    runner.token_source(args, "train", 30000, 100.0)
                blocked.assert_not_called()

    def test_nonfinite_or_exhausted_budget_rejected_before_work(self):
        for option, value in (
            ("--budget-usd", "nan"),
            ("--budget-usd", "inf"),
            ("--prior-spend-usd", "nan"),
            ("--hourly-usd", "inf"),
            ("--generation-budget-usd", "nan"),
            ("--deadline-unix", "inf"),
            ("--prior-spend-usd", "3"),
        ):
            with (
                self.subTest(option=option, value=value),
                patch.object(runner, "run") as run,
            ):
                with self.assertRaises(SystemExit) as error:
                    runner.main(
                        [
                            "--tiny",
                            "--root",
                            "/tmp/unused-humanizer-budget-test",
                            "--manifest",
                            "/tmp/unused-manifest",
                            option,
                            value,
                        ]
                    )
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def invoke(self, root, *extra):
        env = os.environ.copy()
        env.update(
            SPECFORGE_DEVICE="cpu",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONPATH=str(SCRIPT.parents[1]),
        )
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(root / "campaign"),
                "--tiny",
                "--manifest",
                str(root / "family.json"),
                "--budget-usd",
                "100",
                *extra,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )

    def make_fixture(self, root):
        online_fixture(root, seeded=True, gemma=True)
        manifest = json.loads((root / "family.json").read_text())
        for row, name in zip(manifest["runs"], ("dspark", "dflash", "dflash2")):
            row["name"] = name
        (root / "family.json").write_text(json.dumps(manifest))
        rows = (root / "train.jsonl").read_text().splitlines()
        (root / "train.jsonl").write_text("\n".join(rows * 3) + "\n")

    def test_tiny_real_train_resume_export_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            result = self.invoke(root)
            worker_log = root / "campaign/run/attempt-5/worker.log"
            self.assertEqual(
                result.returncode,
                0,
                result.stdout
                + result.stderr
                + (worker_log.read_text() if worker_log.exists() else ""),
            )
            receipt = json.loads((root / "campaign/status.json").read_text())
            self.assertEqual(
                (receipt["status"], receipt["completed_step"]), ("complete", 10)
            )
            self.assertEqual(
                {arm["name"] for arm in receipt["arms"]},
                {"dspark", "dflash", "dflash2"},
            )
            first = root / "campaign/run/attempt-5/joint-step5.pt"
            second = root / "campaign/run/attempt-10/joint-step10.pt"
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            resumed = json.loads((second.parent / "worker-result.json").read_text())
            self.assertEqual({run["global_step"] for run in resumed["runs"]}, {10})
            rows = (root / "train.jsonl").read_text().splitlines()
            changed = json.loads(rows[0])
            changed["input_ids"][0] = 34
            rows[0] = json.dumps(changed)
            (root / "train.jsonl").write_text("\n".join(rows) + "\n")
            reused = self.invoke(root)
            self.assertNotEqual(reused.returncode, 0)
            self.assertEqual(
                json.loads((root / "campaign/status.json").read_text())["status"],
                "failed",
            )

    def test_expired_deadline_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            result = self.invoke(root, "--deadline-unix", str(time.time() - 1))
            self.assertNotEqual(result.returncode, 0)
            receipt = json.loads((root / "campaign/status.json").read_text())
            self.assertEqual(receipt["status"], "stopped")
            self.assertEqual(receipt["completed_step"], 0)
            self.assertEqual(receipt["arms"], [])

    def test_invalid_manifest_cannot_publish_ready_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            manifest = json.loads((root / "family.json").read_text())
            manifest["runs"].pop()
            (root / "family.json").write_text(json.dumps(manifest))
            result = self.invoke(root)
            self.assertNotEqual(result.returncode, 0)
            receipt = json.loads((root / "campaign/status.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse((root / "campaign/exports").exists())


if __name__ == "__main__":
    unittest.main()
