"""Monitoring import checks use synthetic timings, not performance evidence."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from draftfit.training.joint_monitor import (
    JointMonitor,
    append,
    bind_manifest,
    ingest_mlx,
)


class MonitorTests(unittest.TestCase):
    def test_periodic_aggregation_and_final_partial_window(self):
        from types import SimpleNamespace as NS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = NS(
                name="ce",
                config=NS(
                    model=NS(draft_checkpoint_path=None),
                    training=NS(strategy="dflash", log_interval=2),
                ),
            )
            monitor = JointMonitor(root, [stage], 0, None)

            def record(step, seconds, loss, final=False):
                monitor.train(
                    step,
                    "ce",
                    final=final,
                    train_loss=loss,
                    train_accuracy=0.5,
                    grad_norm=1,
                    joint_step_seconds=seconds,
                    joint_samples_per_second=2 / seconds,
                    joint_valid_tokens_per_second=10 / seconds,
                    learning_rate=0.001,
                    cuda_peak_allocated_bytes=step * 10,
                    cuda_peak_reserved_bytes=None,
                )

            record(1, 1, 2)
            self.assertFalse((root / "learning-curves.jsonl").exists())
            record(2, 3, 4)
            record(3, 2, 6, final=True)
            rows = [
                json.loads(x)
                for x in (root / "learning-curves.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["step"] for r in rows], [2, 3])
            self.assertEqual(rows[0]["train_loss"], 3)
            self.assertEqual(rows[0]["joint_samples_per_second"], 1)
            self.assertEqual(rows[0]["joint_valid_tokens_per_second"], 5)
            self.assertEqual(rows[0]["cuda_peak_allocated_bytes"], 20)
            self.assertEqual(rows[1]["steps_in_window"], 1)
            self.assertEqual(rows[1]["step_start"], 3)

    def setup_run(self, root):
        (root / "monitor.json").write_text(
            json.dumps({"run_id": "r", "origin": "scratch", "arms": {"ce": "dflash"}})
        )
        append(
            root / "learning-curves.jsonl",
            dict(
                kind="checkpoint", run_id="r", step=2, arm="ce", checkpoint_id="r:2:ce"
            ),
        )
        provenance = dict(
            kind="provenance",
            manifest={
                "joint_monitor": {
                    "run_id": "r",
                    "step": 2,
                    "checkpoints": {"ce": "r:2:ce"},
                },
                "arms": [
                    {"name": "base", "mode": "baseline"},
                    {"name": "ce", "mode": "dflash"},
                ],
            },
            selected_ids=["a", "b"],
            trials=1,
            max_new_tokens=2,
            data_sha256="data",
            runtime_commit="commit",
            runtime_diff_sha256="patch",
        )
        rows = [provenance]
        for case, domain, accepted, drafted in [
            ("a", "code", 1, 1),
            ("b", "science", 0, 3),
        ]:
            for arm, speed in [("base", 10), ("ce", 20)]:
                rows.append(
                    dict(
                        kind="measurement",
                        id=case,
                        domain=domain,
                        trial=0,
                        arm=arm,
                        parity=True,
                        tok_s=speed,
                        decode_tok_s=speed + 1,
                        result={"token_ids": [1, 2]},
                        rounds=(
                            [{"accepted": accepted, "drafted": drafted}]
                            if arm == "ce"
                            else []
                        ),
                    )
                )
        rows.append(dict(kind="summary", failed_arms=[]))
        return rows

    def test_cli_import_weighted_acceptance_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.setup_run(root)
            bound = bind_manifest(root, 2, rows[0]["manifest"])
            self.assertEqual(bound["joint_monitor"]["checkpoints"]["ce"], "r:2:ce")
            with self.assertRaises(ValueError):
                bind_manifest(root, 3, rows[0]["manifest"])
            path = root / "bench.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            command = [
                sys.executable,
                "-m",
                "draftfit.training.joint_monitor",
                "--run-dir",
                str(root),
            ]
            subprocess.run(
                command + ["--import-mlx", str(path)], check=True, capture_output=True
            )
            output = [
                json.loads(x)
                for x in (root / "generation-curves.jsonl").read_text().splitlines()
            ]
            overall = next(r for r in output if r["domain"] is None)
            self.assertEqual(overall["acceptance_rate"], 0.25)
            self.assertEqual(overall["speedup_vs_target"], 2)
            self.assertEqual(len(output), 3)
            subprocess.run(
                command + ["--csv", str(root / "curves.csv")],
                check=True,
                capture_output=True,
            )
            with (root / "curves.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            with self.assertRaisesRegex(ValueError, "already imported"):
                ingest_mlx(root, path)

    def test_reject_bad_results_without_partial_write(self):
        for case in (
            "wrong_run",
            "wrong_checkpoint",
            "partial",
            "parity",
            "tokens",
            "nan",
            "counter",
            "duplicate",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = self.setup_run(root)
                if case == "wrong_run":
                    rows[0]["manifest"]["joint_monitor"]["run_id"] = "wrong"
                if case == "wrong_checkpoint":
                    rows[0]["manifest"]["joint_monitor"]["step"] = 9
                if case == "partial":
                    rows.pop()
                if case == "parity":
                    rows[2]["parity"] = False
                if case == "tokens":
                    rows[2]["result"]["token_ids"] = [9]
                if case == "nan":
                    rows[2]["tok_s"] = float("nan")
                if case == "counter":
                    rows[2]["rounds"][0]["accepted"] = 4
                if case == "duplicate":
                    rows.insert(2, rows[1])
                path = root / "bench.jsonl"
                path.write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaises(ValueError):
                    ingest_mlx(root, path)
                self.assertFalse((root / "generation-curves.jsonl").exists())
