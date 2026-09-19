import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_ling_1k.sh"


class Ling1KRunnerScriptTest(unittest.TestCase):
    def test_dry_run_prints_pinned_capture_train_eval_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"id": "one"}) + "\n", encoding="utf-8")
            result = subprocess.run(
                [str(RUNNER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "RUN_ROOT": str(root / "run"),
                    "SOURCE_ROWS": str(source),
                    "TRAIN_SIZE": "1",
                    "HOLDOUT_SIZE": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--revision a2ee06c0f2de5b171701aee7f73f70a1da75483b", result.stdout
        )
        self.assertIn("--output-path", result.stdout)
        self.assertIn("features/train", result.stdout)
        self.assertIn("features/holdout", result.stdout)
        self.assertIn("data.eval_hidden_states_path", result.stdout)
        self.assertIn("specforge export --to hf", result.stdout)
        self.assertIn(
            "no download, split, capture, training, or export will run", result.stdout
        )

    def test_dry_run_regeneration_requires_no_gpu_or_server(self):
        result = subprocess.run(
            [str(RUNNER)],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "REGENERATE": "1",
                "RUN_ROOT": "/tmp/dspark-ling-regenerate-dry",
                "SOURCE_ROWS": "/tmp/source-not-needed-in-dry-run.jsonl",
                "TRAIN_SIZE": "1",
                "HOLDOUT_SIZE": "1",
                "REGEN_SERVER": "127.0.0.1:30000",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--regenerated-by ling-t0", result.stdout)
        self.assertIn("--reasoning disable", result.stdout)
        self.assertIn("127.0.0.1:30000", result.stdout)

    def test_dry_run_exposes_two_gpu_capture_and_training(self):
        result = subprocess.run(
            [str(RUNNER)],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "RUN_ROOT": "/tmp/dspark-ling-tp2-dry",
                "SOURCE_ROWS": "/tmp/source-not-needed-in-dry-run.jsonl",
                "GPU_IDS": "0,1",
                "CAPTURE_NPROC": "2",
                "CAPTURE_TP_SIZE": "2",
                "TRAIN_NPROC": "2",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "CUDA_VISIBLE_DEVICES=0\\,1 torchrun --nproc_per_node=2", result.stdout
        )
        self.assertIn("--tp-size 2", result.stdout)
        self.assertIn("deployment.trainer.nproc_per_node=2", result.stdout)
        self.assertIn("training.num_epochs=1", result.stdout)
        self.assertIn("training.max_steps=450", result.stdout)
        self.assertIn("training.total_steps=450", result.stdout)

    def test_dry_run_exposes_same_world_resume(self):
        result = subprocess.run(
            [str(RUNNER)],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "1",
                "RUN_ROOT": "/tmp/dspark-ling-resume-dry",
                "SOURCE_ROWS": "/tmp/source-not-needed-in-dry-run.jsonl",
                "GPU_IDS": "0,1",
                "TRAIN_NPROC": "2",
                "TRAIN_MAX_STEPS": "200",
                "TRAIN_TOTAL_STEPS": "450",
                "TRAIN_NUM_EPOCHS": "2",
                "RESUME_FROM": "/tmp/dspark-ling-resume-dry/checkpoints/ling-step100",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training.max_steps=200", result.stdout)
        self.assertIn("training.total_steps=450", result.stdout)
        self.assertIn("training.num_epochs=2", result.stdout)
        self.assertIn(
            "training.resume_from=/tmp/dspark-ling-resume-dry/checkpoints/ling-step100",
            result.stdout,
        )

    def test_dry_run_is_independent_of_calling_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(RUNNER)],
                cwd=directory,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "PYTHONPATH": "",
                    "RUN_ROOT": str(Path(directory) / "run"),
                    "SOURCE_ROWS": str(Path(directory) / "source-not-needed.jsonl"),
                    "TRAIN_SIZE": "1",
                    "HOLDOUT_SIZE": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / "scripts" / "prepare_hidden_states.py"), result.stdout)
        self.assertIn(
            str(ROOT / "examples" / "configs" / "ling-3.0-tiny-dspark-offline.yaml"),
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
