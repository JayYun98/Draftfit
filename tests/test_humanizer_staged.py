"""Staged campaign data identity, patience and tiny real joint training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

from scripts.humanizer_staged_policy import decide, publish_slice, token_content_hash
from scripts.humanizer_staged import _prune_checkpoints, _prune_exports, _generate, _continuation_contract
from tests.test_humanizer_campaign_runner import HumanizerCampaignRunner


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_humanizer_campaign.py"


class StagedPolicyTests(unittest.TestCase):
    def test_new_journal_rows_reject_old_tokens_even_with_new_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}
            journal = root / "train.partial.jsonl"
            journal.write_text(json.dumps({"id": "new-id", "status": "accepted", "tokens": token}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlaps prior"):
                publish_slice(journal, root / "stage", 0, 1, set(), {token_content_hash(token)})
            self.assertFalse((root / "stage/train.tokens.jsonl").exists())

    def test_generation_uses_original_directory_and_unchanged_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(root=root, train_input=root / "train.inputs.jsonl", train_reserve=[],
                             generation_batch_size=64, hourly_usd=1., generation_budget_usd=1.,
                             deadline_unix=12345.)
            journal = root / "original/train.partial.jsonl"
            for target in (6852, 30000):
                with patch("scripts.humanizer_staged._accepted_count", side_effect=[5852, target]), \
                     patch("scripts.humanizer_staged.campaign.check_limit", return_value=1.), \
                     patch("scripts.humanizer_staged.subprocess.run", return_value=Namespace(returncode=0)) as run:
                    _generate(args, target, 0., journal=journal)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--output-dir") + 1], str(journal.parent))
                self.assertEqual(command[command.index("--required-count") + 1], "30000")
                self.assertEqual("--stop-after-accepted" in command, target < 30000)
                self.assertLessEqual(float(command[command.index("--deadline-unix") + 1]), 12345.)

    def test_immutable_disjoint_slice_and_crash_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "train.partial.jsonl"
            rows = [{"id": f"r{i}", "status": "accepted", "tokens": {"input_ids": [i, i+1],
                     "loss_mask": [1, 1]}} for i in range(5)]
            rows.insert(2, {"id": "bad", "status": "rejected"})
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            first = publish_slice(journal, root / "first", 0, 2, set())
            self.assertEqual(first, publish_slice(journal, root / "first", 0, 2, set()))
            second = publish_slice(journal, root / "second", 2, 2, set(first["ids"]))
            self.assertEqual(second["ids"], ["r2", "r3"])
            with self.assertRaises(ValueError):
                publish_slice(journal, root / "conflict", 2, 2, {"r3"})
            journal.write_text(journal.read_text().replace('"r2"', '"changed"'))
            with self.assertRaises(ValueError):
                publish_slice(journal, root / "second", 2, 2, set())

    def test_patience_requires_all_three_arms_stale(self):
        def values(a, b, c):
            return dict(zip(("dflash", "dspark", "dflash2"), (a, b, c)))
        history = []
        for step, losses in ((250, values(3, 3, 3)), (500, values(4, 4, 4)),
                             (750, values(4, 2, 4)), (1000, values(4, 3, 4)),
                             (1250, values(4, 3, 4))):
            result = decide(history, losses, step=step, patience=2)
            history.append(result)
        self.assertFalse(history[2]["stop"])
        self.assertFalse(history[3]["stop"])
        self.assertTrue(history[4]["stop"])
        with self.assertRaises(ValueError):
            decide(history, values(float("nan"), 2, 2), step=1500)

    def test_min_delta_accumulates_since_meaningful_improvement(self):
        # Actual Humanizer policy counterexample from the production review:
        # moving the threshold with each raw minimum falsely stops at step 1000.
        history = []
        for step, loss in zip((250, 500, 750, 1000), (1., .9994, .9988, .9982)):
            history.append(decide(history, dict.fromkeys(("dflash", "dspark", "dflash2"), loss),
                                  step=step, patience=3, min_delta=.001))
        self.assertFalse(history[-1]["stop"])
        self.assertTrue(all(history[2]["improved"].values()))
        self.assertEqual(set(history[-1]["stale"].values()), {1})
        self.assertEqual(set(history[-1]["best"].values()), {.9982})

    def test_min_delta_replays_old_history_without_new_state_fields(self):
        names = ("dflash", "dspark", "dflash2")
        history = [{"step": step, "losses": dict.fromkeys(names, loss),
                    "improved": dict.fromkeys(names, index == 0)}
                   for index, (step, loss) in enumerate(((250, 1.), (500, .9994)))]
        result = decide(history, dict.fromkeys(names, .9988), step=750,
                        patience=3, min_delta=.001)
        self.assertTrue(all(result["improved"].values()))
        self.assertEqual(set(result["stale"].values()), {0})

    def test_checkpoint_prune_keeps_latest_exact_resume_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for phase in range(2):
                attempt = root / f"run/stage-{phase:03d}/attempt-000"
                attempt.mkdir(parents=True)
                checkpoint = attempt / f"joint-step{phase + 1}.pt"
                checkpoint.write_bytes(f"phase{phase}".encode())
                (attempt / "worker.log").write_text("evidence")
                arm = attempt / "dflash" / f"dflash-step{phase + 1}"
                arm.mkdir(parents=True)
                (arm / "training_state.pt").write_bytes(b"redundant")
                paths.append(checkpoint)
            import hashlib
            _prune_checkpoints(root, [{"checkpoint": str(paths[-1]),
                                       "checkpoint_sha256": hashlib.sha256(paths[-1].read_bytes()).hexdigest()}])
            self.assertFalse(paths[0].exists())
            self.assertTrue(paths[1].exists())
            self.assertTrue((paths[0].parent / "worker.log").exists())
            self.assertFalse((paths[0].parent / "dflash/dflash-step1/training_state.pt").exists())

    def test_export_index_checks_bytes_before_deleting_old_exports(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports = {}
            for phase in (0, 1):
                exports[phase] = {}
                for name in ("dflash", "dspark", "dflash2"):
                    folder = root / f"exports/stage-{phase:03d}" / name
                    folder.mkdir(parents=True)
                    weight = folder / "model.safetensors"
                    weight.write_bytes(f"{phase}-{name}".encode())
                    exports[phase][name] = {"name": name, "export": str(folder),
                        "weights_sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}
            history = [{"phase": 0, "losses": dict.fromkeys(exports[0], 2.0), "exports": exports[0]},
                       {"phase": 1, "losses": dict.fromkeys(exports[1], 1.0), "exports": exports[1]}]
            (Path(exports[1]["dflash"]["export"]) / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                _prune_exports(root, history)
            self.assertTrue(Path(exports[0]["dflash"]["export"]).exists())
            self.assertFalse((root / "exports/latest.json").exists())


class StagedTinyTests(unittest.TestCase):

    def test_exhausted_pool_extends_with_new_phase_and_keeps_provenance(self):
        from scripts.generate_humanizer_campaign_pairs import read_sources
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            HumanizerCampaignRunner().make_fixture(root)
            tokens = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            for i, row in enumerate(tokens):
                row["input_ids"][0] = 35 + i
            pool = root / "pool.tokens.jsonl"
            pool.write_text("".join(json.dumps(row) + "\n" for row in tokens[:5]))
            validation = [json.loads(line) for line in (root / "eval.jsonl").read_text().splitlines()]
            accepted = tokens[:5] + validation + tokens[5:10]
            original = root / "original"
            original.mkdir()
            inputs = original / "train.inputs.jsonl"
            inputs.write_text("".join(json.dumps({"id": f"id{i}", "draft": f"draft {i}"}) + "\n"
                                      for i in range(len(accepted))))
            journal = original / "train.partial.jsonl"
            journal.write_text("".join(json.dumps({"id": f"id{i}", "status": "accepted",
                                                   "tokens": row, "pair": {"draft": f"draft {i}"}}) + "\n"
                                       for i, row in enumerate(accepted)))
            _, digest = read_sources([inputs])
            (original / "train.progress.json").write_text(json.dumps({"input_sha256": digest,
                                                                       "required_count": 30000}))
            env = os.environ.copy()
            env.update(SPECFORGE_DEVICE="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(SCRIPT.parents[1]))
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            command = [sys.executable, str(SCRIPT), "--root", str(root / "campaign"), "--staged",
                       "--tiny", "--manifest", str(root / "family.json"), "--train-tokens", str(pool),
                       "--validation-tokens", str(root / "eval.jsonl"), "--stage-rows", "5",
                       "--stage-first-steps", "5", "--budget-usd", "100"]
            def invoke(extra=()):
                return subprocess.run(command + list(extra), cwd=root, env=env,
                                      capture_output=True, text=True, timeout=240)
            first = invoke()
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            state_path = root / "campaign/staged-status.json"
            first_state = json.loads(state_path.read_text())
            self.assertEqual((first_state["status"], first_state["completed_step"]), ("data_exhausted", 5))
            extension = ["--continuation-journal", str(journal), "--continuation-start-row", "7",
                         "--train-input", str(inputs)]
            original_pool = pool.read_text()
            pool.write_text(original_pool.replace('35', '55', 1))
            bad = invoke(extension)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("original train pool changed", bad.stderr)
            pool.write_text(original_pool)
            # An already early-stopped campaign must never enter generation or training.
            state_path.write_text(json.dumps({**first_state, "status": "early_stopped"}))
            stopped = invoke(extension)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertEqual(json.loads(state_path.read_text())["completed_step"], 5)
            state_path.write_text(json.dumps(first_state))
            extended = invoke(extension)
            self.assertEqual(extended.returncode, 0, extended.stdout + extended.stderr)
            state = json.loads(state_path.read_text())
            self.assertEqual((state["status"], state["completed_step"]), ("complete", 10))
            self.assertEqual([row["phase"] for row in state["history"]], [0, 1])
            self.assertEqual([row["start_row"] for row in state["history"]], [0, 7])
            self.assertEqual(state["history"][-1]["source_kind"], "continuation")
            self.assertEqual(state["history"][0], first_state["history"][0])
            self.assertEqual(pool.read_text(), original_pool)

    def test_staged_tiny_real_resume_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            HumanizerCampaignRunner().make_fixture(root)
            env = os.environ.copy()
            env.update(SPECFORGE_DEVICE="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(SCRIPT.parents[1]))
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root / "campaign"),
                                     "--tiny", "--manifest", str(root / "family.json"), "--staged",
                                     "--train-tokens", str(root / "train.jsonl"),
                                     "--validation-tokens", str(root / "eval.jsonl"),
                                     "--stage-rows", "5", "--stage-first-steps", "5",
                                     "--budget-usd", "100"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            status = json.loads((root / "campaign/staged-status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual([r["step"] for r in status["history"]], [5, 10])
            self.assertEqual([r["start_row"] for r in status["history"]], [0, 5])
            self.assertTrue((root / "campaign/exports/latest.json").exists())


if __name__ == "__main__":
    unittest.main()
