"""Real CPU three-arm checks for immutable data phase handoff."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.training.joint_family import run_joint_family
from tests.test_joint_deadline import stages_for
from tests.test_joint_online_cpu import online_fixture


class JointDataPhaseTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.device = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.device.start()

    def tearDown(self):
        self.device.stop()
        torch.set_num_threads(self.threads)

    def prepare(self, root):
        online_fixture(root, False)
        first = root / "chunk0.jsonl"
        second = root / "chunk1.jsonl"
        first.write_bytes((root / "train.jsonl").read_bytes())
        rows = [json.loads(line) for line in first.read_text().splitlines()]
        for index, row in enumerate(rows):
            row["input_ids"][0] = 40 + index
        second.write_text("".join(json.dumps(row) + "\n" for row in rows))
        base = json.loads((root / "base.json").read_text())
        base["data"]["prompts_path"] = str(first)
        base["training"]["max_steps"] = None
        base["training"]["total_steps"] = 6
        (root / "base.json").write_text(json.dumps(base))
        return first, second

    def test_new_shard_keeps_optimizer_and_exact_resume_within_phase(self):
        with tempfile.TemporaryDirectory(prefix="joint-phases-") as directory:
            root = Path(directory)
            _, second = self.prepare(root)
            first_root = root / "phase0"
            run_joint_family(first_root, stages_for(root, first_root),
                             data_phase=0, stop_at_step=2)
            first = torch.load(first_root / "joint-step2.pt", weights_only=True)
            self.assertEqual((first["step"], first["horizon"], first["data_phase"]),
                             (2, 6, 0))
            base = json.loads((root / "base.json").read_text())
            base["data"]["prompts_path"] = str(second)
            base["training"]["num_epochs"] = 2  # per-stage budget may grow
            (root / "base.json").write_text(json.dumps(base))

            full_root = root / "phase1-full"
            run_joint_family(full_root, stages_for(root, full_root),
                             resume_from=first_root / "joint-step2.pt",
                             data_phase=1, stop_at_step=4)
            self.assertFalse((full_root / "joint-step2.pt").exists())
            self.assertFalse((full_root / "ce" / "ce-step2").exists())
            full = torch.load(full_root / "joint-step4.pt", weights_only=True)
            self.assertEqual((full["step"], full["epoch"], full["next_batch"],
                              full["phase_start_step"], full["data_phase"],
                              full["horizon"]), (4, 1, 0, 2, 1, 6))
            self.assertEqual(
                full["optimizer"]["scheduler_state_dict"]["after_scheduler_dict"]["last_epoch"],
                first["optimizer"]["scheduler_state_dict"]["after_scheduler_dict"]["last_epoch"] + 2)
            self.assertTrue(any(not torch.equal(t, full["optimizer"]["fp32_params"][i])
                                for i, t in enumerate(first["optimizer"]["fp32_params"])))

            partial_root = root / "phase1-partial"
            run_joint_family(partial_root, stages_for(root, partial_root),
                             resume_from=first_root / "joint-step2.pt",
                             data_phase=1, stop_at_step=3)
            self.assertFalse((partial_root / "joint-step2.pt").exists())
            resumed_root = root / "phase1-resumed"
            run_joint_family(resumed_root, stages_for(root, resumed_root),
                             resume_from=partial_root / "joint-step3.pt",
                             data_phase=1, stop_at_step=4)
            resumed = torch.load(resumed_root / "joint-step4.pt", weights_only=True)
            self.assertFalse((resumed_root / "joint-step3.pt").exists())
            for arm in full["drafts"]:
                for key, value in full["drafts"][arm].items():
                    self.assertTrue(torch.equal(value, resumed["drafts"][arm][key]),
                                    (arm, key))
            for a, b in zip(full["optimizer"]["fp32_params"],
                            resumed["optimizer"]["fp32_params"]):
                self.assertTrue(torch.equal(a, b))

    def test_new_phase_requires_explicit_increment_and_fixed_eval(self):
        with tempfile.TemporaryDirectory(prefix="joint-phase-guard-") as directory:
            root = Path(directory)
            _, second = self.prepare(root)
            first_root = root / "phase0"
            run_joint_family(first_root, stages_for(root, first_root), stop_at_step=1)
            source = first_root / "joint-step1.pt"
            base = json.loads((root / "base.json").read_text())
            base["data"]["prompts_path"] = str(second)
            (root / "base.json").write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_joint_family(root / "implicit", stages_for(root, root / "implicit"),
                                 resume_from=source)
            with self.assertRaisesRegex(ValueError, "data_phase"):
                run_joint_family(root / "skipped", stages_for(root, root / "skipped"),
                                 resume_from=source, data_phase=2)
            changed_eval = root / "changed-eval.jsonl"
            rows = [json.loads(line) for line in (root / "eval.jsonl").read_text().splitlines()]
            for index, row in enumerate(rows):
                row["input_ids"][0] = 55 + index
            changed_eval.write_text("".join(json.dumps(row) + "\n" for row in rows))
            base["data"]["eval_data_path"] = str(changed_eval)
            (root / "base.json").write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "fixed eval"):
                run_joint_family(root / "bad-eval", stages_for(root, root / "bad-eval"),
                                 resume_from=source, data_phase=1)

    def test_expired_resumed_phase_still_saves_local_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="joint-phase-expired-") as directory:
            root = Path(directory)
            _, second = self.prepare(root)
            first_root = root / "phase0"
            run_joint_family(first_root, stages_for(root, first_root), stop_at_step=1)
            base = json.loads((root / "base.json").read_text())
            base["data"]["prompts_path"] = str(second)
            (root / "base.json").write_text(json.dumps(base))
            expired_root = root / "expired"
            with mock.patch("draftfit.training.joint_family.time.time", return_value=100.0):
                result = run_joint_family(expired_root, stages_for(root, expired_root),
                                          resume_from=first_root / "joint-step1.pt",
                                          data_phase=1, stop_at_step=2,
                                          deadline_unix=100.0)
            self.assertEqual((result["status"], result["reason"]), ("stopped", "deadline"))
            checkpoint = torch.load(expired_root / "joint-step1.pt", weights_only=True)
            self.assertEqual((checkpoint["data_phase"], checkpoint["phase_start_step"],
                              checkpoint["epoch"], checkpoint["next_batch"]),
                             (1, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
