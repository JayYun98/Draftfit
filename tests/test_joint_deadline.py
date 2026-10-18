"""Deadline boundaries on the real three-arm CPU online training path."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.config import Config
from draftfit.family import FamilyStage
from draftfit.optimizer import BF16Optimizer
from draftfit.training import joint_family
from draftfit.training.joint_family import run_joint_family
from draftfit.training.model_loading import (
    convert_family_draft_config,
    load_draft_config_source,
)
from tests.test_joint_online_cpu import online_fixture


def stages_for(root, output):
    base = json.loads((root / "base.json").read_text())
    source = load_draft_config_source(str(root / "draft.json"))
    stages = []
    for name, strategy in (
        ("dspark", "dspark"),
        ("ce", "dflash"),
        ("selector", "dflash2"),
    ):
        draft = convert_family_draft_config(source, strategy)
        raw = json.loads(json.dumps(base))
        raw["output_dir"] = str(output / name)
        raw["run_id"] = name
        raw["training"]["strategy"] = strategy
        raw["training"]["eval_interval"] = 50
        raw["training"]["save_interval"] = 0
        raw["training"]["log_interval"] = 2
        raw["model"]["draft_model_config"] = str(output / name / "draft.json")
        (output / name).mkdir(parents=True, exist_ok=True)
        draft.to_json_file(raw["model"]["draft_model_config"])
        stages.append(FamilyStage(name, Config.model_validate(raw), draft, None))
    return stages


class JointDeadlineTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.device = mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.device.start()

    def tearDown(self):
        self.device.stop()
        torch.set_num_threads(self.threads)

    def test_optimizer_boundary_and_exact_new_directory_resume(self):
        with tempfile.TemporaryDirectory(prefix="joint-deadline-") as directory:
            root = Path(directory)
            online_fixture(root, False)
            full_root = root / "full"
            full = run_joint_family(full_root, stages_for(root, full_root))
            self.assertEqual([r["status"] for r in full["runs"]], ["complete"] * 3)

            stopped_root = root / "stopped"
            clock = [100.0]
            original_step = BF16Optimizer.step

            def finish_first_step(optimizer):
                result = original_step(optimizer)
                clock[0] = 200.0
                return result

            with mock.patch(
                "draftfit.training.joint_family.time.time", side_effect=lambda: clock[0]
            ):
                with mock.patch.object(BF16Optimizer, "step", finish_first_step):
                    stopped = run_joint_family(
                        stopped_root,
                        stages_for(root, stopped_root),
                        deadline_unix=150.0,
                    )
            self.assertEqual(
                (stopped["status"], stopped["reason"]), ("stopped", "deadline")
            )
            self.assertEqual(
                {(r["status"], r["reason"], r["global_step"]) for r in stopped["runs"]},
                {("stopped", "deadline", 1)},
            )
            self.assertEqual(
                json.loads((stopped_root / "results.json").read_text()), stopped["runs"]
            )
            state = torch.load(stopped_root / "joint-step1.pt", weights_only=True)
            self.assertEqual(
                (state["step"], state["epoch"], state["next_batch"], state["horizon"]),
                (1, 0, 2, 2),
            )
            self.assertFalse((stopped_root / "joint-step2.pt").exists())
            rows = [
                json.loads(line)
                for line in (stopped_root / "learning-curves.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                {r["arm"] for r in rows if r["kind"] == "train"},
                {"dspark", "ce", "selector"},
            )
            self.assertEqual(
                {r["capture_calls"] for r in rows if r["kind"] == "capture"}, {4}
            )  # two held-out and two training captures
            self.assertFalse(
                any(r["kind"] == "evaluation" and r["step"] == 1 for r in rows)
            )

            resumed_root = root / "resumed"
            resumed = run_joint_family(
                resumed_root,
                stages_for(root, resumed_root),
                resume_from=stopped_root / "joint-step1.pt",
            )
            self.assertEqual([r["status"] for r in resumed["runs"]], ["complete"] * 3)
            expected = torch.load(full_root / "joint-step2.pt", weights_only=True)
            actual = torch.load(resumed_root / "joint-step2.pt", weights_only=True)
            self.assertEqual(
                (actual["step"], actual["epoch"], actual["next_batch"]), (2, 1, 0)
            )
            for arm in expected["drafts"]:
                for name, value in expected["drafts"][arm].items():
                    self.assertTrue(
                        torch.equal(value, actual["drafts"][arm][name]), (arm, name)
                    )

    def test_expired_deadline_saves_seed_and_skips_capture(self):
        with tempfile.TemporaryDirectory(prefix="joint-expired-") as directory:
            root = Path(directory)
            online_fixture(root, False)
            output = root / "expired"
            with mock.patch(
                "draftfit.training.joint_family.time.time", return_value=100.0
            ):
                result = run_joint_family(
                    output, stages_for(root, output), deadline_unix=100.0
                )
            self.assertEqual(
                (result["status"], result["reason"]), ("stopped", "deadline")
            )
            state = torch.load(output / "joint-step0.pt", weights_only=True)
            self.assertEqual(
                (state["step"], state["epoch"], state["next_batch"], state["horizon"]),
                (0, 0, 0, 2),
            )
            self.assertFalse((output / "joint-step1.pt").exists())
            rows = [
                json.loads(line)
                for line in (output / "learning-curves.jsonl").read_text().splitlines()
            ]
            self.assertEqual({r["kind"] for r in rows}, {"checkpoint", "queue_closed"})
            self.assertTrue(
                all(
                    r["produced"] == 0 and r["consumed"] == 0
                    for r in rows
                    if r["kind"] == "queue_closed"
                )
            )

            resumed_root = root / "from-seed"
            resumed = run_joint_family(
                resumed_root,
                stages_for(root, resumed_root),
                resume_from=output / "joint-step0.pt",
            )
            self.assertEqual([r["status"] for r in resumed["runs"]], ["complete"] * 3)

    def test_deadline_after_checkpoint_prevents_next_capture(self):
        with tempfile.TemporaryDirectory(prefix="joint-between-") as directory:
            root = Path(directory)
            online_fixture(root, False)
            output = root / "between"
            stages = stages_for(root, output)
            for stage in stages:
                stage.config.training.save_interval = 1
            clock = [100.0]
            save = joint_family._atomic_torch_save

            def save_then_expire(payload, path):
                save(payload, path)
                if path.name == "joint-step1.pt":
                    clock[0] = 200.0

            with mock.patch(
                "draftfit.training.joint_family.time.time", side_effect=lambda: clock[0]
            ):
                with mock.patch.object(
                    joint_family, "_atomic_torch_save", save_then_expire
                ):
                    result = run_joint_family(output, stages, deadline_unix=150.0)
            self.assertEqual(
                (result["status"], result["reason"]), ("stopped", "deadline")
            )
            self.assertEqual({r["global_step"] for r in result["runs"]}, {1})
            state = torch.load(output / "joint-step1.pt", weights_only=True)
            self.assertEqual((state["epoch"], state["next_batch"]), (0, 2))
            rows = [
                json.loads(line)
                for line in (output / "learning-curves.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {r["capture_calls"] for r in rows if r["kind"] == "capture"}, {4}
            )
            self.assertEqual(
                {
                    (r["arm"], r["steps_in_window"])
                    for r in rows
                    if r["kind"] == "train"
                },
                {("dspark", 1), ("ce", 1), ("selector", 1)},
            )
            self.assertEqual(
                json.loads((output / "results.json").read_text()), result["runs"]
            )

    def test_invalid_deadlines_rejected_before_setup(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "100"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "deadline_unix"),
            ):
                run_joint_family("unused", [], deadline_unix=value)


if __name__ == "__main__":
    unittest.main()
