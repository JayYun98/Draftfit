"""Compact joint storage and optional arm files, using real CPU train/resume."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from draftfit.training import joint_family
from tests import test_joint_family_core as core_tests


class JointCheckpointRetentionTest(unittest.TestCase):
    def test_cpu_transfer_once_per_identity_and_serialized_aliases(self):
        shared = nn.Parameter(torch.arange(32768, dtype=torch.float32))
        stages, bundles = [], []
        for name in ("a", "b", "c"):
            model = nn.Module()
            model.register_parameter("shared", shared)
            model.register_parameter("private", nn.Parameter(torch.ones(8)))
            # Equal mutable buffers must stay independent, as must private heads.
            model.register_buffer("rotary", torch.ones(8))
            stages.append(SimpleNamespace(name=name))
            bundles.append(SimpleNamespace(draft_model=model))
        transfers = []

        def copy_to_cpu(tensor, *args, **kwargs):
            transfers.append(tensor.numel())
            # CPU .cpu() is normally a no-op. Force a fresh storage to exercise
            # the copy semantics of CUDA -> CPU without claiming a GPU test.
            return tensor.clone()

        with mock.patch.object(torch.Tensor, "cpu", copy_to_cpu):
            states = joint_family._joint_draft_state_dicts(stages, bundles)
        self.assertEqual(transfers.count(32768), 1)
        self.assertEqual(len(transfers), 7)
        serialized = io.BytesIO()
        torch.save(states, serialized)
        serialized.seek(0)
        loaded = torch.load(serialized, weights_only=True)
        for name in ("b", "c"):
            self.assertEqual(loaded["a"]["shared"].data_ptr(),
                             loaded[name]["shared"].data_ptr())
            for key in ("private", "rotary"):
                self.assertNotEqual(loaded["a"][key].data_ptr(),
                                    loaded[name][key].data_ptr())
        loaded["a"]["rotary"].zero_()
        self.assertTrue(loaded["b"]["rotary"].eq(1).all())
        duplicate = io.BytesIO()
        torch.save({arm: {k: v.clone() for k, v in state.items()}
                    for arm, state in states.items()}, duplicate)
        self.assertGreater(len(duplicate.getvalue()) - len(serialized.getvalue()),
                           2 * shared.numel() * shared.element_size() - 4096)

    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def compact(stages):
        for stage in stages:
            # Exercise the trainer boundary independently of the schema change,
            # which is owned by the other agent.
            stage.config.training = stage.config.training.model_copy(
                update={"joint_emit_arm_checkpoints": False})

    def test_real_compact_resume_matches_default_and_preserves_source(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="joint-compact-") as directory:
                root = Path(directory)
                reference, stages = core_tests.JointFamilyCoreTest().stages(root)
                for stage in stages:
                    stage.config.training.max_checkpoints = 1
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    joint_family.run_joint_family(reference, stages)
                    for stage in stages:
                        self.assertTrue((reference / stage.name /
                                         f"{stage.name}-latest/training_state.pt").is_file())
                    expected = torch.load(reference / "joint-step2.pt", weights_only=True)
                    self.compact(stages)
                    # Checkpoint serialization options must not alter identity.
                    self.assertEqual(expected["fingerprint"],
                                     joint_family._semantic_fingerprint(stages))
                    partial = root / "partial"
                    for stage in stages:
                        stage.config.output_dir = str(partial / stage.name)
                    with mock.patch("draftfit.training.checkpoint.CheckpointManager.save",
                                    side_effect=AssertionError("arm save must be disabled")):
                        result = joint_family.run_joint_family(partial, stages, stop_at_step=1)
                    self.assertEqual({r["global_step"] for r in result["runs"]}, {1})
                    source = partial / "joint-step1.pt"
                    self.assertTrue(source.is_file())
                    self.assertFalse(list(partial.rglob("training_state.pt")))
                    self.assertFalse(list(partial.rglob("*-latest")))
                    for stage in stages:
                        self.assertTrue((partial / stage.name / "config.json").is_file())
                    resumed = root / "resumed"
                    for stage in stages:
                        stage.config.output_dir = str(resumed / stage.name)
                    with mock.patch("draftfit.training.checkpoint.CheckpointManager.save",
                                    side_effect=AssertionError("arm save must be disabled")):
                        joint_family.run_joint_family(resumed, stages, resume_from=source)
                    self.assertTrue(source.is_file())
                    self.assertEqual([p.name for p in resumed.glob("joint-step*.pt")],
                                     ["joint-step2.pt"])
                    actual = torch.load(resumed / "joint-step2.pt", weights_only=True)
                    for key in ("drafts", "optimizer", "step", "epoch", "next_batch", "horizon"):
                        self.assert_state_equal(actual[key], expected[key])
                    key = "layers.0.self_attn.q_proj.weight"
                    self.assertEqual(len({s[key].data_ptr() for s in actual["drafts"].values()}), 1)
                    self.assertFalse(list(resumed.rglob("training_state.pt")))
                    results = json.loads((resumed / "results.json").read_text())
                    for result in results:
                        self.assertEqual(result["joint_checkpoint"], str(resumed / "joint-step2.pt"))
                        self.assertTrue(Path(result["joint_checkpoint"]).is_file())
                        self.assertFalse(result["emit_arm_checkpoints"])
                    rows = [json.loads(line) for line in
                            (resumed / "learning-curves.jsonl").read_text().splitlines()]
                    checkpoints = [row for row in rows if row["kind"] == "checkpoint"]
                    self.assertEqual(len(checkpoints), 3)
                    for row in checkpoints:
                        self.assertEqual(row["checkpoint_path"], str(resumed / "joint-step2.pt"))
                        self.assertTrue(Path(row["checkpoint_path"]).is_file())
                        self.assertFalse(row["emit_arm_checkpoints"])
                        self.assertIsNone(row["arm_checkpoint"])
        finally:
            torch.set_num_threads(threads)

    def test_mixed_arm_emission_rejected(self):
        with tempfile.TemporaryDirectory(prefix="joint-compact-config-") as directory:
            _, stages = core_tests.JointFamilyCoreTest().stages(Path(directory))
            self.compact(stages[:1])
            with self.assertRaisesRegex(ValueError, "joint_emit_arm_checkpoints"):
                joint_family.validate_joint_family(stages)


if __name__ == "__main__":
    unittest.main()
