"""Public configuration can retain joint resume state without duplicate arm saves."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from draftfit.config import Config
from draftfit.training.joint_family import _semantic_fingerprint, run_joint_family
from tests import test_joint_family_core as core


class JointCheckpointSerializationTest(unittest.TestCase):
    def test_default_and_compact_fingerprints_match_legacy_omitted_field(self):
        with tempfile.TemporaryDirectory(
            prefix="joint-legacy-fingerprint-"
        ) as directory:
            _, stages = core.JointFamilyCoreTest().stages(Path(directory))
            self.assertTrue(
                all(
                    stage.config.training.joint_emit_arm_checkpoints for stage in stages
                )
            )
            original_dump = Config.model_dump

            def legacy_dump(config, *args, **kwargs):
                payload = original_dump(config, *args, **kwargs)
                payload["training"].pop("joint_emit_arm_checkpoints", None)
                return payload

            for scheduler in ("cosine", "wsd"):
                for stage in stages:
                    stage.config.training.lr_scheduler = scheduler
                for fixed_only in (False, True):
                    with self.subTest(scheduler=scheduler, fixed_only=fixed_only):
                        with mock.patch.object(Config, "model_dump", legacy_dump):
                            legacy = _semantic_fingerprint(
                                stages, fixed_only=fixed_only
                            )
                        for emit in (True, False):
                            for stage in stages:
                                stage.config.training.joint_emit_arm_checkpoints = emit
                            self.assertEqual(
                                _semantic_fingerprint(stages, fixed_only=fixed_only),
                                legacy,
                            )

    def test_joint_only_serialization_preserves_resumable_state(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(
                prefix="joint-serialization-"
            ) as directory:
                output, stages = core.JointFamilyCoreTest().stages(Path(directory))
                for stage in stages:
                    raw = stage.config.model_dump()
                    raw["training"]["joint_emit_arm_checkpoints"] = False
                    configured = Config.model_validate(raw)
                    self.assertFalse(configured.training.joint_emit_arm_checkpoints)
                    stage.config.training.joint_emit_arm_checkpoints = False
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    result = run_joint_family(output, stages)
                self.assertTrue((output / "joint-step2.pt").is_file())
                state = torch.load(output / "joint-step2.pt", weights_only=True)
                self.assertEqual(state["step"], 2)
                self.assertIn("optimizer", state)
                self.assertIn("rng", state)
                self.assertEqual(set(state["drafts"]), {s.name for s in stages})
                for row, stage in zip(result["runs"], stages):
                    self.assertFalse(row["emit_arm_checkpoints"])
                    self.assertNotIn("arm_checkpoint", row)
                    self.assertFalse(
                        list(Path(stage.config.output_dir).glob("*-step*"))
                    )
                resumed = []
                for stage in stages:
                    cfg = stage.config.model_copy(deep=True)
                    cfg.output_dir = str(Path(directory) / "resumed" / stage.name)
                    cfg.training.joint_emit_arm_checkpoints = True
                    resumed.append(replace(stage, config=cfg))
                with mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                    continued = run_joint_family(
                        Path(directory) / "resumed",
                        resumed,
                        resume_from=output / "joint-step1.pt",
                    )
                for row, stage in zip(continued["runs"], resumed):
                    self.assertTrue(row["emit_arm_checkpoints"])
                    self.assertEqual(row["global_step"], 2)
                    self.assertTrue(Path(row["arm_checkpoint"]).is_dir())
                restored = torch.load(
                    Path(directory) / "resumed" / "joint-step2.pt", weights_only=True
                )
                for arm in state["drafts"]:
                    for name, value in state["drafts"][arm].items():
                        torch.testing.assert_close(
                            value, restored["drafts"][arm][name], rtol=0, atol=0
                        )
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
