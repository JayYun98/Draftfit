"""Real joint updates around the opt-in external verifier memory boundary."""

import os
import unittest

import torch

from draftfit.family import build_family_plan
from draftfit.training.joint_family import run_joint_family
from tests import test_unattended_controller as fixtures


class JointServingHookTest(unittest.TestCase):
    execution_device = "cpu"

    def test_milestone_handoff_preserves_next_update_and_committed_cursor(self):
        f = fixtures.UnattendedControllerTest()
        self.addCleanup(f.doCleanups)
        f.setUp()
        os.environ["SPECFORGE_DEVICE"] = self.execution_device
        _, stages = build_family_plan(str(f.root / "family.json"))
        for stage in stages:
            path = f.root / (stage.name + "-draft.json")
            stage.draft_config.to_json_file(str(path))
            stage.config.model.draft_model_config = str(path)
        plain = f.root / "plain"
        observed = f.root / "observed"

        def noop(**kw):
            return dict(stop=kw["step"] >= 2)

        run_joint_family(plain, stages, milestone_observer=noop, stop_at_step=2)
        seen = []

        def observe(**kw):
            saved = torch.load(kw["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["step"], kw["step"])
            with kw["memory_handoff"]() as receipt:
                self.assertFalse(receipt["target_resident"])
                self.assertEqual(
                    receipt["cuda_release_observed"], self.execution_device == "cuda"
                )
                torch.rand(7)
                if self.execution_device == "cuda":
                    torch.rand(7, device="cuda")
            seen.append(kw["step"])
            return dict(stop=kw["step"] >= 2)

        run_joint_family(
            observed,
            stages,
            milestone_observer=observe,
            milestone_runtime=True,
            stop_at_step=2,
        )
        self.assertEqual(seen, [1, 2])
        left = torch.load(
            plain / "joint-step2.pt", map_location="cpu", weights_only=True
        )
        right = torch.load(
            observed / "joint-step2.pt", map_location="cpu", weights_only=True
        )
        for name, weights in left["drafts"].items():
            for key, value in weights.items():
                self.assertTrue(
                    torch.equal(value, right["drafts"][name][key]), (name, key)
                )
        for key in ("step", "epoch", "next_batch", "consumed_samples"):
            self.assertEqual(left[key], right[key])


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("DRAFTFIT_RUN_QUEUE_CUDA") == "1",
    "explicit CUDA opt-in required",
)
class JointServingHookCUDA(JointServingHookTest):
    execution_device = "cuda"
