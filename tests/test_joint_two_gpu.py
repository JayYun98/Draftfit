"""Opt-in tiny real two-GPU joint training and exact generation gate.

Run: python -m unittest tests.test_joint_two_gpu
Requires two visible CUDA GPUs and the provisioned Transformers runtime.
"""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from draftfit.family import build_family_plan
from draftfit.training.joint_family import run_joint_family
from tests.test_joint_online_cpu import online_fixture
from tests import test_joint_verification as verification


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                     "requires two visible CUDA GPUs")
class JointTwoGPU(verification.JointVerificationTest):
    execution_device = "cuda"
    target_device = "cuda:1"

    def test_online_training_checkpoint_and_verification(self):
        with tempfile.TemporaryDirectory(prefix="joint-two-gpu-") as directory:
            root = Path(directory)
            online_fixture(root, seeded=False)
            path = root / "base.json"
            config = json.loads(path.read_text())
            config["training"]["joint_target_device"] = "cuda:1"
            config["training"]["joint_verification"] = dict(
                interval=1, max_prompts=1, max_new_tokens=4,
                draft_tokens=2, trials=1, offload_optimizer=True)
            path.write_text(json.dumps(config))
            output, stages = build_family_plan(str(root / "family.json"))
            result = run_joint_family(output, stages, stop_at_step=1)
            self.assertTrue((Path(output) / "joint-step1.pt").is_file())
            report = json.loads((Path(output) / "verification-step1.json").read_text())
            self.assertTrue(report["tokenparity"])
            self.assertEqual(report["bindings"]["runtime"]["device"], "cuda:1")
            self.assertEqual({r["status"] for r in result["runs"]}, {"complete"})


if __name__ == "__main__":
    unittest.main()
