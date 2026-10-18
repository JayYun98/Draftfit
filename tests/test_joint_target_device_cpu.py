"""CPU checks for the opt-in joint teacher device contract and routing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from draftfit.config import Config
from draftfit.family import build_family_plan
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.training.joint_family import _semantic_fingerprint, validate_joint_family
from draftfit.training.joint_online import JointOnlineSource
from draftfit.training.joint_verification import _clock
from tests.test_joint_online_cpu import online_fixture


class JointTargetDeviceCPU(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="joint-device-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        online_fixture(self.root, seeded=False)
        self.raw = json.loads((self.root / "base.json").read_text())

    def stages(self, device=None):
        raw = json.loads(json.dumps(self.raw))
        if device is not None:
            raw["training"]["joint_target_device"] = device
        (self.root / "base.json").write_text(json.dumps(raw))
        return build_family_plan(str(self.root / "family.json"))[1]

    def test_schema_and_family_reject_invalid_or_mixed_devices(self):
        for value in ("cuda", "cuda:-1", "cuda:01", "cpu", "cuda:1 "):
            raw = json.loads(json.dumps(self.raw))
            raw["training"]["joint_target_device"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config.model_validate(raw)
        raw = json.loads(json.dumps(self.raw))
        raw["training"]["joint_online_capture"] = False
        raw["training"]["joint_target_device"] = "cuda:1"
        with self.assertRaises(ValueError):
            Config.model_validate(raw)
        stages = self.stages("cuda:1")
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=2):
            validate_joint_family(stages)
            stages[1].config.training.joint_target_device = "cuda:0"
            with self.assertRaisesRegex(ValueError, "differing training.joint_target_device"):
                validate_joint_family(stages)
        stages = self.stages("cuda:1")
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "not visible"):
                validate_joint_family(stages)

    def test_default_fingerprint_and_explicit_device_identity(self):
        stages = self.stages()
        default = _semantic_fingerprint(stages, JointOnlineSource(stages[0], True))
        self.assertIsNone(stages[0].config.training.joint_target_device)
        stages = self.stages("cuda:1")
        selected = _semantic_fingerprint(stages, JointOnlineSource(stages[0], True))
        self.assertNotEqual(default, selected)

    def test_explicit_device_reaches_teacher_without_changing_draft_local_device(self):
        stages = self.stages("cuda:1")
        source = JointOnlineSource(stages[0], True)
        actual = OfflineTransformersCapture.from_pretrained
        selected = []

        def load(*args, device, **kwargs):
            selected.append(device)
            return actual(*args, device="cpu", **kwargs)

        with mock.patch.object(OfflineTransformersCapture, "from_pretrained", side_effect=load), \
             mock.patch("draftfit.training.joint_online.get_device", side_effect=AssertionError("local device used")), \
             mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
             mock.patch.object(torch.cuda, "set_rng_state_all"), \
             mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
             mock.patch.object(torch.cuda, "max_memory_reserved", return_value=0):
            loader, _ = source.loader(stages[0], 0, False, False)
            batch = next(iter(loader))
        self.assertEqual(selected, [torch.device("cuda:1")])
        self.assertEqual(batch.tensors["hidden_states"].device.type, "cpu")

    def test_timing_synchronizes_both_cuda_devices(self):
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            _clock(torch.device("cuda:0"), torch.device("cuda:1"))
        self.assertEqual({call.args[0] for call in synchronize.call_args_list},
                         {torch.device("cuda:0"), torch.device("cuda:1")})


if __name__ == "__main__":
    unittest.main()
