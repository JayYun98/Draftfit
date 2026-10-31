"""CPU contracts for exact Prism joint capture; native execution is a separate gate."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from draftfit.config import Config, ModelConfig
from draftfit.family import build_family_plan
from draftfit.inference.capture import TeacherCaptureBatch
from draftfit.offline_capture.prism_gguf import PRISM_REVISION, OfflinePrismGGUFCapture
from draftfit.training.joint_family import _semantic_fingerprint
from draftfit.training.joint_online import JointOnlineSource
from tests.test_joint_online_cpu import online_fixture


class PrismJointConfigTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="prism-joint-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        online_fixture(self.root, seeded=False)
        self.raw = json.loads((self.root / "base.json").read_text())
        self.gguf = self.root / "target.gguf"
        self.gguf.write_bytes(b"GGUF CPU fixture; not native model weights")
        self.bridge = self.root / "capture.so"
        self.bridge.write_bytes(b"bridge CPU fixture; not a loadable library")
        self.raw["model"].update(
            target_backend="prism_gguf",
            target_model_path=str(self.gguf),
            prism_capture_bridge=str(self.bridge),
            target_gguf_sha256=hashlib.sha256(self.gguf.read_bytes()).hexdigest(),
        )
        self.env = patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def stages(self, raw=None):
        (self.root / "base.json").write_text(
            json.dumps(self.raw if raw is None else raw)
        )
        return build_family_plan(str(self.root / "family.json"))[1]

    def test_defaults_and_native_bounds(self):
        model = Config.model_validate(self.raw).model
        self.assertEqual((model.gguf_context_size, model.gguf_chunk_size), (1024, 32))
        for field, minimum, maximum in (
            ("gguf_context_size", 64, 8192),
            ("gguf_chunk_size", 1, 512),
        ):
            for value in (minimum, maximum):
                with self.subTest(field=field, value=value):
                    raw = copy.deepcopy(self.raw["model"])
                    raw[field] = value
                    self.assertEqual(
                        getattr(ModelConfig.model_validate(raw), field), value
                    )
            for value in (minimum - 1, 0, -1, maximum + 1, True, 1.5, "32"):
                with self.subTest(field=field, value=value):
                    raw = copy.deepcopy(self.raw["model"])
                    raw[field] = value
                    with self.assertRaises(ValueError):
                        ModelConfig.model_validate(raw)

    def test_sha_and_local_files_are_required(self):
        for value in (
            None,
            "",
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            "a" * 64 + "\n",
            123,
        ):
            with self.subTest(sha=value):
                raw = copy.deepcopy(self.raw)
                raw["model"]["target_gguf_sha256"] = value
                with self.assertRaises(ValueError):
                    Config.model_validate(raw)
        for field, values in (
            (
                "target_model_path",
                (
                    "org/remote",
                    str(self.root),
                    str(self.bridge),
                    str(self.root / "missing.gguf"),
                    f"file://{self.gguf}",
                ),
            ),
            (
                "prism_capture_bridge",
                (
                    None,
                    "",
                    "org/bridge",
                    str(self.root),
                    str(self.root / "missing.so"),
                    f"file://{self.bridge}",
                ),
            ),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    raw = copy.deepcopy(self.raw)
                    raw["model"][field] = value
                    with self.assertRaises(ValueError):
                        Config.model_validate(raw)

    def test_joint_only_text_explicit_draft_and_no_remote_code(self):
        for section, field, value in (
            ("training", "joint_online_capture", False),
            ("model", "input_modality", "image"),
            ("model", "draft_model_config", None),
            ("model", "draft_model_config", ""),
            ("model", "trust_remote_code", True),
            ("training", "joint_target_device", "cuda:1"),
            ("training", "joint_target_device", "mps"),
        ):
            with self.subTest(field=field, value=value):
                raw = copy.deepcopy(self.raw)
                raw[section][field] = value
                with self.assertRaises(ValueError):
                    Config.model_validate(raw)
        raw = copy.deepcopy(self.raw)
        raw["training"]["joint_target_device"] = "cuda:0"
        Config.model_validate(raw)  # Config validation does not initialize CUDA.

    def test_context_reserves_bounded_verification_tokens(self):
        raw = copy.deepcopy(self.raw)
        raw["data"]["max_length"] = 64
        raw["model"]["gguf_context_size"] = 64
        Config.model_validate(raw)
        raw["data"]["max_length"] = 65
        with self.assertRaisesRegex(ValueError, "context"):
            Config.model_validate(raw)
        raw["data"]["max_length"] = 64
        raw["training"]["joint_verification"] = {"max_new_tokens": 3}
        raw["model"]["gguf_context_size"] = 67
        Config.model_validate(raw)
        raw["model"]["gguf_context_size"] = 66
        with self.assertRaisesRegex(ValueError, "context"):
            Config.model_validate(raw)

    def test_prism_options_cannot_leak_to_other_backends(self):
        for backend in ("sglang", "transformers", "vllm"):
            for field, value in (
                ("prism_capture_bridge", str(self.bridge)),
                ("target_gguf_sha256", "a" * 64),
                ("gguf_context_size", 2048),
                ("gguf_chunk_size", 64),
            ):
                with self.subTest(backend=backend, field=field):
                    with self.assertRaisesRegex(ValueError, "prism_gguf"):
                        ModelConfig.model_validate(
                            {
                                "target_model_path": "org/model",
                                "target_backend": backend,
                                field: value,
                            }
                        )
            ModelConfig.model_validate(
                {"target_model_path": "org/model", "target_backend": backend}
            )
        for backend in ("sglang", "vllm"):
            raw = json.loads((self.root / "base.json").read_text())
            raw["model"]["target_backend"] = backend
            with self.assertRaisesRegex(ValueError, "joint_online_capture"):
                Config.model_validate(raw)

    def test_lazy_prism_factory_arguments_and_shared_capture(self):
        stages = self.stages()
        with (
            patch.object(OfflinePrismGGUFCapture, "from_pretrained") as factory,
            patch(
                "draftfit.training.joint_online.OfflineTransformersCapture.from_pretrained"
            ) as hf,
        ):
            source = JointOnlineSource(stages[0], True)
            factory.assert_not_called()
            teacher = factory.return_value

            def capture(*, input_ids, attention_mask, loss_mask):
                shape = (*input_ids.shape, 32)
                return TeacherCaptureBatch(
                    torch.ones(*input_ids.shape, 64),
                    torch.ones(shape),
                    input_ids,
                    attention_mask,
                    loss_mask,
                )

            teacher.capture.side_effect = capture
            for evaluation in (False, True):
                loader, _ = source.loader(
                    stages[0], 0, False, False, evaluation=evaluation
                )
                batch = next(iter(loader))
                self.assertEqual(batch.tensors["hidden_states"].device.type, "cpu")
            factory.assert_called_once_with(
                str(self.gguf.resolve()),
                device=torch.device("cpu"),
                bridge_path=str(self.bridge.resolve()),
                expected_sha256=self.raw["model"]["target_gguf_sha256"],
                torch_dtype=torch.bfloat16,
                context_size=1024,
                chunk_size=32,
            )
            teacher.set_capture_layers.assert_called_once_with(
                [1, 5], capture_method="dspark"
            )
            self.assertEqual(source.capture_calls, 2)
            hf.assert_not_called()

    def test_implicit_unsupported_device_fails_before_native_load(self):
        stage = self.stages()[0]
        with (
            patch(
                "draftfit.training.joint_online.get_device",
                return_value=torch.device("cuda:1"),
            ),
            patch.object(OfflinePrismGGUFCapture, "from_pretrained") as factory,
        ):
            with self.assertRaisesRegex(ValueError, "Prism"):
                source = JointOnlineSource(stage, True)
                loader, _ = source.loader(stage, 0, False, False)
                next(iter(loader))
            factory.assert_not_called()

    def test_resume_identity_tracks_bridge_bytes_revision_and_gguf_sha(self):
        stages = self.stages()
        source = JointOnlineSource(stages[0], True)
        identity = source.fingerprint_sources()
        self.assertIn(self.raw["model"]["target_gguf_sha256"], json.dumps(identity))
        self.assertIn(
            hashlib.sha256(self.bridge.read_bytes()).hexdigest(), json.dumps(identity)
        )
        self.assertIn(PRISM_REVISION, json.dumps(identity))
        for info in identity.values():
            info.clear()
        self.assertNotEqual(identity, source.fingerprint_sources())
        before = [
            _semantic_fingerprint(stages, source, fixed_only=fixed)
            for fixed in (False, True)
        ]
        # Same path/size/timestamp: resume must bind to the actual bridge bytes.
        original = self.bridge.stat()
        self.bridge.write_bytes(b"X" * original.st_size)
        os.utime(self.bridge, ns=(original.st_atime_ns, original.st_mtime_ns))
        changed = JointOnlineSource(stages[0], True)
        for fixed, expected in zip((False, True), before):
            self.assertNotEqual(
                expected, _semantic_fingerprint(stages, changed, fixed_only=fixed)
            )
        before = [
            _semantic_fingerprint(stages, changed, fixed_only=fixed)
            for fixed in (False, True)
        ]
        for stage in stages:
            stage.config.model.target_gguf_sha256 = "a" * 64
        changed = JointOnlineSource(stages[0], True)
        for fixed, expected in zip((False, True), before):
            self.assertNotEqual(
                expected, _semantic_fingerprint(stages, changed, fixed_only=fixed)
            )
        before = [
            _semantic_fingerprint(stages, changed, fixed_only=fixed)
            for fixed in (False, True)
        ]
        with patch("draftfit.offline_capture.prism_gguf.PRISM_REVISION", "b" * 40):
            changed = JointOnlineSource(stages[0], True)
        for fixed, expected in zip((False, True), before):
            self.assertNotEqual(
                expected, _semantic_fingerprint(stages, changed, fixed_only=fixed)
            )

    def test_bridge_replacement_after_fingerprinting_is_rejected(self):
        stage = self.stages()[0]
        source = JointOnlineSource(stage, True)
        self.bridge.write_bytes(b"changed bridge")
        with patch.object(OfflinePrismGGUFCapture, "from_pretrained") as factory:
            loader, _ = source.loader(stage, 0, False, False)
            with self.assertRaisesRegex(ValueError, "bridge changed"):
                next(iter(loader))
            factory.assert_not_called()

    def test_actual_family_cli_plan_without_loading_native_library(self):
        self.stages()
        env = os.environ.copy()
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        )
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "draftfit.cli",
                "family",
                "--manifest",
                "family.json",
                "--joint",
                "--plan",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["execution"], "joint")
        self.assertFalse((self.root / "outputs").exists())


class LegacyFingerprintTest(unittest.TestCase):
    def test_non_prism_fingerprints_match_pre_integration_revision(self):
        # Captured from 2fa7cdd before adding Prism fields (full, then fixed).
        expected = {
            "sglang": (
                "cdeb886f5e7ff73dea8d3e31d0aa7450833258c63465e2b15910dcdd3d0bf47c",
                "50ac5cfdc47f44cdc083da38a466395d56e0731aa695bcc9163487c331343fc0",
            ),
            "transformers": (
                "7e263e78c76565f492c91f53c0e33a6e4a9d783888241f891723d5dde89c359d",
                "3b073eb5c1702de7b7f5cb114281b46ed7c843616390949e2e822be339996939",
            ),
            "vllm": (
                "3abf43ab512d34ef367bd4f23f2d1834dd6eb7217888ca9853b1a43cef90d066",
                "58670565665ac50d485ff87906d0fda27b091ee02015800d9b64c13786e9993a",
            ),
        }
        for backend, fingerprints in expected.items():
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "/legacy/target",
                        "draft_model_config": "/legacy/draft.json",
                        "target_backend": backend,
                    },
                    "data": {
                        "hidden_states_path": "/legacy/train",
                        "eval_hidden_states_path": "/legacy/eval",
                    },
                    "training": {"strategy": "dspark", "eval_interval": 1},
                }
            )
            stage = NS(
                name="legacy",
                config=cfg,
                draft_config=NS(to_dict=lambda: {"fixture": "legacy"}),
            )
            source = NS(
                fingerprint_sources=lambda: {
                    "train": {"sha256": "train"},
                    "eval": {"sha256": "eval"},
                }
            )
            with patch(
                "draftfit.training.provenance.model_source_identity",
                return_value=("reference", "/legacy/target"),
            ):
                for fixed, fingerprint in zip((False, True), fingerprints):
                    with self.subTest(backend=backend, fixed=fixed):
                        self.assertEqual(
                            _semantic_fingerprint([stage], source, fixed_only=fixed),
                            fingerprint,
                        )


if __name__ == "__main__":
    unittest.main()
