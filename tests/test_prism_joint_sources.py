"""CPU preflight contracts for a single exact native source across joint arms."""

import copy
import unittest

from draftfit.config import Config, ModelConfig
from draftfit.training.joint_family import validate_joint_family
from tests import test_prism_joint_config as fixtures


class PrismJointSourcesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrismJointConfigTest()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.stages = self.fixture.stages()

    def test_matching_native_source_is_accepted(self):
        validate_joint_family(self.stages)

    def test_each_native_source_mismatch_is_rejected_in_either_arm(self):
        bridge = self.fixture.root / "other-capture.so"
        bridge.write_bytes(b"another CPU fixture; never loaded")
        changes = {
            "prism_capture_bridge": str(bridge),
            "target_gguf_sha256": "a" * 64,
            "gguf_context_size": 2048,
            "gguf_chunk_size": 64,
        }
        for field, value in changes.items():
            for index in (0, -1):
                with self.subTest(field=field, arm=index):
                    stages = copy.deepcopy(self.stages)
                    raw = stages[index].config.model_dump(mode="json")
                    raw["model"][field] = value
                    stages[index].config.model = Config.model_validate(raw).model
                    with self.assertRaisesRegex(
                        ValueError, rf"differing model\.{field}$"
                    ):
                        validate_joint_family(stages)

    def test_matching_nondefault_native_settings_are_accepted(self):
        for stage in self.stages:
            raw = stage.config.model_dump(mode="json")
            raw["model"].update(gguf_context_size=2048, gguf_chunk_size=64)
            stage.config.model = Config.model_validate(raw).model
        validate_joint_family(self.stages)

    def _use_transformers(self, stage):
        raw = stage.config.model_dump(mode="json")
        raw["model"]["target_backend"] = "transformers"
        for field in (
            "prism_capture_bridge",
            "target_gguf_sha256",
            "gguf_context_size",
            "gguf_chunk_size",
        ):
            raw["model"][field] = ModelConfig.model_fields[field].default
        stage.config.model = Config.model_validate(raw).model

    def test_mixed_backend_types_are_rejected_in_either_arm(self):
        for index in (0, -1):
            with self.subTest(arm=index):
                stages = copy.deepcopy(self.stages)
                self._use_transformers(stages[index])
                with self.assertRaisesRegex(
                    ValueError, r"differing model\.target_backend$"
                ):
                    validate_joint_family(stages)

    def test_matching_transformers_source_is_accepted(self):
        for stage in self.stages:
            self._use_transformers(stage)
        validate_joint_family(self.stages)


if __name__ == "__main__":
    unittest.main()
