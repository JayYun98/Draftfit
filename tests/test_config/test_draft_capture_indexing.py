"""Capture indexing is an explicit, persisted run declaration."""

import unittest

from pydantic import ValidationError

from draftfit.config import Config, apply_overrides


class DraftCaptureIndexingConfigTest(unittest.TestCase):
    def test_json_and_cli_declarations_roundtrip_and_reject_unknowns(self):
        base = {
            "model": {"target_model_path": "/target"},
            "data": {"hidden_states_path": "/features"},
        }
        cfg = Config.model_validate(base)
        self.assertIsNone(cfg.model.draft_capture_layer_indexing)
        for indexing in ("block_output", "embedding_inclusive"):
            declared = apply_overrides(
                cfg, [f"model.draft_capture_layer_indexing={indexing}"]
            )
            self.assertEqual(
                Config.model_validate_json(
                    declared.model_dump_json()
                ).model.draft_capture_layer_indexing,
                indexing,
            )
        with self.assertRaises(ValidationError):
            apply_overrides(cfg, ["model.draft_capture_layer_indexing=auto"])
