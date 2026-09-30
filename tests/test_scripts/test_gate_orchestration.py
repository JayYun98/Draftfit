"""Dependency-light contracts for the retained gate helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "scripts" / "gates"
NORMALIZE = GATE_DIR / "normalize_dflash_export.py"


class TestNormalizeDFlashExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "normalize_dflash_export", NORMALIZE
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_normalizes_dispatch_without_dropping_domino_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DominoDraftModel"],
                        "auto_map": {"AutoModel": "domino.DominoDraftModel"},
                        "block_size": 16,
                        "dflash_config": {
                            "projector_type": "domino",
                            "gru_hidden_dim": 1024,
                        },
                    }
                ),
                encoding="utf-8",
            )

            normalized = self.module.normalize_export(str(path), 16)

            self.assertEqual(normalized["architectures"], ["DFlashDraftModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertEqual(normalized["dflash_config"]["projector_type"], "domino")
            self.assertEqual(normalized["dflash_config"]["gru_hidden_dim"], 1024)
            self.assertEqual(json.loads(path.read_text()), normalized)

    def test_normalizes_dspark_for_sglang(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DSparkDraftModel"],
                        "auto_map": {"AutoModel": "dspark.DSparkDraftModel"},
                        "block_size": 16,
                        "model_type": "qwen3",
                        "dflash_config": {
                            "projector_type": "dspark",
                            "markov_rank": 256,
                            "markov_head_type": "vanilla",
                            "enable_confidence_head": True,
                            "confidence_head_with_markov": True,
                            "mask_token_id": 248070,
                            "target_layer_ids": [2, 17, 32, 47, 62],
                        },
                    }
                ),
                encoding="utf-8",
            )

            normalized = self.module.normalize_export(str(path), 16)

            self.assertEqual(normalized["architectures"], ["Qwen3DSparkModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertEqual(normalized["markov_rank"], 256)
            self.assertEqual(normalized["markov_head_type"], "vanilla")
            self.assertTrue(normalized["enable_confidence_head"])
            self.assertTrue(normalized["confidence_head_with_markov"])
            self.assertEqual(
                normalized["dflash_config"]["target_layer_ids"],
                [2, 17, 32, 47, 62],
            )
            self.assertEqual(json.loads(path.read_text()), normalized)

    def test_rejects_dspark_without_a_positive_markov_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "block_size": 16,
                        "model_type": "qwen3",
                        "dflash_config": {
                            "projector_type": "dspark",
                            "markov_rank": 0,
                            "markov_head_type": "vanilla",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive integer markov_rank"):
                self.module.normalize_export(str(path), 16)

    def test_rejects_a_mismatched_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "block_size": 8,
                        "dflash_config": {"projector_type": "dflash"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 16"):
                self.module.normalize_export(str(path), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
