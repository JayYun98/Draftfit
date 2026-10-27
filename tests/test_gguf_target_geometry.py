"""Real GGUF descriptors must reach bundle assembly without HF/JSON loading."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from draftfit.modeling.target import gguf_parts
from draftfit.modeling.target.target_utils import load_target_config, target_vocab_size
from tests.test_gguf_parts import _fixture


class GGUFGeometryTest(unittest.TestCase):
    def test_bundle_geometry_comes_from_local_gguf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.gguf"
            _fixture(path)
            with (patch.object(gguf_parts, "_VOCAB_SIZE", 4),
                  patch("draftfit.modeling.target.target_utils.AutoConfig.from_pretrained",
                        side_effect=AssertionError("GGUF passed to Transformers"))):
                config = load_target_config(str(path), revision="pinned-native-target")
            self.assertEqual((config.hidden_size, target_vocab_size(config),
                              config.num_hidden_layers), (5120, 4, 64))
            self.assertEqual(config.model_type, "prism_gguf_qwen35")
            self.assertFalse(config.tie_word_embeddings)

    def test_malformed_gguf_does_not_fall_back_to_hub(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.gguf"
            path.write_bytes(b"not a GGUF")
            with patch("draftfit.modeling.target.target_utils.AutoConfig.from_pretrained") as hf:
                with self.assertRaises(ValueError):
                    load_target_config(str(path))
                hf.assert_not_called()
