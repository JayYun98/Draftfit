"""Actual local target heads retain requested dtype and exact tensor geometry."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3Config, Qwen3ForCausalLM

from draftfit.modeling.target.target_head import TargetHead


class TargetHeadLoadingTest(unittest.TestCase):
    def test_nested_text_config_loads_the_exact_language_head(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3_5Config(
                text_config={
                    "vocab_size": 16,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "layer_types": ["full_attention"],
                }
            ).save_pretrained(directory)
            torch.manual_seed(7)
            weight = torch.randn(16, 16)
            save_file(
                {"lm_head.weight": weight},
                str(Path(directory) / "model.safetensors"),
            )
            with patch(
                "draftfit.modeling.target.target_head.get_local_device",
                return_value=torch.device("cpu"),
            ):
                head = TargetHead.from_pretrained(directory, dtype=torch.float32)
            hidden = torch.randn(1, 3, 16)
            self.assertEqual((head.hidden_size, head.vocab_size), (16, 16))
            self.assertFalse(head.fc.weight.requires_grad)
            torch.testing.assert_close(head.fc.weight, weight, atol=0, rtol=0)
            torch.testing.assert_close(head(hidden), hidden @ weight.T, atol=0, rtol=0)

    def test_unsharded_projection_matches_target_at_requested_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(13)
            target = Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=16,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    tie_word_embeddings=False,
                )
            )
            target.save_pretrained(directory)
            hidden = torch.randn(1, 3, 16)
            with patch(
                "draftfit.modeling.target.target_head.get_local_device",
                return_value=torch.device("cpu"),
            ):
                head = TargetHead.from_pretrained(directory, dtype=torch.float32)
                default = TargetHead.from_pretrained(directory)
            self.assertEqual(head.fc.weight.dtype, torch.float32)
            self.assertEqual(default.fc.weight.dtype, torch.bfloat16)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in head.parameters())
            )
            torch.testing.assert_close(
                head(hidden), target.lm_head(hidden), atol=0, rtol=0
            )

    def test_packed_head_is_rejected_without_changing_float_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config(vocab_size=16, hidden_size=16).save_pretrained(directory)
            save_file(
                {"lm_head.weight": torch.ones(16, 16, dtype=torch.uint8)},
                str(Path(directory) / "model.safetensors"),
            )
            head = TargetHead(directory)
            before = head.fc.weight.detach().clone()
            with self.assertRaisesRegex(
                ValueError, "packed embeddings or output heads"
            ):
                head.load_weights(directory)
            torch.testing.assert_close(head.fc.weight, before, atol=0, rtol=0)

    def test_incorrect_head_shape_is_rejected_before_copy_broadcast(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config(vocab_size=16, hidden_size=16).to_json_file(
                Path(directory) / "config.json"
            )
            save_file(
                {"lm_head.weight": torch.zeros(1, 16)},
                str(Path(directory) / "model.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                TargetHead.from_pretrained(directory, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
