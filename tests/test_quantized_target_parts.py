"""Portable CPU regression for packed readout math, dtype and strict metadata."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from draftfit.modeling.target.quantized_parts import export_embedding, load_target_parts


class QuantizedTargetPartsTest(unittest.TestCase):
    def fixture(self, path):
        rows, width = 4, 512
        signs = torch.where(torch.arange(width) % 3 != 0, 1.0, -1.0)
        weights = {}
        for name in ["model.embed_tokens", "lm_head"]:
            weights[name + ".weight"] = torch.full(
                (rows, width // 16), 0xE4E4E4E4, dtype=torch.uint32
            )
            weights[name + ".scales"] = torch.full(
                (rows, width // 128), 0.25, dtype=torch.float16
            )
            weights[name + ".biases"] = torch.full(
                (rows, width // 128), -0.25, dtype=torch.float16
            )
            weights[name + ".signs"] = signs.clone()
        config = {
            "schema_version": 1,
            "model_type": "prism_hadamard_qwen35",
            "text_config": {
                "vocab_size": rows,
                "hidden_size": width,
                "tie_word_embeddings": False,
            },
            "modules": [
                {
                    "path": name,
                    "embedding": name.startswith("model."),
                    "dtype": "float16",
                    "block": 512,
                }
                for name in ["model.embed_tokens", "lm_head"]
            ],
        }
        (path / "config.json").write_text(json.dumps(config))
        save_file(weights, str(path / "model.safetensors"))
        return config, weights, signs

    def test_independent_sylvester_reference_readout_gradient_and_embedding_export(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, weights, signs = self.fixture(path)
            matrix = np.array([[1]], dtype=np.float32)
            for _ in range(9):
                matrix = np.kron(matrix, np.array([[1, 1], [1, -1]], dtype=np.float32))
            matrix = torch.from_numpy(matrix) / 512**0.5
            dense = torch.tensor([-0.25, 0, 0.25, 0.5]).repeat(4, 128)
            expected_embedding = ((dense @ matrix) * signs).half()
            parts = load_target_parts(path, device="cpu")
            torch.testing.assert_close(
                export_embedding(path), expected_embedding, atol=1e-4, rtol=1e-4
            )
            torch.manual_seed(37)
            x = torch.randn(1, 2, 512).half().requires_grad_()
            reference = torch.nn.functional.linear(
                ((x.float() * signs) @ matrix).half(), dense.half()
            )
            logits = parts.lm_head(x)
            torch.testing.assert_close(logits, reference, atol=0.02, rtol=0.003)
            logits.float().square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            frozen = parts.lm_head.weight.clone()
            parts.to(dtype=torch.bfloat16)
            self.assertEqual(parts.lm_head.weight.dtype, torch.float16)
            torch.testing.assert_close(parts.lm_head.weight, frozen, rtol=0, atol=0)
            self.assertEqual(
                parts.embed_tokens(torch.tensor([0, 2])).dtype, torch.bfloat16
            )

    def test_reject_bad_signs_kind_and_missing_readout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, weights, signs = self.fixture(path)
            weights["lm_head.signs"] = torch.zeros_like(signs)
            save_file(weights, str(path / "model.safetensors"))
            with self.assertRaises(ValueError):
                load_target_parts(path, device="cpu")
            weights["lm_head.signs"] = signs
            del weights["lm_head.biases"]
            save_file(weights, str(path / "model.safetensors"))
            with self.assertRaises(ValueError):
                load_target_parts(path, device="cpu")
            # Embedding-only export must not load or require the output head.
            self.assertEqual(export_embedding(path).shape, (4, 512))
