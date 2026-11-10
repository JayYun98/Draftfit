import unittest

import torch
from transformers import Qwen3Config

from draftfit.modeling.draft.dflash import DFlashDraftModel
from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from draftfit.modeling.draft.dspark import DSparkDraftModel


class DFlashMixedPrecisionTest(unittest.TestCase):
    def test_fp32_residual_features_match_explicit_bf16_conversion(self):
        for model_class, config_class in (
            (DFlashDraftModel, Qwen3Config),
            (DSparkDraftModel, Qwen3Config),
            (DFlash2DraftModel, DFlash2Config),
        ):
            with self.subTest(family=model_class.__name__):
                torch.manual_seed(17)
                config = config_class(
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    num_target_layers=6,
                    vocab_size=32,
                    block_size=2,
                    conv_group_size=4,
                    mask_token_id=31,
                    max_position_embeddings=64,
                    layer_types=["full_attention"],
                    dflash_config={"target_layer_ids": [0, 2, 5], "markov_rank": 2},
                )
                config._attn_implementation = "sdpa"
                model = model_class(config).to(dtype=torch.bfloat16).eval()
                features = torch.randn(1, 4, model.fc.in_features)
                original = features.clone()
                noise = torch.randn(1, 2, 8, dtype=torch.bfloat16)
                inputs = dict(
                    position_ids=torch.arange(6).unsqueeze(0), noise_embedding=noise
                )
                expected = model(target_hidden=features.to(torch.bfloat16), **inputs)
                actual = model(target_hidden=features, **inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(features, original, rtol=0, atol=0)
                self.assertEqual(features.dtype, torch.float32)
                actual.float().square().mean().backward()
                self.assertIsNotNone(model.fc.weight.grad)
                self.assertTrue(torch.isfinite(model.fc.weight.grad).all())
                self.assertGreater(model.fc.weight.grad.float().abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
