"""Gemma target semantics and mixed-attention family training on real tensors."""

import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForCausalLM,
    Gemma4UnifiedForConditionalGeneration,
)

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from draftfit.modeling.draft.dflash2 import DFlash2DraftModel
from draftfit.modeling.target.target_utils import TargetEmbeddingsAndHead
from tests.test_dflash2_integration import tiny_config


class HumanizerTrainingTest(unittest.TestCase):
    def test_native_gemma_components_and_tied_weights(self):
        config = Gemma4UnifiedTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            num_global_key_value_heads=1,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
            final_logit_softcapping=30.0,
        )
        ids = torch.tensor([[2, 5, 6, 7]])
        for full in (False, True):
            model = (
                Gemma4UnifiedForConditionalGeneration(
                    Gemma4UnifiedConfig(text_config=config)
                )
                if full
                else Gemma4UnifiedForCausalLM(config)
            ).eval()
            text = model.model.language_model if full else model.model
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                parts = TargetEmbeddingsAndHead.from_pretrained(
                    directory, device="cpu", dtype=torch.float32
                )
            torch.testing.assert_close(parts.embed_tokens(ids), text.embed_tokens(ids))
            self.assertEqual(
                parts.embed_tokens.weight.data_ptr(), parts.lm_head.weight.data_ptr()
            )
            torch.testing.assert_close(
                parts.embed_tokens.weight, text.embed_tokens.weight
            )
            with torch.no_grad():
                hidden = text(input_ids=ids).last_hidden_state
                torch.testing.assert_close(
                    parts.lm_head(hidden), model(input_ids=ids).logits
                )
            # Nonlinear correction must occur once, after output_multiplier.
            hidden = torch.randn(1, 4, 32) * 100
            expected = 30 * torch.tanh(model.lm_head(hidden) * 1.5 / 30)
            for wrapper_cls in (OnlineDFlashModel, OnlineDFlash2Model):
                draft = DFlash2DraftModel(
                    tiny_config(final_logit_softcapping=30.0, output_multiplier=1.5)
                )
                wrapper = wrapper_cls(
                    draft,
                    parts.lm_head,
                    parts.embed_tokens,
                    63,
                    block_size=4,
                    attention_backend="sdpa",
                )
                torch.testing.assert_close(wrapper._compute_logits(hidden), expected)
            for dtype in (torch.float32, torch.bfloat16):
                text.to(dtype=dtype)
                parts.to(dtype=dtype)
                torch.testing.assert_close(
                    parts.embed_tokens(ids), text.embed_tokens(ids), rtol=0, atol=0
                )

    def test_mixed_masks_updates_and_roundtrip(self):
        torch.manual_seed(8)
        for causal in (None, False, True):
            config = tiny_config(
                layer_types=["sliding_attention", "full_attention"],
                sliding_window=2,
                is_causal=causal,
            )
            draft = DFlash2DraftModel(config)
            wrapper = OnlineDFlash2Model(
                draft,
                torch.nn.Linear(32, 64, bias=False),
                torch.nn.Embedding(64, 32),
                63,
                block_size=4,
                attention_backend="sdpa",
                num_anchors=2,
            )
            args = dict(
                anchor_positions=torch.tensor([[5, 8]]),
                block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
                S=12,
                block_size=4,
                device="cpu",
            )
            seen = []

            def record(module, args, kwargs):
                seen.append(kwargs["attention_mask"].clone())

            hooks = [
                layer.self_attn.register_forward_pre_hook(record, with_kwargs=True)
                for layer in draft.layers
            ]
            before = (
                draft.layers[0].attention_conv.kernel_projection.weight.detach().clone()
            )
            optim = torch.optim.AdamW(draft.parameters(), lr=1e-3)
            with patch.object(
                wrapper,
                "_sample_anchor_positions",
                return_value=(args["anchor_positions"], args["block_keep_mask"]),
            ):
                loss, _, _ = wrapper(
                    torch.randint(1, 63, (1, 12)),
                    torch.randn(1, 12, 64),
                    torch.ones(1, 12),
                )
            for hook in hooks:
                hook.remove()
            for i, window in enumerate((2, None)):
                effective = bool(window) if causal is None else causal
                expected = create_dflash_sdpa_mask(
                    **args, is_causal=effective, sliding_window=window
                )
                torch.testing.assert_close(seen[i], expected)
                block = create_dflash_block_mask(
                    **args, is_causal=effective, sliding_window=window
                )
                for q in range(8):
                    for kv in range(20):
                        got = block.mask_mod(
                            torch.tensor(0),
                            torch.tensor(0),
                            torch.tensor(q),
                            torch.tensor(kv),
                        )
                        self.assertEqual(bool(got), bool(expected[0, 0, q, kv]))
            # Last draft position cannot see first when block length exceeds window.
            self.assertFalse(bool(seen[0][0, 0, 3, 12]))
            self.assertTrue(bool(seen[1][0, 0, 3, 12]))
            self.assertFalse(torch.equal(seen[0], seen[1]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in draft.parameters()
                    if p.grad is not None
                )
            )
            optim.step()
            self.assertFalse(
                torch.equal(
                    before, draft.layers[0].attention_conv.kernel_projection.weight
                )
            )
            with tempfile.TemporaryDirectory() as directory:
                draft.save_pretrained(directory)
                restored = DFlash2DraftModel.from_pretrained(directory)
                self.assertEqual(restored.config.layer_types, config.layer_types)
                self.assertEqual(restored.config.is_causal, causal)
                for key, value in draft.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[key], value, rtol=0, atol=0
                    )
