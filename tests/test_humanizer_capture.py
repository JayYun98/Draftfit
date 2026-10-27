"""Real tiny Gemma 4 models: prenorm taps and full-wrapper selection."""

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModel
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
)

from draftfit.offline_capture.transformers import OfflineTransformersCapture


def check_gemma4_text_and_humanizer_wrapper(tmp_path):
    config = Gemma4UnifiedTextConfig(
        vocab_size=32,
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
    ids = torch.tensor([[2, 5, 6, 7, 8, 9]])
    for wrapped in (False, True):
        model = AutoModel.from_config(
            Gemma4UnifiedConfig(text_config=config) if wrapped else config,
            attn_implementation="eager",
        ).eval()
        path = tmp_path / str(wrapped)
        model.save_pretrained(path)
        teacher = OfflineTransformersCapture.from_pretrained(
            path, attn_implementation="eager"
        )
        text = teacher._model
        teacher.set_capture_layers([0, 1], capture_method="dflash")
        with torch.no_grad():
            expected = text(input_ids=ids, use_cache=False).last_hidden_state
        result = teacher.capture(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            loss_mask=torch.ones_like(ids),
        )
        torch.testing.assert_close(result.last_hidden_states, expected)
        torch.testing.assert_close(text.norm(result.hidden_states[..., 32:]), expected)
        torch.testing.assert_close(
            text.embed_tokens(ids), text.embed_tokens.weight[ids] * 32**0.5
        )
        assert result.hidden_states.shape == (1, 6, 64)
        assert not result.hidden_states.requires_grad
        assert not text.layers[0]._forward_hooks


class HumanizerCaptureTest(unittest.TestCase):
    def test_gemma4_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            check_gemma4_text_and_humanizer_wrapper(Path(directory))
