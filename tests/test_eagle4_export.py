"""CPU weights-only EAGLE4 HF export and reload checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from draftfit.export.to_hf import export_to_hf
from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from tests.test_modeling.test_eagle4 import tiny_config

DRAFTFIT_HF_KEYS = {
    "embed_tokens.weight",
    "midlayer.self_attn.q_proj.weight",
    "midlayer.self_attn.k_proj.weight",
    "midlayer.self_attn.v_proj.weight",
    "midlayer.self_attn.o_proj.weight",
    "midlayer.mlp.gate_proj.weight",
    "midlayer.mlp.up_proj.weight",
    "midlayer.mlp.down_proj.weight",
    "midlayer.hidden_norm.weight",
    "midlayer.input_layernorm.weight",
    "midlayer.post_attention_layernorm.weight",
    "fc.weight",
    "fc_norm.0.weight",
    "fc_norm.1.weight",
    "fc_norm.2.weight",
    "norm.weight",
    "lm_head.weight",
    "efficient_lm_head_down_proj.weight",
    "efficient_lm_head.weight",
}


class Eagle4ExportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = LlamaForCausalLMEagle4(tiny_config()).bfloat16().eval()
        self.model.config.save_pretrained(self.root)
        self.weights = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    def export(self, weights, **kwargs):
        checkpoint = self.root / "training_state.pt"
        torch.save({"strategy": "eagle4", "draft_state_dict": weights}, checkpoint)
        output = self.root / "export"
        with patch.object(
            LlamaForCausalLMEagle4,
            "initialize_candidate_head",
            side_effect=AssertionError(
                "export must preserve candidate factors without SVD"
            ),
        ):
            export_to_hf(
                str(checkpoint), str(self.root / "config.json"), str(output), **kwargs
            )
            reloaded, info = AutoDraftModel.from_pretrained(
                output,
                local_files_only=True,
                dtype=torch.bfloat16,
                output_loading_info=True,
            )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(info.get(key), (key, info))
        saved = load_file(str(output / "model.safetensors"))
        self.assertEqual(set(saved), DRAFTFIT_HF_KEYS)
        self.assertEqual(set(reloaded.state_dict()), DRAFTFIT_HF_KEYS)
        for key, expected in self.weights.items():
            torch.testing.assert_close(saved[key], expected, rtol=0, atol=0, msg=key)
            torch.testing.assert_close(
                reloaded.state_dict()[key], expected, rtol=0, atol=0, msg=key
            )
        return reloaded.eval()

    @torch.inference_mode()
    def test_draftfit_hf_export_reloads_all_19_learned_tensors(self):
        reloaded = self.export(self.weights)
        hidden, embedded = (
            torch.randn(1, 4, 48).bfloat16(),
            torch.randn(1, 4, 16).bfloat16(),
        )
        expected = self.model(hidden, embedded)
        actual = reloaded(hidden, embedded)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            reloaded.compute_logits(actual),
            self.model.compute_logits(expected),
            rtol=0,
            atol=0,
        )
        for actual_part, expected_part in zip(
            reloaded.candidate_logits(actual), self.model.candidate_logits(expected)
        ):
            torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)

    def test_official_input_aliases_export_to_internal_hf_keys(self):
        official = {
            key.replace("midlayer.", "layers.0.", 1): value
            for key, value in self.weights.items()
        }
        self.export(official)
        saved = load_file(str(self.root / "export" / "model.safetensors"))
        self.assertFalse(any(key.startswith("layers.0.") for key in saved))
        self.assertIn("midlayer.self_attn.q_proj.weight", saved)
        for key, expected in official.items():
            internal = key.replace("layers.0.", "midlayer.", 1)
            torch.testing.assert_close(saved[internal], expected, rtol=0, atol=0)

    def test_native_precision_config_survives_weights_only_export(self):
        self.model.config.eagle4_precision = "vllm"
        self.model.config.save_pretrained(self.root)
        reloaded = self.export(self.weights)
        self.assertEqual(reloaded.config.eagle4_precision, "vllm")

    def test_export_restores_excluded_embedding_from_floating_target(self):
        target = self.root / "target"
        target.mkdir()
        save_file(
            {"model.embed_tokens.weight": self.weights["embed_tokens.weight"]},
            str(target / "model.safetensors"),
        )
        state = {
            key: value
            for key, value in self.weights.items()
            if key != "embed_tokens.weight"
        }
        self.export(state, embedding_source=str(target))

    def test_missing_embedding_requires_a_real_source(self):
        state = {
            key: value
            for key, value in self.weights.items()
            if key != "embed_tokens.weight"
        }
        with self.assertRaisesRegex(ValueError, "embedding_source"):
            self.export(state)

    def test_packed_target_embedding_remains_rejected(self):
        target = self.root / "packed-target"
        target.mkdir()
        save_file(
            {"model.embed_tokens.weight": torch.ones(32, 16, dtype=torch.uint8)},
            str(target / "model.safetensors"),
        )
        state = {
            key: value
            for key, value in self.weights.items()
            if key != "embed_tokens.weight"
        }
        with self.assertRaisesRegex(ValueError, "packed embeddings or output heads"):
            self.export(state, embedding_source=str(target))

    def test_missing_candidate_factors_are_not_reinitialized(self):
        for name in ("efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.export(
                    {key: value for key, value in self.weights.items() if key != name}
                )

    def test_ambiguous_official_and_canonical_keys_are_rejected(self):
        state = dict(self.weights)
        state["layers.0.self_attn.q_proj.weight"] = state[
            "midlayer.self_attn.q_proj.weight"
        ]
        with self.assertRaisesRegex(ValueError, "conflicting EAGLE4 checkpoint key"):
            self.export(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
