"""Published DSpark configuration and target-copy compatibility checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from draftfit.modeling.draft.dflash import apply_rotary_pos_emb
from draftfit.modeling.auto import AutoDraftModel
from draftfit.training.model_loading import (
    _draft_config_from_dict, _load_pretrained_draft_state,
    convert_family_draft_config, warm_start_draft_model,
)
from draftfit.training.official_dspark import verify_taichu_target_copies


def _base_config():
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "max_position_embeddings": 128,
        "layer_types": ["full_attention"], "block_size": 4,
        "num_target_layers": 4, "dflash_config": {
            "target_layer_ids": [1], "mask_token_id": 31,
            "projector_type": "dspark", "markov_rank": 4,
            "sample_from_anchor": True,
        },
    }


def _nested_taichu_config():
    base = _base_config()
    transformer = {key: value for key, value in base.items() if key not in (
        "architectures", "block_size", "num_target_layers", "dflash_config"
    )}
    return {
        "architectures": ["DSparkDraftModel"],
        "transformer_layer_config": transformer,
        "aux_hidden_state_layer_ids": [1], "block_size": 4,
        "mask_token_id": 31, "sample_from_anchor": True,
        "markov_rank": 4, "markov_head_type": "vanilla",
        "enable_confidence_head": False,
    }


class OfficialDSparkWarmStartTest(unittest.TestCase):
    def test_taichu_nested_config_keeps_official_geometry(self):
        payload = _nested_taichu_config()
        config = _draft_config_from_dict(payload)
        self.assertEqual(config.dflash_config["target_layer_ids"], [1])
        self.assertEqual(config.dflash_config["mask_token_id"], 31)
        self.assertEqual(config.official_dspark_format, "taichu_speculators")

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            incompatible = _draft_config_from_dict({**_base_config(), "block_size": 5})
            with self.assertRaisesRegex(ValueError, "incompatible block_size"):
                _load_pretrained_draft_state(
                    directory, draft_config=incompatible, cache_dir=None,
                    trust_remote_code=False, strategy="dspark",
                )

    def test_published_taichu_tensors_load_with_extra_target_copies(self):
        payload = _nested_taichu_config()
        config = _draft_config_from_dict(payload)
        original = AutoDraftModel.from_config(config)
        state = {key: value.detach().clone().contiguous()
                 for key, value in original.state_dict().items()}
        state["embed_tokens.weight"] = torch.zeros(32, 16)
        state["lm_head.weight"] = torch.zeros(32, 16)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            save_file(state, str(Path(directory, "model.safetensors")))
            destination = AutoDraftModel.from_config(config)
            report = warm_start_draft_model(
                destination, directory, draft_config=config, strategy="dspark"
            )
            self.assertEqual(report.loaded_keys, len(state) - 2)
            for key in original.state_dict():
                self.assertTrue(
                    torch.equal(destination.state_dict()[key], state[key]), key
                )

    def test_lfm_alias_preserves_interleaved_rope(self):
        payload = _base_config()
        payload.update(architectures=["Lfm2DSparkDraftModel"],
                       rope_is_neox_style=False, rope_theta=5_000_000,
                       enable_confidence_head=True, markov_rank=4,
                       markov_head_type="vanilla")
        payload["dflash_config"] = {
            "target_layer_ids": [1], "mask_token_id": 31,
            "num_target_layers": 4,
        }
        payload.pop("num_target_layers")
        config = _draft_config_from_dict(payload)
        self.assertEqual(config.architectures, ["DSparkDraftModel"])
        self.assertFalse(config.rope_is_neox_style)
        self.assertTrue(config.dflash_config["sample_from_anchor"])
        self.assertTrue(config.dflash_config["confidence_head_with_markov"])

        converted = convert_family_draft_config(
            config, "dflash2", retrain_anchor_alignment=True
        )
        self.assertEqual(converted.architectures, ["DFlash2DraftModel"])
        self.assertFalse(converted.dflash_config["sample_from_anchor"])
        with tempfile.TemporaryDirectory() as directory:
            source_model = AutoDraftModel.from_config(config)
            source_model.save_pretrained(directory)
            destination = AutoDraftModel.from_config(converted)
            report = warm_start_draft_model(
                destination, directory, draft_config=converted,
                strategy="dflash2", retrain_anchor_alignment=True,
            )
            self.assertEqual(report.loaded_keys, 14)
            torch.testing.assert_close(
                destination.layers[0].self_attn.q_proj.weight,
                source_model.layers[0].self_attn.q_proj.weight,
            )

    def test_taichu_mixed_layout_selects_text_dflash2(self):
        payload = _nested_taichu_config()
        payload["transformer_layer_config"].update(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16, use_sliding_window=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000,
                             "partial_rotary_factor": 1.0,
                             "mrope_section": [2, 1, 1]},
        )
        config = _draft_config_from_dict(payload)
        converted = convert_family_draft_config(
            config, "dflash2", retrain_anchor_alignment=True
        )
        self.assertEqual(converted.architectures, ["TaichuDFlash2DraftModel"])
        self.assertEqual(converted.layer_types, ["sliding_attention", "full_attention"])
        self.assertFalse(converted.dflash_config["sample_from_anchor"])

    def test_interleaved_rotary_pairs_adjacent_features(self):
        q = torch.tensor([[[[1., 2., 3., 4.]]]])
        cos = torch.zeros((1, 1, 4))
        sin = torch.ones((1, 1, 4))
        result, _ = apply_rotary_pos_emb(
            q, q, cos, sin, is_neox_style=False
        )
        self.assertTrue(torch.equal(result, torch.tensor([[[[-2., 1., -4., 3.]]]])))
        frequencies, _ = apply_rotary_pos_emb(
            torch.ones_like(q), torch.ones_like(q),
            torch.tensor([[[1., 2., 1., 2.]]]), torch.zeros_like(sin),
            is_neox_style=False,
        )
        self.assertTrue(
            torch.equal(frequencies, torch.tensor([[[[1., 1., 2., 2.]]]]))
        )

    def test_taichu_target_copies_must_match(self):
        embed = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        head = embed + 1
        parts = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=embed.clone()),
            lm_head=SimpleNamespace(weight=head.clone()),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file({"embed_tokens.weight": embed, "lm_head.weight": head},
                      str(Path(directory, "model.safetensors")))
            verify_taichu_target_copies(directory, parts)
            parts.lm_head.weight[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "lm_head.weight differs"):
                verify_taichu_target_copies(directory, parts)

    def test_published_lfm_tensors_load_without_reinitialization(self):
        payload = _base_config()
        payload.update(
            architectures=["Lfm2DSparkDraftModel"],
            rope_is_neox_style=False, rope_theta=5_000_000,
            enable_confidence_head=True, markov_rank=4,
            markov_head_type="vanilla",
        )
        payload["dflash_config"] = {
            "target_layer_ids": [1], "mask_token_id": 31,
            "num_target_layers": 4,
        }
        payload.pop("num_target_layers")
        config = _draft_config_from_dict(payload)
        original = AutoDraftModel.from_config(config)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            state = {key: value.detach().clone().contiguous()
                     for key, value in original.state_dict().items()}
            save_file(state, str(Path(directory, "model.safetensors")))
            destination = AutoDraftModel.from_config(config)
            report = warm_start_draft_model(
                destination, directory, draft_config=config, strategy="dspark"
            )
            self.assertEqual(report.loaded_keys, len(state))
            for key, value in state.items():
                self.assertTrue(torch.equal(destination.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
