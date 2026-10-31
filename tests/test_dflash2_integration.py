"""Actual CPU DFlash2 tensors: distinct heads, updates and config/weight reload."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from draftfit.algorithms.common.dflash_family_model import OnlineDFlash2Model
from draftfit.algorithms.taichu_dflash2_model import OnlineTaichuDFlash2Model
from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from draftfit.modeling.draft.taichu_dflash2 import (
    TaichuDFlash2Config,
    TaichuDFlash2DraftModel,
)


def tiny_config(config_class=DFlash2Config, **overrides):
    values = dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        target_num_hidden_layers=8,
        target_layer_ids=[1, 5],
        mask_token_id=63,
        block_size=4,
        conv_group_size=8,
        selector_rank=8,
        selector_top_k=4,
    )
    values.update(overrides)
    return config_class(**values)


class DFlash2IntegrationTest(unittest.TestCase):
    def test_taichu_text_mrope_and_mixed_attention(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,
        )

        from draftfit.export.to_hf import export_to_hf
        from draftfit.export.to_sglang import export_to_sglang
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.training.model_loading import load_draft_config_source

        rope = {
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 1.0,
        }
        config = tiny_config(
            config_class=TaichuDFlash2Config,
            architectures=["TaichuDFlash2DraftModel"],
            layer_types=["sliding_attention", "full_attention"],
            use_sliding_window=True,
            sliding_window=4,
            rope_parameters=rope,
        )
        draft = TaichuDFlash2DraftModel(config)
        self.assertEqual(
            [layer.self_attn.sliding_window for layer in draft.layers], [4, None]
        )
        positions = torch.arange(6).unsqueeze(0)
        hidden = torch.zeros(1, 6, 32)
        reference = Qwen3_5TextRotaryEmbedding(
            Qwen3_5TextConfig(
                hidden_size=32,
                num_attention_heads=4,
                head_dim=8,
                rope_parameters=rope,
            )
        )
        for got, want in zip(
            draft.rotary_emb(hidden, positions), reference(hidden, positions)
        ):
            torch.testing.assert_close(got, want)
        wrapper = OnlineTaichuDFlash2Model(
            draft_model=draft,
            target_lm_head=torch.nn.Linear(32, 64, bias=False),
            target_embed_tokens=torch.nn.Embedding(64, 32),
            mask_token_id=63,
            block_size=4,
            attention_backend="sdpa",
        )
        with mock.patch.object(draft, "forward", wraps=draft.forward) as forward:
            loss, _, _ = wrapper(
                torch.randint(0, 63, (1, 8)),
                torch.randn(1, 8, 64),
                torch.ones(1, 8),
            )
        masks = forward.call_args.kwargs["attention_mask"]
        self.assertTrue((masks["full_attention"] & ~masks["sliding_attention"]).any())
        loss.backward()
        self.assertTrue(
            torch.isfinite(draft.layers[0].self_attn.q_proj.weight.grad).all()
        )
        from draftfit.algorithms.model_providers import build_dflash2_model

        settings = SimpleNamespace(
            training=SimpleNamespace(
                loss_type="dflash",
                dpace_alpha=0.5,
                dflash2_selector_loss_alpha=1.0,
            )
        )
        with mock.patch(
            "draftfit.algorithms.model_providers._build_dflash_family_model"
        ) as build:
            build_dflash2_model(settings, draft, config, None, None)
        selected = build.call_args.args[3]
        self.assertIsInstance(
            selected(
                dict(
                    draft_model=draft,
                    target_lm_head=torch.nn.Linear(32, 64, bias=False),
                    target_embed_tokens=torch.nn.Embedding(64, 32),
                    mask_token_id=63,
                    block_size=4,
                    attention_backend="sdpa",
                )
            ),
            OnlineTaichuDFlash2Model,
        )
        with tempfile.TemporaryDirectory() as directory:
            draft.save_pretrained(directory)
            restored = AutoDraftModel.from_pretrained(
                directory, config=load_draft_config_source(directory)
            )
            self.assertIsInstance(restored, TaichuDFlash2DraftModel)
            checkpoint = Path(directory) / "training_state.pt"
            torch.save(
                {"strategy": "dflash2", "draft_state_dict": draft.state_dict()},
                checkpoint,
            )
            hf_export = Path(directory) / "hf"
            export_to_hf(
                str(checkpoint), str(Path(directory) / "config.json"), str(hf_export)
            )
            automatic = AutoDraftModel.from_pretrained(hf_export)
            self.assertIsInstance(automatic, TaichuDFlash2DraftModel)
            for key, value in draft.state_dict().items():
                actual = automatic.state_dict()[key]
                self.assertTrue(torch.equal(actual, value.to(dtype=actual.dtype)), key)
            serving = Path(directory) / "serving"
            export_to_sglang(
                str(checkpoint), str(Path(directory) / "config.json"), str(serving)
            )
            self.assertEqual(
                json.loads((serving / "config.json").read_text())["architectures"],
                ["DFlash2DraftModel"],
            )
        with self.assertRaises(ValueError):
            tiny_config(rope_parameters=rope)
        with self.assertRaises(ValueError):
            tiny_config(
                config_class=TaichuDFlash2Config,
                rope_parameters={**rope, "partial_rotary_factor": 0.5},
            )

    def test_yarn_rotary_matches_transformers_and_survives_reload(self):
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        rope = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 64.0,
            "original_max_position_embeddings": 128,
        }
        config = tiny_config(rope_parameters=rope)
        draft = DFlash2DraftModel(config)
        reference = Qwen3RotaryEmbedding(
            Qwen3Config(
                hidden_size=32,
                num_attention_heads=4,
                head_dim=8,
                max_position_embeddings=config.max_position_embeddings,
                rope_parameters=rope,
            )
        )
        positions = torch.tensor([[0, 127, 128, 4096]])
        x = torch.zeros(1, 4, 32)
        for actual, expected in zip(
            draft.rotary_emb(x, positions), reference(x, positions)
        ):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(
            DFlash2Config.from_dict(config.to_dict()).rope_parameters["factor"], 64.0
        )

    def test_multiple_batches_blocks_update_selector_and_convolution(self):
        torch.manual_seed(7)
        draft = DFlash2DraftModel(tiny_config())
        wrapper = OnlineDFlash2Model(
            draft_model=draft,
            target_lm_head=torch.nn.Linear(32, 64, bias=False),
            target_embed_tokens=torch.nn.Embedding(64, 32),
            mask_token_id=63,
            block_size=4,
            attention_backend="sdpa",
        )
        watched = [
            draft.candidate_selector.hidden_projection.weight,
            draft.layers[0].attention_conv.kernel_projection.weight,
        ]
        before = [parameter.detach().clone() for parameter in watched]
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
        loss, _, _ = wrapper(
            torch.randint(0, 63, (2, 16)), torch.randn(2, 16, 64), torch.ones(2, 16)
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in watched:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.norm().item(), 0)
        optimizer.step()
        for initial, parameter in zip(before, watched):
            self.assertFalse(torch.equal(initial, parameter))
        with tempfile.TemporaryDirectory() as directory:
            draft.save_pretrained(directory)
            restored = DFlash2DraftModel.from_pretrained(directory)
            self.assertEqual(restored.config.target_layer_ids, [1, 5])
            self.assertEqual(restored.config.target_num_hidden_layers, 8)
            self.assertEqual(set(draft.state_dict()), set(restored.state_dict()))
            for key, value in draft.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_selector_includes_true_successor_outside_top_k(self):
        draft = DFlash2DraftModel(tiny_config())
        logits = torch.zeros(2, 3, 3, 64)
        logits[..., :4] = 10
        truth = torch.full((2, 3, 3), 63, dtype=torch.long)
        scores, candidates = draft.score_candidates(
            torch.randn(2, 3, 3, 32),
            logits,
            torch.zeros_like(truth),
            training_successor_ids=truth,
        )
        self.assertEqual(scores.shape, (2, 3, 3, 4))
        self.assertTrue((candidates == truth.unsqueeze(-1)).any(-1).all().item())

    def test_config_roundtrip_and_incompatible_settings_fail(self):
        config = tiny_config()
        restored = DFlash2Config.from_dict(config.to_dict())
        self.assertEqual(restored.target_num_hidden_layers, 8)
        self.assertEqual(restored.target_layer_ids, [1, 5])
        for options in (
            dict(draft_vocab_size=32),
            dict(selector_top_k=65),
            dict(conv_group_size=7),
            dict(layer_types=["full_attention", "sliding_attention"]),
            dict(input_embedding_scale=float("nan")),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                tiny_config(**options)

    def test_runtime_checkpoint_exports_hf_and_sglang_schema(self):
        from safetensors.torch import load_file

        from draftfit.export.to_hf import export_to_hf
        from draftfit.export.to_sglang import export_to_sglang

        draft = DFlash2DraftModel(tiny_config()).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "draft.json"
            draft.config.to_json_file(config_path)
            checkpoint = root / "training_state.pt"
            torch.save(
                {"strategy": "dflash2", "draft_state_dict": draft.state_dict()},
                checkpoint,
            )
            hf, serving = root / "hf", root / "serving"
            export_to_hf(str(checkpoint), str(config_path), str(hf))
            restored = DFlash2DraftModel.from_pretrained(hf, dtype=torch.bfloat16)
            from draftfit.modeling.auto import AutoDraftModel

            automatic = AutoDraftModel.from_pretrained(hf, dtype=torch.bfloat16)
            self.assertIsInstance(automatic, DFlash2DraftModel)
            for key, value in draft.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
                self.assertTrue(torch.equal(value, automatic.state_dict()[key]), key)
            export_to_sglang(str(checkpoint), str(config_path), str(serving))
            exported = json.loads((serving / "config.json").read_text())
            self.assertEqual(exported["model_type"], "qwen3")
            self.assertEqual(exported["architectures"], ["DFlash2DraftModel"])
            self.assertEqual(exported["dflash_config"]["target_layer_ids"], [1, 5])
            self.assertEqual(exported["num_target_layers"], 8)
            self.assertNotIn("selector_top_k", exported)
            weights = load_file(serving / "model.safetensors")
            for key, value in draft.state_dict_for_serving().items():
                self.assertTrue(torch.equal(value, weights[key]), key)


if __name__ == "__main__":
    unittest.main()
