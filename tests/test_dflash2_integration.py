"""Actual CPU DFlash2 tensors: distinct heads, updates and config/weight reload."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlash2Model
from specforge.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel


def tiny_config(**overrides):
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
    return DFlash2Config(**values)


class DFlash2IntegrationTest(unittest.TestCase):
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
        selector = DFlash2DraftModel(tiny_config()).candidate_selector
        logits = torch.zeros(2, 3, 3, 64)
        logits[..., :4] = 10
        truth = torch.full((2, 3, 3), 63, dtype=torch.long)
        scores, candidates = selector.score_candidates(
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

        from specforge.export.to_hf import export_to_hf
        from specforge.export.to_sglang import export_to_sglang

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
            for key, value in draft.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
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
