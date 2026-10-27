"""Regressions for Osprey source attention and CAST sampling/stopping boundaries."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from draftfit.modeling.auto import AutoDraftModel
from draftfit.research.cast.decode import (
    ar_generate,
    cast_generate,
    pick,
    spec_sample_node,
    target_probs,
    walk_tree,
)
from draftfit.research.cast.tree import TreeNode
from draftfit.research.osprey import convert
from tests.test_research_methods import draft_model, target_model


def file_hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def source_with_attention(**attention):
    config = target_model().config.to_dict()
    config.update(attention)
    return Qwen3ForCausalLM(Qwen3Config.from_dict(config)).eval()


class OspreyAttentionRegressionTest(unittest.TestCase):
    def test_sliding_source_is_rejected_before_output_mutation(self):
        for layer_types in (
            ["sliding_attention"] * 3,
            ["full_attention", "sliding_attention", "full_attention"],
        ):
            with self.subTest(layer_types=layer_types), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = source_with_attention(
                    use_sliding_window=True, sliding_window=2, layer_types=layer_types,
                )
                source.save_pretrained(root / "source")
                before = file_hashes(root)
                with self.assertRaisesRegex(ValueError, "full.attention"):
                    convert(str(root / "source"), str(root / "new" / "draft"), 3, 16, [0, 1, 2])
                self.assertFalse((root / "new").exists())
                self.assertEqual(file_hashes(root), before)

    def test_legacy_sliding_source_is_rejected_before_output_mutation(self):
        source = source_with_attention(
            use_sliding_window=True, sliding_window=2, max_window_layers=0, layer_types=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.save_pretrained(root / "source")
            path = root / "source" / "config.json"
            config = json.loads(path.read_text())
            config.pop("layer_types")
            path.write_text(json.dumps(config))
            before = file_hashes(root)
            with self.assertRaisesRegex(ValueError, "full.attention"):
                convert(str(root / "source"), str(root / "new" / "draft"), 3, 16, [0, 1, 2])
            self.assertFalse((root / "new").exists())
            self.assertEqual(file_hashes(root), before)

    def test_full_attention_with_window_metadata_preserves_logits(self):
        for attention, legacy in (
            ({"use_sliding_window": True, "sliding_window": 2,
              "layer_types": ["full_attention"] * 3}, False),
            ({"use_sliding_window": False, "sliding_window": 2}, True),
            ({"use_sliding_window": True, "sliding_window": 2,
              "max_window_layers": 3, "layer_types": None}, True),
        ):
            with self.subTest(attention=attention), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = source_with_attention(**attention)
                source.save_pretrained(root / "source")
                if legacy:
                    path = root / "source" / "config.json"
                    config = json.loads(path.read_text())
                    config.pop("layer_types")
                    config["sliding_window"] = 2
                    path.write_text(json.dumps(config))
                before = file_hashes(root / "source")
                convert(str(root / "source"), str(root / "draft"), 3, 16, [0, 1, 2])
                draft = AutoDraftModel.from_pretrained(root / "draft").eval()
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                features = torch.randn(1, ids.shape[1], 48)
                with torch.no_grad():
                    mask = draft.prepare_decoder_attention_mask(
                        torch.ones_like(ids), features, 1, ids.shape[1], 0,
                    )
                    hidden = draft.backbone(
                        input_embeds=draft.embed_input_ids(ids),
                        hidden_states=draft.project_hidden_states(features), cache_hidden=None,
                        attention_mask=mask, position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
                        use_cache=False,
                    )
                    torch.testing.assert_close(
                        draft.compute_logits(hidden), source(ids, use_cache=False).logits,
                        atol=1e-6, rtol=1e-5,
                    )
                self.assertEqual(file_hashes(root / "source"), before)


class CastBoundaryRegressionTest(unittest.TestCase):
    def test_small_positive_temperature_keeps_sampling_distribution(self):
        logits = torch.zeros(32)
        probabilities = torch.full_like(logits, 1 / logits.numel())
        for temperature in (1e-6, 1e-5):
            with self.subTest(temperature=temperature, helper="target_probs"):
                torch.testing.assert_close(target_probs(logits, temperature), probabilities)
            with self.subTest(temperature=temperature, helper="pick"):
                expected = int(torch.multinomial(
                    probabilities, 1, generator=torch.Generator().manual_seed(0),
                ))
                actual = pick(logits, temperature, torch.Generator().manual_seed(0))
                self.assertEqual(actual, expected)
                self.assertNotEqual(actual, 0)

    def test_small_positive_temperature_walk_uses_speculative_sampling(self):
        probabilities = torch.full((3,), 1 / 3)
        nodes = [TreeNode(token=0, depth=1, parent=-1, score=0.0)]
        expected_generator = torch.Generator().manual_seed(0)
        token, child = spec_sample_node(
            probabilities, torch.tensor([0]), probabilities, expected_generator,
        )
        self.assertIsNone(child)
        actual = walk_tree(
            nodes, torch.zeros(2, 3), temperature=1e-6, d_logits=torch.zeros(1, 3),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(actual, ([], token))

    def test_small_positive_temperature_samples_real_target_prefill(self):
        target = target_model()
        draft = draft_model(4).eval()
        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            target.lm_head.weight.zero_()
        for decoder in (ar_generate, cast_generate):
            for seed in range(4):
                with self.subTest(decoder=decoder.__name__, seed=seed):
                    expected = int(torch.multinomial(
                        torch.full((32,), 1 / 32), 1,
                        generator=torch.Generator().manual_seed(seed),
                    ))
                    args = (draft, target) if decoder is cast_generate else (target,)
                    actual = decoder(
                        *args, prompt, 1, [], temperature=1e-6,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    self.assertEqual(int(actual[0, -1]), expected)

    def test_zero_temperature_remains_greedy_without_consuming_rng(self):
        generator = torch.Generator().manual_seed(0)
        before = generator.get_state().clone()
        logits = torch.zeros(3)
        self.assertEqual(pick(logits, 0.0, generator), 0)
        torch.testing.assert_close(target_probs(logits, 0.0), torch.tensor([1., 0., 0.]))
        nodes = [TreeNode(token=0, depth=1, parent=-1, score=0.0)]
        self.assertEqual(walk_tree(nodes, logits.repeat(2, 1), generator=generator), ([0], 0))
        self.assertTrue(torch.equal(generator.get_state(), before))

    def test_bonus_eos_stops_without_an_extra_round(self):
        target = target_model()
        draft = draft_model(4).eval()
        prompt = torch.tensor([[1, 2, 3]])
        expected = ar_generate(target, prompt, 2, [])
        stop = int(expected[0, -1])
        self.assertNotEqual(stop, int(expected[0, -2]))
        target_calls, draft_calls = [], []
        handles = [
            target.register_forward_pre_hook(lambda _module, args: target_calls.append(args[0].clone())),
            draft.register_forward_pre_hook(lambda _module, _args: draft_calls.append(True)),
        ]
        try:
            actual, stats = cast_generate(
                draft, target, prompt, 6, [stop], budget=1, branch_k=1, return_stats=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(torch.equal(actual, expected), (actual, expected))
        self.assertNotIn(stop, target_calls[1][0, 1:].tolist())  # EOS was the bonus, not a tree child.
        self.assertEqual(stats["rounds"], 1)
        self.assertEqual(stats["n_out"], 2)
        self.assertEqual(len(target_calls), 2)  # Prefill and the first verification only.
        self.assertEqual(len(draft_calls), 1)


if __name__ == "__main__":
    unittest.main()
