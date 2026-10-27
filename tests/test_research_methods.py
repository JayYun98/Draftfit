"""Native research paths with real tiny CPU models and independent correctness gates."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.modeling.draft.dflash import DFlashDraftModel


def target_model():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def draft_model(block=6):
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=3,
        block_size=block,
        dflash_config={"target_layer_ids": [0, 2], "mask_token_id": 31},
    )
    config._attn_implementation = "sdpa"
    return DFlashDraftModel(config)


class ResearchIntegrationsTest(unittest.TestCase):
    def test_cast_tree_mask_and_real_target_greedy_parity(self):
        from draftfit.research.cast.decode import cast_generate
        from draftfit.research.cast.tree import ancestor_matrix, block_topk, build_tree

        ids, lp = block_topk([[4.0, 3.0, 2.0], [2.0, 3.0, 4.0]], 2)
        nodes = build_tree(ids, lp, 6, 2)
        mask = ancestor_matrix(nodes)
        for index, node in enumerate(nodes):
            self.assertLess(node.parent, index)
            self.assertTrue(mask[index + 1, node.parent + 1])
        target = target_model()
        draft = draft_model(4).eval()
        prompt = torch.tensor([[1, 2, 3]])
        for budget in (0, 1, 7):
            expected = target.generate(
                prompt,
                max_new_tokens=max(1, budget),
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
            if budget == 0:
                expected = prompt
            actual = cast_generate(
                draft, target, prompt, budget, [], budget=7, branch_k=2
            )
            self.assertTrue(torch.equal(actual, expected), (actual, expected))
            self.assertLessEqual(actual.shape[1] - prompt.shape[1], budget)
        from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel

        second = DFlash2DraftModel(
            DFlash2Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=32,
                target_num_hidden_layers=3,
                target_layer_ids=[0, 2],
                mask_token_id=31,
                block_size=4,
                conv_group_size=8,
                selector_rank=4,
                selector_top_k=4,
            )
        ).eval()
        expected = target.generate(
            prompt, max_new_tokens=7, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        actual = cast_generate(second, target, prompt, 7, [], budget=7, branch_k=2)
        self.assertTrue(torch.equal(actual, expected))

    def test_osprey_source_preservation_and_ttt_learning(self):
        from draftfit.algorithms.eagle3.model import OnlineEagle3Model
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.research.osprey import convert
        from draftfit.training.model_loading import load_draft_config_source

        source = target_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.save_pretrained(root / "source")
            convert(str(root / "source"), str(root / "draft"), 3, 16, [0, 1, 2])
            draft = AutoDraftModel.from_pretrained(
                root / "draft", config=load_draft_config_source(str(root / "draft"))
            )
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
            features = torch.randn(1, 6, 48)
            mask = draft.prepare_decoder_attention_mask(
                torch.ones_like(ids), features, 1, 6, 0
            )
            hidden = draft.backbone(
                input_embeds=draft.embed_input_ids(ids),
                hidden_states=draft.project_hidden_states(features),
                cache_hidden=None,
                attention_mask=mask,
                position_ids=torch.arange(6).unsqueeze(0),
                use_cache=False,
            )
            torch.testing.assert_close(
                draft.compute_logits(hidden), source(ids).logits, atol=1e-6, rtol=1e-5
            )
            wrapper = OnlineEagle3Model(draft, length=2, attention_backend="sdpa")
            losses, *_ = wrapper(
                ids,
                torch.ones_like(ids),
                source(ids).logits.detach(),
                torch.ones_like(ids, dtype=torch.float32).unsqueeze(-1),
                features,
            )
            sum(losses).backward()
            self.assertGreater(
                draft.layers[0].self_attn.q_proj.weight.grad[:, :16].abs().sum().item(),
                0,
            )
            draft.freeze_embedding()
            step = (
                builtin_algorithm_registry()
                .resolve("osprey")
                .providers.step.build(wrapper)
            )
            self.assertIn(
                "embed_tokens.weight",
                step.checkpoint_state_filter(wrapper.state_dict()),
            )

    def test_cast_sampling_keeps_target_distribution(self):
        from draftfit.research.cast.decode import spec_sample_node

        target = torch.tensor([0.6, 0.3, 0.1])
        proposal = torch.tensor([0.2, 0.4, 0.4])
        generator = torch.Generator().manual_seed(11)
        counts = torch.zeros(3)
        for _ in range(2000):
            token, _ = spec_sample_node(
                target, torch.tensor([0, 1]), proposal, generator
            )
            counts[token] += 1
        torch.testing.assert_close(counts / counts.sum(), target, atol=0.04, rtol=0)

    def test_osprey_llama_source_preservation_including_llama3_rope(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.research.osprey import convert
        from draftfit.training.model_loading import load_draft_config_source

        ropes = [
            dict(rope_type="default", rope_theta=10000.0),
            dict(
                rope_type="llama3",
                rope_theta=10000.0,
                factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=16,
            ),
        ]
        for rope in ropes:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = LlamaForCausalLM(
                    LlamaConfig(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        vocab_size=32,
                        max_position_embeddings=64,
                        rope_parameters=rope,
                    )
                ).eval()
                source.save_pretrained(root / "source")
                convert(str(root / "source"), str(root / "draft"), 1, 16, [0, 1, 2])
                draft = AutoDraftModel.from_pretrained(
                    root / "draft", config=load_draft_config_source(str(root / "draft"))
                )
                ids = torch.tensor([[1, 2, 3, 4]])
                features = torch.randn(1, 4, 48)
                mask = draft.prepare_decoder_attention_mask(
                    torch.ones_like(ids), features, 1, 4, 0
                )
                hidden = draft.backbone(
                    input_embeds=draft.embed_input_ids(ids),
                    hidden_states=draft.project_hidden_states(features),
                    cache_hidden=None,
                    attention_mask=mask,
                    position_ids=torch.arange(4).unsqueeze(0),
                    use_cache=False,
                )
                torch.testing.assert_close(
                    draft.compute_logits(hidden),
                    source(ids).logits,
                    atol=1e-6,
                    rtol=1e-5,
                )

    def test_alr_rollout_cache_ira_reuse_and_real_loader_update(self):
        from draftfit.algorithms.alr_ira.model import ALRTrainingModel
        from draftfit.algorithms.alr_ira.prepare import prepare
        from draftfit.research.config import ALRConfig
        from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
        from draftfit.runtime.data_plane.feature_store import LocalFeatureStore

        target = target_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corpus.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "reviewed",
                        "input_ids": [1, 2, 3, 4, 5, 6],
                        "loss_mask": [0, 0, 1, 1, 1, 1],
                    }
                )
                + "\n"
            )
            original = source.read_bytes()
            cfg = ALRConfig.model_validate(
                {
                    "method": "alr_ira",
                    "model": {"target_model_path": "tiny"},
                    "data": {"input_path": str(source)},
                    "training": {
                        "block_size": 6,
                        "target_layer_ids": [0, 2],
                        "anchors_per_sample": 2,
                    },
                    "output_dir": str(root / "features"),
                }
            )
            receipt = prepare(cfg, target=target)
            self.assertEqual(receipt["blocks"], receipt["rollouts"] * 2)
            self.assertEqual(receipt["forward_calls"], receipt["rollouts"] * 6)
            self.assertEqual(source.read_bytes(), original)
            provider = (
                builtin_algorithm_registry().resolve("alr_ira").providers.offline[0]
            )
            reader = provider.build_reader(
                str(root / "features"), run_id="alr", ttt_length=1, max_len=64
            )
            loader = FeatureDataLoader(
                LocalFeatureStore(),
                refs=reader.read(),
                batch_size=2,
                strategy="alr_ira",
                device="cpu",
                gc_interval_s=None,
                per_sample_transform=provider.build_normalizer(max_len=64),
                collate_fn=provider.build_collator(),
            )
            try:
                batch = next(iter(loader))
            finally:
                loader.close()
            draft = draft_model()
            wrapper = ALRTrainingModel(
                draft_model=draft,
                target_lm_head=target.lm_head,
                target_embed_tokens=target.get_input_embeddings(),
                mask_token_id=31,
                block_size=6,
                attention_backend="sdpa",
            )
            strategy = (
                builtin_algorithm_registry()
                .resolve("alr_ira")
                .providers.step.build(wrapper)
            )
            optimizer = torch.optim.AdamW(draft.parameters(), lr=0.01)
            before = draft.fc.weight.detach().clone()
            result = strategy.forward_loss(batch)
            self.assertTrue(torch.isfinite(result.loss))
            changed = {key: value.clone() for key, value in batch.tensors.items()}
            for row, anchor in enumerate(changed["alr_anchor"].flatten().tolist()):
                changed["hidden_states"][row, anchor:] += 1000
            no_future_leak, _, _ = wrapper(**changed)
            torch.testing.assert_close(result.loss, no_future_leak)
            result.loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, draft.fc.weight))


class ResearchBoundaryTest(unittest.TestCase):
    def test_osprey_tokenizer_alignment_preserves_rows_and_target_special_ids(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.research.osprey import convert
        from draftfit.research.tokenizer_alignment import align, load_state

        def save_tokenizer(path, vocab, **special):
            tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="[UNK]", **special
            ).save_pretrained(path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = target_model()
            source.config.bos_token_id = 6
            source.save_pretrained(root / "source")
            save_tokenizer(
                root / "source",
                {
                    "[UNK]": 0,
                    "[PAD]": 1,
                    "hello": 2,
                    "world": 3,
                    "two": 4,
                    "words": 5,
                    "[BOS]": 6,
                },
                pad_token="[PAD]",
                bos_token="[BOS]",
            )
            target_config = target_model().config
            target_config.vocab_size = 40
            target_config.bos_token_id = None
            target_config.eos_token_id = 9
            target_config.pad_token_id = 1
            target_config.save_pretrained(root / "target")
            save_tokenizer(
                root / "target",
                {
                    "[UNK]": 0,
                    "[PAD]": 1,
                    "[EOS]": 2,
                    "hello": 3,
                    "world": 4,
                    "two words": 5,
                },
                pad_token="[PAD]",
                eos_token="[EOS]",
            )
            convert(str(root / "source"), str(root / "draft"), 3, 16, [0, 1, 2])
            original = load_state(root / "draft")
            stats = align(root / "draft", str(root / "target"), root / "aligned")
            self.assertGreater(stats["direct"], 0)
            self.assertGreaterEqual(stats["averaged"], 1)
            aligned = AutoDraftModel.from_pretrained(root / "aligned")
            self.assertEqual(aligned.embed_tokens.weight.shape[0], 40)
            torch.testing.assert_close(
                aligned.embed_tokens.weight[3], original["embed_tokens.weight"][2]
            )
            torch.testing.assert_close(
                aligned.lm_head.weight[5], original["lm_head.weight"][[4, 5]].mean(0)
            )
            torch.testing.assert_close(aligned.fc.weight, original["fc.weight"])
            config = json.loads((root / "aligned/config.json").read_text())
            generation = json.loads(
                (root / "aligned/generation_config.json").read_text()
            )
            for record in (config, generation):
                self.assertNotIn("bos_token_id", record)
                self.assertEqual(record["eos_token_id"], 9)
                self.assertEqual(record["pad_token_id"], 1)

    def test_eagle_cpu_loss_matches_dense_value_and_gradient(self):
        from draftfit.algorithms.eagle3.model import _teacher_cross_entropy

        logits = torch.randn(2, 4, 7, requires_grad=True)
        teacher = torch.randn(2, 4, 7).softmax(-1)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]]).unsqueeze(-1)
        expected = -(mask * teacher * logits.log_softmax(-1)).sum(-1).mean()
        actual = _teacher_cross_entropy(logits, teacher, mask)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, logits)[0],
        )

    def test_sparse_kd_matches_dense_ce_and_predecessor_alignment(self):
        from draftfit.algorithms.alr_ira.loss import harvest_target_topk, targetkd_loss

        final = torch.randn(2, 5, 4)
        anchors = torch.tensor([[1], [2]])
        head = torch.nn.Linear(4, 6, bias=False)
        index, prob, tail = harvest_target_topk(
            final, anchors, 3, head, 6, chunk_size=2
        )
        rows = torch.stack([final[0, :3], final[1, 1:4]]).flatten(0, 1)
        expected_prob = head(rows).softmax(-1)
        actual_prob = torch.zeros_like(expected_prob).scatter(1, index, prob)
        torch.testing.assert_close(actual_prob, expected_prob)
        logits = torch.randn(6, 6, requires_grad=True)
        weight = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        expected = -(expected_prob.detach() * logits.log_softmax(-1)).sum(-1)
        expected = (expected * weight).sum() / (weight.sum() + 1e-6)
        loss = targetkd_loss(logits, index, prob, tail, weight)
        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=1e-6)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(gradient[weight == 0].abs().sum().item(), 0)

    def test_ira_secondary_reuses_actual_features_without_target_forwards(self):
        from draftfit.algorithms.alr_ira.prepare import prepare
        from draftfit.research.config import ALRConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]})
                + "\n"
            )
            config = ALRConfig.model_validate(
                {
                    "method": "alr_ira",
                    "model": {"target_model_path": "tiny"},
                    "data": {"input_path": str(corpus)},
                    "training": {
                        "block_size": 6,
                        "target_layer_ids": [0, 2],
                        "anchors_per_sample": 1,
                    },
                    "output_dir": str(root / "features"),
                }
            )
            target = target_model()
            calls = []
            handle = target.register_forward_hook(lambda *args: calls.append(1))
            try:
                result = prepare(config, target=target)
            finally:
                handle.remove()
            self.assertEqual(len(calls), 6)
            self.assertEqual(result["secondary_blocks"], 1)
            primary = torch.load(root / "features/00000000.ckpt", weights_only=True)
            secondary = torch.load(root / "features/00000001.ckpt", weights_only=True)
            for key in ("input_ids", "hidden_states", "target_last_hidden_states"):
                torch.testing.assert_close(primary[key], secondary[key], atol=0, rtol=0)
            self.assertGreater(
                secondary["alr_anchor"].item(), primary["alr_anchor"].item()
            )

    def test_cast_small_cap_eos_and_width_larger_than_vocab(self):
        from draftfit.research.cast.decode import ar_generate, cast_generate

        target = target_model()
        draft = draft_model(4).eval()
        prompt = torch.tensor([[1, 2, 3]])
        first = int(target(prompt).logits[0, -1].argmax())
        for generate, kwargs in ((cast_generate, {"draft": draft}), (ar_generate, {})):
            with self.subTest(generate=generate.__name__):
                torch.testing.assert_close(
                    generate(
                        target=target,
                        input_ids=prompt,
                        max_new_tokens=0,
                        stop_token_ids=[],
                        **kwargs,
                    ),
                    prompt,
                )
                output = generate(
                    target=target,
                    input_ids=prompt,
                    max_new_tokens=5,
                    stop_token_ids=[first],
                    **kwargs,
                )
                self.assertEqual(output[0, 3:].tolist(), [first])
        output = cast_generate(draft, target, prompt, 5, [], branch_k=64, budget=7)
        expected = target.generate(
            prompt, max_new_tokens=5, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
