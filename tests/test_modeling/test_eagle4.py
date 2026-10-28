"""Behavioral tests for candidate selection, recurrent normalization and masking."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from transformers import LlamaConfig

from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4


def tiny_config(**overrides):
    return LlamaConfig(**{
        "architectures": ["LlamaForCausalLMEagle4"],
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8,
        "max_position_embeddings": 64, "rope_theta": 10000.,
        "norm_output": True, "fc_norm": True,
        "sliding_window": 3, "use_sliding_window": True,
        "efficient_lm_head_config": {"type": "low_rank", "rank": 8, "num_candidates": 4},
        **overrides,
    })


class Eagle4Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = LlamaForCausalLMEagle4(tiny_config()).eval()

    def test_candidate_logits_use_exact_selected_full_head_rows(self):
        h = torch.randn(2, 3, 16)
        logits, ids = self.model.candidate_logits(h)
        torch.testing.assert_close(logits, self.model.compute_logits(h).gather(-1, ids))
        self.assertTrue(torch.equal(ids, self.model.candidate_scores(h).topk(4).indices))

    def test_mask_combines_window_padding_and_causality(self):
        mask = self.model.prepare_decoder_attention_mask(
            torch.tensor([[0, 1, 1, 1, 1]]), torch.randn(1, 5, 16), 1, 5, 0
        )[0, 0]
        expected = torch.tensor([[0,0,0,0,0], [0,1,0,0,0], [0,1,1,0,0],
                                 [0,1,1,1,0], [0,0,1,1,1]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask == 0, expected))

    def test_outputs_ignore_future_and_out_of_window_inputs(self):
        h = torch.randn(1, 7, 48)
        emb = torch.randn(1, 7, 16)
        out = self.model(h, emb)
        changed = h.clone(); changed[:, 6] += 100
        torch.testing.assert_close(out[:, :6], self.model(changed, emb)[:, :6])
        changed = h.clone(); changed[:, :3] += 100
        other_emb = emb.clone(); other_emb[:, :3] += 100
        torch.testing.assert_close(out[:, 5:], self.model(changed, other_emb)[:, 5:])

    def test_candidate_stage_only_updates_candidate_factors_and_masks_padding(self):
        self.model.compute_logits(torch.randn(1, 4, 16)).sum().backward()
        self.assertIsNotNone(self.model.lm_head.weight.grad)
        self.model.set_training_stage("candidates")
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        h = torch.randn(1, 4, 16)
        dense = self.model.compute_logits(h)
        mask = torch.tensor([[[1.], [1.], [0.], [0.]]])
        loss = self.model.candidate_loss(h, dense, mask)
        changed = h.clone(); changed[:, 2:] += 100
        torch.testing.assert_close(loss, self.model.candidate_loss(changed, self.model.compute_logits(changed), mask))
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.02)
        for _ in range(20):
            optimizer.zero_grad()
            current = self.model.candidate_loss(h, dense, mask)
            current.backward(); optimizer.step()
        self.assertLess(self.model.candidate_loss(h, dense, mask).item(), loss.item())
        for name, parameter in self.model.named_parameters():
            if not name.startswith("efficient_lm_head"):
                self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_export_reload_preserves_candidates(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            reloaded = AutoDraftModel.from_pretrained(path).eval()
            h = torch.randn(1, 2, 16)
            a, ai = self.model.candidate_logits(h)
            b, bi = reloaded.candidate_logits(h)
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.equal(ai, bi))

    def test_sglang_export_preserves_candidate_factors(self):
        from draftfit.export.to_sglang import export_to_sglang
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.config.save_pretrained(root)
            torch.save({"strategy": "eagle4", "draft_state_dict": self.model.state_dict()},
                       root / "training_state.pt")
            export_to_sglang(str(root), str(root / "config.json"), str(root / "serving"))
            weights = load_file(root / "serving" / "model.safetensors")
            for name in ("efficient_lm_head.weight", "efficient_lm_head_down_proj.weight"):
                torch.testing.assert_close(weights[name], self.model.state_dict()[name].bfloat16())
            self.assertNotIn("embed_tokens.weight", weights)

    def test_full_rank_svd_reconstructs_the_dense_scores(self):
        model = LlamaForCausalLMEagle4(tiny_config(
            efficient_lm_head_config={"type":"low_rank", "rank":16, "num_candidates":4}
        ))
        model.initialize_candidate_head()
        h = torch.randn(1, 3, 16)
        torch.testing.assert_close(model.candidate_scores(h), model.compute_logits(h), atol=1e-6, rtol=1e-5)

    def test_invalid_architectures_are_rejected(self):
        for overrides in ({"num_hidden_layers": 2}, {"draft_vocab_size": 16},
                          {"sliding_window": 0}, {"use_rope": False},
                          {"efficient_lm_head_config": {"type":"low_rank", "rank":True, "num_candidates":4}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                LlamaForCausalLMEagle4(tiny_config(**overrides))

    @unittest.skipUnless(torch.cuda.is_available(), "Registered EAGLE training requires the CUDA/Triton runtime")
    def test_registered_cuda_capture_loader_dense_and_candidate_training(self):
        from transformers import LlamaForCausalLM
        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.config import Config
        from draftfit.offline_capture.transformers import OfflineTransformersCapture
        from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
        from draftfit.runtime.data_plane.feature_store import LocalFeatureStore

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"SPECFORGE_DEVICE":"cuda"}
        ):
            root = Path(directory)
            target_path, features = root / "target", root / "features"
            features.mkdir()
            config = tiny_config(num_hidden_layers=3, layer_types=["full_attention"] * 3)
            target = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
            target.save_pretrained(target_path, max_shard_size="2KB")
            capture = OfflineTransformersCapture(target)
            capture.set_capture_layers([0, 1, 2], capture_method="eagle3")
            registration = builtin_algorithm_registry().resolve("eagle4")
            provider = registration.providers.offline_for("text")
            for index, length in enumerate((8, 11)):
                ids = torch.arange(1, length+1, device="cuda").view(1, -1)
                result = capture.capture(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         loss_mask=torch.ones_like(ids))
                record = provider.capture_layout.materialize(dict(
                    input_ids=ids[0], loss_mask=result.loss_mask[0],
                    aux_hidden_states=result.hidden_states,
                    last_hidden_states=result.last_hidden_states,
                ))
                torch.save(record, features / f"{index}.ckpt")
            run = Config.model_validate(dict(
                model=dict(target_model_path=str(target_path), torch_dtype="bfloat16"),
                data=dict(hidden_states_path=str(features)),
                training=dict(strategy="eagle4", attention_backend="sdpa", ttt_length=2,
                              max_steps=2, batch_size=2),
            ))
            draft = registration.providers.model.build_draft(run, tiny_config())
            parts = registration.providers.model.build_training_model(
                run, draft, draft.config, config, None
            )
            strategy = registration.providers.step.build(parts.model, target_head=parts.target_head)
            reader = provider.build_reader(str(features), run_id="eagle4", ttt_length=2, max_len=12)
            loader = FeatureDataLoader(
                LocalFeatureStore(), refs=reader.read(), batch_size=2, strategy="eagle4",
                device="cpu", gc_interval_s=None,
                per_sample_transform=provider.build_normalizer(max_len=12),
                collate_fn=provider.build_collator(),
            )
            try:
                batch = next(iter(loader))
            finally:
                loader.close()
            self.assertEqual(batch.strategy, "eagle4")
            for stage, watched in (("draft", draft.fc.weight),
                                   ("candidates", draft.efficient_lm_head.weight)):
                draft.set_training_stage(stage)
                optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=.01)
                before = watched.detach().clone()
                output = strategy.forward_loss(batch)
                self.assertTrue(torch.isfinite(output.loss))
                # Reported position losses must include the optimized candidate objective.
                for actual, reported in zip(output.metrics["plosses"], output.metrics["metric_losses"]):
                    torch.testing.assert_close(actual, reported)
                output.loss.backward()
                self.assertGreater(watched.grad.float().norm().item(), 0)
                optimizer.step()
                self.assertFalse(torch.equal(before, watched))
