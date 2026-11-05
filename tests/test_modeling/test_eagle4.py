"""Behavioral tests for candidate selection, recurrent normalization and masking."""

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4, candidate_argmax


def tiny_config(**overrides):
    return LlamaConfig(
        **{
            "architectures": ["LlamaForCausalLMEagle4"],
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "norm_output": True,
            "fc_norm": True,
            "sliding_window": 3,
            "use_sliding_window": True,
            "efficient_lm_head_config": {
                "type": "low_rank",
                "rank": 8,
                "num_candidates": 4,
            },
            **overrides,
        }
    )


class Eagle4Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = LlamaForCausalLMEagle4(tiny_config()).eval()

    def test_candidate_greedy_matches_vocabulary_argmax_for_ties(self):
        scores = torch.tensor([[3.0, 3.0, 1.0], [2.0, 4.0, 4.0]]).bfloat16()
        ids = torch.tensor([[9, 3, 5], [2, 8, 7]])
        dense = torch.full((2, 32), -torch.inf, dtype=scores.dtype).scatter(
            -1, ids, scores
        )
        expected = dense.argmax(-1)
        for order in (torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0])):
            self.assertTrue(
                torch.equal(candidate_argmax(scores[:, order], ids[:, order]), expected)
            )

    def test_native_precision_keeps_checkpoint_keys_and_rejects_cpu_forward(self):
        native = LlamaForCausalLMEagle4(tiny_config(eagle4_precision="vllm"))
        self.assertEqual(set(native.state_dict()), set(self.model.state_dict()))
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            native(torch.randn(1, 3, 48), torch.randn(1, 3, 16))

    @unittest.skipUnless(
        torch.cuda.is_available() and find_spec("vllm") is not None,
        "requires a vLLM CUDA runtime",
    )
    def test_native_rms_norm_gradients_match_full_precision_formula(self):
        from draftfit.modeling.draft.eagle4_precision import Eagle4NativeRMSNorm

        norm = Eagle4NativeRMSNorm(5120).cuda().bfloat16()
        with torch.no_grad():
            norm.weight.uniform_(0.5, 2.0)
        value = torch.randn(
            2, 3, 5120, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad = torch.randn_like(value)
        norm(value).backward(grad)
        x = value.detach().float().requires_grad_()
        w = norm.weight.detach().float().requires_grad_()
        reference = (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)
            * w
        )
        reference.backward(grad.float())
        torch.testing.assert_close(value.grad, x.grad.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(norm.weight.grad, w.grad.bfloat16(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_large_vocabulary_loss_matches_dense_value_and_gradient(self):
        from draftfit.algorithms.eagle3.model import _teacher_cross_entropy

        for size in (131072, 131073, 248320):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(vocabulary=size, dtype=dtype):
                    logits = torch.randn(
                        1, 3, size, device="cuda", dtype=dtype, requires_grad=True
                    )
                    teacher = torch.randn_like(logits).float().softmax(-1)
                    mask = torch.tensor([[[1.0], [0.0], [1.0]]], device="cuda")
                    expected = (
                        -(mask * teacher * logits.float().log_softmax(-1))
                        .sum(-1)
                        .mean()
                    )
                    actual = _teacher_cross_entropy(logits, teacher, mask)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, logits)[0],
                        torch.autograd.grad(expected, logits)[0],
                        rtol=1e-2 if dtype == torch.bfloat16 else 1e-4,
                        atol=1e-6,
                    )

    def test_onebody_stage_clears_stale_gradients_and_trains_both_heads(self):
        self.model.compute_logits(torch.randn(1, 2, 16)).sum().backward()
        self.model.set_training_stage("onebody")
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for name, parameter in self.model.named_parameters():
            self.assertEqual(
                parameter.requires_grad, name != "embed_tokens.weight", name
            )
        hidden = torch.randn(1, 2, 16)
        logits = self.model.compute_logits(hidden)
        loss = logits.square().mean() + self.model.candidate_loss(
            hidden, logits, torch.ones(1, 2, 1)
        )
        loss.backward()
        for parameter in (
            self.model.lm_head.weight,
            self.model.efficient_lm_head.weight,
            self.model.efficient_lm_head_down_proj.weight,
        ):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.model.set_training_stage("draft")
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertFalse(self.model.efficient_lm_head.weight.requires_grad)
        self.model.set_training_stage("candidates")
        self.assertFalse(self.model.lm_head.weight.requires_grad)

    def test_packed_target_embedding_is_rejected_without_changing_weights(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"model.embed_tokens.weight": torch.ones(32, 16, dtype=torch.uint8)},
                str(Path(directory) / "model.safetensors"),
            )
            before = self.model.embed_tokens.weight.detach().clone()
            with self.assertRaisesRegex(
                ValueError, "packed embeddings or output heads"
            ):
                self.model.load_embedding(directory)
            torch.testing.assert_close(
                self.model.embed_tokens.weight, before, atol=0, rtol=0
            )

    def test_freeze_embedding_clears_a_preexisting_gradient(self):
        weight = self.model.embed_tokens.weight
        weight.requires_grad_(True)
        weight.sum().backward()
        optimizer = torch.optim.SGD([weight], lr=0.1)
        before = weight.detach().clone()
        self.model.freeze_embedding()
        optimizer.step()
        self.assertIsNone(weight.grad)
        torch.testing.assert_close(weight, before, atol=0, rtol=0)

    def test_official_pretrained_warm_start_preserves_candidates_without_embedding(
        self,
    ):
        from safetensors.torch import save_file

        from draftfit.training.model_loading import warm_start_draft_model

        with tempfile.TemporaryDirectory() as directory:
            self.model.config.save_pretrained(directory)
            weights = {
                key.replace("midlayer.", "layers.0.", 1): value.contiguous()
                for key, value in self.model.state_dict().items()
                if key != "embed_tokens.weight"
            }
            save_file(weights, str(Path(directory) / "model.safetensors"))
            reloaded = AutoDraftModel.from_config(tiny_config()).eval()
            with patch.object(
                reloaded,
                "initialize_candidate_head",
                side_effect=AssertionError("unexpected SVD"),
            ):
                report = warm_start_draft_model(
                    reloaded,
                    directory,
                    draft_config=reloaded.config,
                    strategy="eagle4",
                    allow_missing_embedding=True,
                )
            self.assertEqual(report.missing_keys, ("embed_tokens.weight",))
            self.assertFalse(report.loaded_embedding)
            for key, expected in self.model.state_dict().items():
                if key != "embed_tokens.weight":
                    torch.testing.assert_close(
                        reloaded.state_dict()[key], expected, atol=0, rtol=0
                    )

    def test_warm_start_rejects_ambiguous_official_and_internal_layer_keys(self):
        from draftfit.training.model_loading import warm_start_draft_model

        with tempfile.TemporaryDirectory() as directory:
            weights = dict(self.model.state_dict())
            weights["layers.0.self_attn.q_proj.weight"] = weights[
                "midlayer.self_attn.q_proj.weight"
            ]
            torch.save(
                {"strategy": "eagle4", "draft_state_dict": weights},
                Path(directory) / "training_state.pt",
            )
            with self.assertRaisesRegex(
                ValueError, "conflicting EAGLE4 checkpoint key"
            ):
                warm_start_draft_model(
                    self.model,
                    directory,
                    draft_config=self.model.config,
                    strategy="eagle4",
                )

    def test_reload_preserves_native_rotary_dtype_policy(self):
        for precision in ("torch", "vllm"):
            with (
                self.subTest(precision=precision),
                tempfile.TemporaryDirectory() as path,
            ):
                model = LlamaForCausalLMEagle4(
                    tiny_config(eagle4_precision=precision)
                ).bfloat16()
                model.save_pretrained(path)
                reloaded = AutoDraftModel.from_pretrained(path, dtype=torch.bfloat16)
                before = dict(model.midlayer.self_attn.rotary_emb.named_buffers())
                after = dict(reloaded.midlayer.self_attn.rotary_emb.named_buffers())
                self.assertEqual(set(before), set(after))
                for key, value in before.items():
                    self.assertEqual(after[key].dtype, value.dtype)
                    torch.testing.assert_close(after[key], value, atol=0, rtol=0)
                self.assertTrue(reloaded.t2d.all())
                self.assertFalse(reloaded.d2t.any())

    def test_official_layers_warm_start_preserves_all_weights(self):
        from draftfit.training.model_loading import warm_start_draft_model

        with tempfile.TemporaryDirectory() as directory:
            self.model.config.save_pretrained(directory)
            torch.save(
                {
                    "strategy": "eagle4",
                    "draft_state_dict": {
                        key.replace("midlayer.", "layers.0.", 1): value
                        for key, value in self.model.state_dict().items()
                    },
                },
                Path(directory) / "training_state.pt",
            )
            reloaded = AutoDraftModel.from_config(tiny_config()).eval()
            with patch.object(
                reloaded,
                "initialize_candidate_head",
                side_effect=AssertionError("unexpected SVD"),
            ):
                report = warm_start_draft_model(
                    reloaded, directory, draft_config=reloaded.config, strategy="eagle4"
                )
            self.assertFalse(report.missing_keys)
            for key, expected in self.model.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[key], expected, atol=0, rtol=0
                )
            hidden, embedding = torch.randn(1, 4, 48), torch.randn(1, 4, 16)
            torch.testing.assert_close(
                reloaded(hidden, embedding),
                self.model(hidden, embedding),
                atol=0,
                rtol=0,
            )

    def test_candidate_logits_use_exact_selected_full_head_rows(self):
        h = torch.randn(2, 3, 16)
        logits, ids = self.model.candidate_logits(h)
        torch.testing.assert_close(logits, self.model.compute_logits(h).gather(-1, ids))
        self.assertTrue(
            torch.equal(ids, self.model.candidate_scores(h).topk(4).indices)
        )

    def test_bfloat16_candidate_rescoring_accumulates_products_in_float32(self):
        model = self.model.bfloat16()
        hidden = torch.randn(2, 3, 16).bfloat16()
        actual, ids = model.candidate_logits(hidden)
        expected = (
            torch.nn.functional.linear(
                model.head_features(hidden).float(), model.lm_head.weight.float()
            )
            .bfloat16()
            .gather(-1, ids)
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_mask_combines_window_padding_and_causality(self):
        mask = self.model.prepare_decoder_attention_mask(
            torch.tensor([[0, 1, 1, 1, 1]]), torch.randn(1, 5, 16), 1, 5, 0
        )[0, 0]
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask == 0, expected))

    def test_outputs_ignore_future_and_out_of_window_inputs(self):
        h = torch.randn(1, 7, 48)
        emb = torch.randn(1, 7, 16)
        out = self.model(h, emb)
        changed = h.clone()
        changed[:, 6] += 100
        torch.testing.assert_close(out[:, :6], self.model(changed, emb)[:, :6])
        changed = h.clone()
        changed[:, :3] += 100
        other_emb = emb.clone()
        other_emb[:, :3] += 100
        torch.testing.assert_close(out[:, 5:], self.model(changed, other_emb)[:, 5:])

    def test_candidate_stage_only_updates_candidate_factors_and_masks_padding(self):
        self.model.compute_logits(torch.randn(1, 4, 16)).sum().backward()
        self.assertIsNotNone(self.model.lm_head.weight.grad)
        self.model.set_training_stage("candidates")
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        h = torch.randn(1, 4, 16)
        dense = self.model.compute_logits(h)
        mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
        loss = self.model.candidate_loss(h, dense, mask)
        changed = h.clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(
            loss,
            self.model.candidate_loss(
                changed, self.model.compute_logits(changed), mask
            ),
        )
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=0.02
        )
        for _ in range(20):
            optimizer.zero_grad()
            current = self.model.candidate_loss(h, dense, mask)
            current.backward()
            optimizer.step()
        self.assertLess(self.model.candidate_loss(h, dense, mask).item(), loss.item())
        for name, parameter in self.model.named_parameters():
            if not name.startswith("efficient_lm_head"):
                self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_export_reload_preserves_candidates(self):
        with tempfile.TemporaryDirectory() as path:
            self.model.save_pretrained(path)
            from safetensors.torch import load_file

            weights = load_file(Path(path) / "model.safetensors")
            self.assertNotIn("t2d", weights)
            self.assertNotIn("d2t", weights)
            reloaded = AutoDraftModel.from_pretrained(path).eval()
            h = torch.randn(1, 2, 16)
            a, ai = self.model.candidate_logits(h)
            b, bi = reloaded.candidate_logits(h)
            torch.testing.assert_close(a, b)
            self.assertTrue(torch.equal(ai, bi))

            hidden = torch.randn(1, 3, 48)
            embedding = torch.randn(1, 3, 16)
            torch.testing.assert_close(
                self.model(hidden, embedding),
                reloaded(hidden, embedding),
                atol=0,
                rtol=0,
            )
            self.assertTrue(reloaded.t2d.all())
            self.assertFalse(reloaded.d2t.any())

    def test_official_layer_names_load_without_random_backbone_weights(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as path:
            self.model.config.save_pretrained(path)
            weights = {
                key.replace("midlayer.", "layers.0.", 1): value.contiguous()
                for key, value in self.model.state_dict().items()
            }
            save_file(weights, str(Path(path) / "model.safetensors"))
            reloaded, info = AutoDraftModel.from_pretrained(
                path, output_loading_info=True
            )
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            ):
                self.assertFalse(info.get(key), (key, info.get(key)))
            for key, expected in self.model.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[key], expected, atol=0, rtol=0
                )
            hidden = torch.randn(1, 3, 48)
            embedding = torch.randn(1, 3, 16)
            torch.testing.assert_close(
                self.model(hidden, embedding),
                reloaded.eval()(hidden, embedding),
                atol=0,
                rtol=0,
            )

    def test_bfloat16_reload_preserves_three_recurrent_outputs(self):
        model = self.model.bfloat16()
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            reloaded = AutoDraftModel.from_pretrained(path, dtype=torch.bfloat16).eval()
            ids = torch.tensor([[1, 2, 3]])
            hidden = torch.randn(1, 3, 48).bfloat16()
            expected = model.project_hidden_states(hidden)
            actual = reloaded.project_hidden_states(hidden)
            mask = model.prepare_decoder_attention_mask(
                torch.ones_like(ids), expected, 1, 3, 0
            )
            expected_cache, actual_cache = [[], []], [[], []]
            for _ in range(3):
                kwargs = {
                    "attention_mask": mask,
                    "position_ids": torch.arange(3)[None],
                }
                expected = model.backbone(
                    input_embeds=model.embed_input_ids(ids),
                    hidden_states=expected,
                    cache_hidden=expected_cache,
                    **kwargs,
                )
                actual = reloaded.backbone(
                    input_embeds=reloaded.embed_input_ids(ids),
                    hidden_states=actual,
                    cache_hidden=actual_cache,
                    **kwargs,
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_sglang_export_preserves_candidate_factors(self):
        from safetensors.torch import load_file

        from draftfit.export.to_sglang import export_to_sglang

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.config.save_pretrained(root)
            torch.save(
                {"strategy": "eagle4", "draft_state_dict": self.model.state_dict()},
                root / "training_state.pt",
            )
            export_to_sglang(
                str(root), str(root / "config.json"), str(root / "serving")
            )
            weights = load_file(root / "serving" / "model.safetensors")
            for name in (
                "efficient_lm_head.weight",
                "efficient_lm_head_down_proj.weight",
            ):
                torch.testing.assert_close(
                    weights[name], self.model.state_dict()[name].bfloat16()
                )
            self.assertNotIn("embed_tokens.weight", weights)

    def test_full_rank_svd_reconstructs_the_dense_scores(self):
        model = LlamaForCausalLMEagle4(
            tiny_config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 16,
                    "num_candidates": 4,
                }
            )
        )
        model.initialize_candidate_head()
        h = torch.randn(1, 3, 16)
        torch.testing.assert_close(
            model.candidate_scores(h), model.compute_logits(h), atol=1e-6, rtol=1e-5
        )

    def test_invalid_architectures_are_rejected(self):
        for overrides in (
            {"num_hidden_layers": 2},
            {"draft_vocab_size": 16},
            {"sliding_window": 0},
            {"use_rope": False},
            {"eagle4_precision": "unsupported"},
            {"eagle4_precision": "vllm", "hidden_act": "relu"},
            {"eagle4_precision": "vllm", "pretraining_tp": 2},
            {
                "efficient_lm_head_config": {
                    "type": "low_rank",
                    "rank": True,
                    "num_candidates": 4,
                }
            },
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                LlamaForCausalLMEagle4(tiny_config(**overrides))

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "Registered EAGLE training requires the CUDA/Triton runtime",
    )
    def test_registered_cuda_capture_loader_dense_and_candidate_training(self):
        from transformers import LlamaForCausalLM

        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.config import Config
        from draftfit.offline_capture.transformers import OfflineTransformersCapture
        from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
        from draftfit.runtime.data_plane.feature_store import LocalFeatureStore

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"SPECFORGE_DEVICE": "cuda"}),
        ):
            root = Path(directory)
            target_path, features = root / "target", root / "features"
            features.mkdir()
            config = tiny_config(
                num_hidden_layers=3, layer_types=["full_attention"] * 3
            )
            target = (
                LlamaForCausalLM(config)
                .to(device="cuda", dtype=torch.bfloat16)
                .eval()
                .requires_grad_(False)
            )
            target.save_pretrained(target_path, max_shard_size="2KB")
            capture = OfflineTransformersCapture(target)
            capture.set_capture_layers([0, 1, 2], capture_method="eagle3")
            registration = builtin_algorithm_registry().resolve("eagle4")
            provider = registration.providers.offline_for("text")
            for index, length in enumerate((8, 11)):
                ids = torch.arange(1, length + 1, device="cuda").view(1, -1)
                result = capture.capture(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    loss_mask=torch.ones_like(ids),
                )
                record = provider.capture_layout.materialize(
                    dict(
                        input_ids=ids[0],
                        loss_mask=result.loss_mask[0],
                        aux_hidden_states=result.hidden_states,
                        last_hidden_states=result.last_hidden_states,
                    )
                )
                torch.save(record, features / f"{index}.ckpt")
            run = Config.model_validate(
                dict(
                    model=dict(
                        target_model_path=str(target_path), torch_dtype="bfloat16"
                    ),
                    data=dict(hidden_states_path=str(features)),
                    training=dict(
                        strategy="eagle4",
                        attention_backend="sdpa",
                        ttt_length=2,
                        max_steps=2,
                        batch_size=2,
                    ),
                )
            )
            draft = registration.providers.model.build_draft(run, tiny_config())
            parts = registration.providers.model.build_training_model(
                run, draft, draft.config, config, None
            )
            strategy = registration.providers.step.build(
                parts.model, target_head=parts.target_head
            )
            reader = provider.build_reader(
                str(features), run_id="eagle4", ttt_length=2, max_len=12
            )
            loader = FeatureDataLoader(
                LocalFeatureStore(),
                refs=reader.read(),
                batch_size=2,
                strategy="eagle4",
                device="cpu",
                gc_interval_s=None,
                per_sample_transform=provider.build_normalizer(max_len=12),
                collate_fn=provider.build_collator(),
            )
            try:
                batch = next(iter(loader))
            finally:
                loader.close()
            self.assertEqual(batch.strategy, "eagle4")
            for stage, watched in (
                ("draft", draft.fc.weight),
                ("candidates", draft.efficient_lm_head.weight),
            ):
                draft.set_training_stage(stage)
                optimizer = torch.optim.AdamW(
                    [p for p in draft.parameters() if p.requires_grad], lr=0.01
                )
                before = watched.detach().clone()
                output = strategy.forward_loss(batch)
                self.assertTrue(torch.isfinite(output.loss))
                # Reported position losses must include the optimized candidate objective.
                for actual, reported in zip(
                    output.metrics["plosses"], output.metrics["metric_losses"]
                ):
                    torch.testing.assert_close(actual, reported)
                output.loss.backward()
                self.assertGreater(watched.grad.float().norm().item(), 0)
                optimizer.step()
                self.assertFalse(torch.equal(before, watched))
