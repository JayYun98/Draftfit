"""CPU behavior checks with a fake native context; no native/GPU correctness claim."""

import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from draftfit.inference.capture import TeacherCaptureBatch
from draftfit.modeling.draft.dflash import DFlashDraftModel
from draftfit.modeling.target.gguf_parts import GGUFEmbedding, GGUFHead
from draftfit.offline_capture.prism_gguf import OfflinePrismGGUFCapture
from draftfit.training import joint_verification as verification
from tests.test_dflash2_integration import tiny_config


class MockNativeCapture(OfflinePrismGGUFCapture):
    """A causal successor oracle, deliberately unrelated to the proposal head."""

    def __init__(self):
        self.path = Path("/mock/exact.gguf")
        self.native_device = "cpu"
        self.context_size, self.chunk_size = 16, 4
        self.dtype = torch.float16
        self.vocab_size, self.hidden_size = 64, 32
        self.capture_layers = [1, 5]
        self.calls = []
        self.fail = False

    @property
    def _model(self):
        raise AssertionError("native context has no Torch teacher root")

    def capture(self, *, input_ids, attention_mask, loss_mask, return_logits=False):
        assert return_logits is True
        assert input_ids.device.type == "cpu"
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert torch.equal(loss_mask, torch.zeros_like(input_ids))
        assert input_ids.shape[1] <= self.context_size
        self.calls.append(input_ids.clone())
        if self.fail:
            raise RuntimeError("injected native capture failure")
        batch = TeacherCaptureBatch(
            torch.zeros(*input_ids.shape, 64, dtype=self.dtype),
            torch.zeros(*input_ids.shape, self.hidden_size, dtype=self.dtype),
            input_ids.clone(),
            attention_mask.clone(),
            loss_mask.clone(),
        )
        logits = torch.full((*input_ids.shape, self.vocab_size), -10.0)
        logits.scatter_(-1, ((input_ids + 1) % self.vocab_size).unsqueeze(-1), 10.0)
        batch.native_logits = logits
        return batch


class ProposalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft = DFlashDraftModel(tiny_config())
        self.embed_tokens = nn.Embedding(64, 32)
        self.lm_head = nn.Linear(32, 64, bias=False)
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def _compute_logits(self, hidden):
        return self.lm_head(hidden)

    def _block_mask_options(self):
        return {}


class PrismJointVerificationTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.teacher = MockNativeCapture()
        self.model = ProposalModel()
        self.model.train()
        self.bundle = SimpleNamespace(
            model=self.model,
            draft_model=self.model.draft,
            target_head=self.model.lm_head,
            target_vocab_size=64,
            draft_config=self.model.draft.config,
        )
        self.stage = SimpleNamespace(
            name="dflash",
            config=SimpleNamespace(
                training=SimpleNamespace(strategy="dflash"),
                model=SimpleNamespace(
                    target_model_path="/mock/exact.gguf", target_revision="gguf-sha"
                ),
            ),
        )
        self.source = SimpleNamespace(
            _teacher=self.teacher,
            _rows={
                True: [
                    {
                        "input_ids": torch.tensor([1, 2, 3, 4]),
                        "loss_mask": torch.tensor([0, 0, 1, 1]),
                        "domain": "tool",
                    }
                ]
            },
        )
        self.optimizer = torch.optim.Adam(self.model.draft.parameters(), lr=1e-3)
        self.config = dict(
            interval=1,
            max_prompts=1,
            max_new_tokens=8,
            draft_tokens=2,
            trials=1,
            offload_optimizer=True,
        )
        self.metadata = {
            "general.architecture": "qwen35",
            "tokenizer.ggml.eos_token_id": 63,
        }
        self.reader = mock.patch(
            "draftfit.modeling.target.gguf_parts.read_gguf_metadata",
            side_effect=lambda path: SimpleNamespace(metadata=dict(self.metadata)),
        )
        self.read = self.reader.start()
        self.addCleanup(self.reader.stop)

    def evaluate(self, **config):
        return verification.evaluate_generation(
            [self.stage],
            [self.bundle],
            self.source,
            self.optimizer,
            7,
            dict(self.config, **config),
        )

    @staticmethod
    def correct_proposal(bundle, features, anchor, count):
        return anchor[:, None] + torch.arange(1, count + 1)

    def test_native_anchor_and_posterior_accept_exact_tokens_without_projecting_teacher(
        self,
    ):
        with (
            mock.patch.object(
                verification, "_propose", side_effect=self.correct_proposal
            ),
            mock.patch.object(
                verification,
                "_project",
                side_effect=AssertionError("approximate teacher readout"),
            ),
        ):
            result = self.evaluate()
        baseline, candidate = result["baseline"]["runs"][0], result["candidates"][0]
        self.assertEqual(baseline["token_ids"], list(range(3, 11)))
        self.assertEqual(candidate["runs"][0]["token_ids"], baseline["token_ids"])
        self.assertTrue(result["tokenparity"])
        self.assertEqual(candidate["accepted_tokens"], candidate["proposed_tokens"])
        self.assertGreater(candidate["accepted_tokens"], 0)
        self.assertFalse(result["production_engine_speed"])
        runtime = result["bindings"]["runtime"]
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["input_device"], "cpu")
        self.assertEqual(runtime["context_size"], 16)
        self.assertEqual(runtime["decode_state"], "native_full_reset_replay")
        self.assertEqual(runtime["dtype"], "torch.float32")
        self.assertEqual(result["bindings"]["target_model_type"], "qwen35")
        self.read.assert_called_once_with(self.teacher.path)
        json.dumps(result, allow_nan=False)

    def test_rejection_replays_only_committed_prefix(self):
        def wrong_second(bundle, features, anchor, count):
            tokens = self.correct_proposal(bundle, features, anchor, count)
            tokens[:, -1] += 7
            return tokens

        with mock.patch.object(verification, "_propose", side_effect=wrong_second):
            result = self.evaluate()
        candidate = result["candidates"][0]
        self.assertTrue(result["tokenparity"])
        self.assertEqual(candidate["runs"][0]["token_ids"], list(range(3, 11)))
        self.assertGreater(candidate["reject_events"], 0)
        trace = candidate["runs"][0]["traces"][0]
        self.assertEqual(trace["accepted_tokens"], 1)
        self.assertTrue(trace["rejected"])
        self.assertIn([1, 2, 3, 4, 5], [ids[0].tolist() for ids in self.teacher.calls])
        # The rejected suffix appears in a verification query, never a later committed prefix.
        wrong = trace["proposed_token_ids"][-1]
        self.assertNotIn(wrong, candidate["runs"][0]["output_token_ids"])

    def test_real_draft_proposal_keeps_bundle_head(self):
        with mock.patch.object(
            self.model, "_compute_logits", wraps=self.model._compute_logits
        ) as head:
            result = self.evaluate()
        self.assertTrue(result["tokenparity"])
        traces = result["candidates"][0]["runs"][0]["traces"]
        self.assertEqual(head.call_count, len(traces))
        self.assertGreater(head.call_count, 0)

    def test_eos_comes_from_gguf_not_seed(self):
        self.teacher.vocab_size = self.bundle.target_vocab_size = 248320
        self.metadata["tokenizer.ggml.eos_token_id"] = 248046
        self.model.draft.config.eos_token_id = 248044
        self.source._rows[True][0]["input_ids"] = torch.tensor([248043])
        self.source._rows[True][0]["loss_mask"] = torch.zeros(1)
        with mock.patch.object(
            verification, "_propose", side_effect=self.correct_proposal
        ):
            result = self.evaluate()
        self.assertEqual(
            result["baseline"]["runs"][0]["token_ids"], [248044, 248045, 248046]
        )
        self.assertEqual(
            result["candidates"][0]["runs"][0]["token_ids"], [248044, 248045, 248046]
        )
        self.assertEqual(result["bindings"]["config"]["stop_token_ids"], [248046])

    def test_context_limit_preflight_and_exact_boundary(self):
        self.teacher.context_size = 9
        with self.assertRaisesRegex(ValueError, "context limit"):
            self.evaluate()
        self.assertEqual(self.teacher.calls, [])
        self.teacher.context_size = 10
        with mock.patch.object(
            verification, "_propose", side_effect=self.correct_proposal
        ):
            result = self.evaluate()
        self.assertTrue(result["tokenparity"])
        self.assertLessEqual(max(ids.shape[1] for ids in self.teacher.calls), 10)

    def test_native_failure_restores_training_state(self):
        for parameter in self.model.draft.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = [(p, p.grad) for p in self.model.draft.parameters()]
        self.model.draft.layers[0].eval()
        modes = [(module, module.training) for module in self.model.modules()]
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        self.teacher.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected native"):
            self.evaluate()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for parameter, gradient in gradients:
            self.assertIs(parameter.grad, gradient)
        for module, mode in modes:
            self.assertEqual(module.training, mode)

    def test_missing_or_invalid_native_logits_fail_without_fallback(self):
        original = self.teacher.capture
        for replacement in [
            None,
            torch.zeros(1, 2, 64, dtype=torch.float16),
            torch.zeros(1, 1, 64),
            torch.full((1, 2, 64), float("nan")),
        ]:

            def capture(**kwargs):
                batch = original(**kwargs)
                batch.native_logits = replacement
                return batch

            with (
                self.subTest(
                    replacement=(
                        None
                        if replacement is None
                        else (replacement.shape, replacement.dtype)
                    )
                ),
                mock.patch.object(self.teacher, "capture", side_effect=capture),
            ):
                with self.assertRaisesRegex(ValueError, "native logits"):
                    self.evaluate()

    def test_metadata_eos_missing_or_outside_vocabulary_fails(self):
        for eos in [None, 64, -1, True]:
            self.metadata["tokenizer.ggml.eos_token_id"] = eos
            with self.subTest(eos=eos), self.assertRaisesRegex(ValueError, "EOS"):
                self.evaluate()
        self.assertEqual(self.teacher.calls, [])

    def test_requires_explicit_native_wrapper_type(self):
        self.source._teacher = SimpleNamespace(
            native_device="cpu", context_size=16, capture=self.teacher.capture
        )
        with self.assertRaisesRegex(ValueError, "requires.*teacher"):
            self.evaluate()

    def test_mock_native_device_report_is_distinct_from_cpu_input(self):
        # Metadata-only simulation: this test never runs a Metal model/kernel.
        self.teacher.native_device = "mps"
        with mock.patch.object(
            verification, "_propose", side_effect=self.correct_proposal
        ):
            result = self.evaluate()
        self.assertEqual(result["bindings"]["runtime"]["native_device"], "mps")
        self.assertEqual(result["bindings"]["runtime"]["input_device"], "cpu")
        self.assertFalse(result["production_engine_speed"])

    def test_immutable_packed_buffers_are_not_copied_but_mutable_buffers_restore(self):
        # Real readout classes with tiny weights, not multi-GB allocations.
        self.model.embed_tokens = GGUFEmbedding(
            torch.zeros(4, 8 * 34, dtype=torch.uint8),
            torch.ones(1024, dtype=torch.float16),
        )
        self.model.lm_head = GGUFHead(
            torch.ones(4, 1024, dtype=torch.float16),
            torch.ones(1024, dtype=torch.float16),
        )
        self.bundle.target_head = self.model.lm_head
        immutable = list(self.model.embed_tokens.buffers()) + list(
            self.model.lm_head.buffers()
        )
        storage_ids = {value.untyped_storage().data_ptr() for value in immutable}
        snapshots = [value.clone() for value in immutable]

        class RejectPackedClone(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.clone.default:
                    if args[0].untyped_storage().data_ptr() in storage_ids:
                        raise AssertionError("immutable packed readout buffer copied")
                return func(*args, **(kwargs or {}))

        # A frozen ordinary module still has mutable buffers. Do not infer
        # immutability from eval(), requires_grad=False or persistent=False.
        ordinary = nn.Module()
        ordinary.register_buffer("weight", torch.tensor([3.0]))
        ordinary.requires_grad_(False).eval()
        self.model.ordinary_frozen = ordinary
        rotary = nn.Module()
        rotary.register_buffer("inv_freq", torch.tensor([1.0, 2.0]), persistent=False)
        rotary.max_seq_len_cached = 8
        self.model.rotary_cache = rotary
        head_output = self.model.lm_head(torch.ones(1, 1024))
        embedding_output = self.model.embed_tokens(torch.tensor([0, 2]))

        for fail in (False, True):
            with self.subTest(fail=fail):
                original_rotary, original_weight = rotary.inv_freq, ordinary.weight
                modes = [(module, module.training) for module in self.model.modules()]

                def exercise():
                    # Duplicate roots mirror families tying one frozen readout.
                    with (
                        RejectPackedClone(),
                        verification.suspend_training_state(
                            [self.bundle, self.bundle], self.optimizer, self.teacher
                        ),
                    ):
                        torch.testing.assert_close(
                            self.model.lm_head(torch.ones(1, 1024)), head_output
                        )
                        torch.testing.assert_close(
                            self.model.embed_tokens(torch.tensor([0, 2])),
                            embedding_output,
                        )
                        original_rotary.add_(9)
                        rotary.register_buffer(
                            "inv_freq", torch.zeros(2), persistent=False
                        )
                        rotary.max_seq_len_cached = 99
                        ordinary.weight.add_(10)
                        if fail:
                            raise RuntimeError("injected after buffer mutation")

                if fail:
                    with self.assertRaisesRegex(
                        RuntimeError, "injected after buffer mutation"
                    ):
                        exercise()
                else:
                    exercise()
                self.assertIs(rotary.inv_freq, original_rotary)
                self.assertIs(ordinary.weight, original_weight)
                torch.testing.assert_close(rotary.inv_freq, torch.tensor([1.0, 2.0]))
                torch.testing.assert_close(ordinary.weight, torch.tensor([3.0]))
                self.assertEqual(rotary.max_seq_len_cached, 8)
                self.assertIn("inv_freq", rotary._non_persistent_buffers_set)
                for module, mode in modes:
                    self.assertEqual(module.training, mode)
                for value, saved in zip(immutable, snapshots):
                    self.assertIn(value.untyped_storage().data_ptr(), storage_ids)
                    torch.testing.assert_close(value, saved)


if __name__ == "__main__":
    unittest.main()
