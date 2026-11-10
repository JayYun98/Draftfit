"""Real CPU generation plus explicitly mocked CUDA selection/accounting tests."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from draftfit.data.generation_batching import AdaptiveGenerationBatcher


def tiny():
    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "one": 3, "two": 4, "three": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
                pad_token_id=0,
                eos_token_id=1,
            )
        )
    return model, tokenizer


class GenerationBatchingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = tiny()

    def batcher(self, **kwargs):
        return AdaptiveGenerationBatcher(
            self.model,
            self.tokenizer,
            generation_kwargs={
                "max_new_tokens": 2,
                "do_sample": False,
                "suppress_tokens": [0, 1, 2],
            },
            **kwargs,
        )

    def test_rendered_template_does_not_add_special_tokens(self):
        model, tokenizer = tiny()
        from tokenizers.processors import TemplateProcessing

        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="[EOS] $A", special_tokens=[("[EOS]", 1)]
        )
        batcher = AdaptiveGenerationBatcher(
            model, tokenizer, generation_kwargs={"max_new_tokens": 2}
        )
        self.assertEqual(batcher._encode(["one two"])["input_ids"].tolist(), [[3, 4]])
        model.config.max_position_embeddings = 4
        self.assertEqual(len(batcher.generate(["one two"])), 1)

    def test_real_cpu_padding_continuations_and_order(self):
        prompts = ["one two three", "two", "three one", "one"]
        expected = self.batcher().generate(prompts)
        with (
            patch.object(
                torch.cuda,
                "get_rng_state",
                side_effect=AssertionError("CPU must not use CUDA"),
            ),
            patch.object(self.model, "generate", wraps=self.model.generate) as generate,
        ):
            actual = self.batcher(batch_size=3).generate(prompts)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(prompts))
        self.assertTrue(all(len(text.split()) == 2 for text in actual))
        first = generate.call_args_list[0].kwargs
        self.assertEqual(
            first["attention_mask"].tolist(), [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
        )
        self.assertEqual(self.tokenizer.padding_side, "right")
        self.assertTrue(self.model.training)

    def test_context_no_truncation_and_empty(self):
        batcher = self.batcher()
        self.assertEqual(batcher.generate([]), [])
        with patch.object(
            self.model,
            "generate",
            side_effect=AssertionError("must validate before generation"),
        ):
            with self.assertRaisesRegex(ValueError, "context"):
                batcher.generate([" ".join(["one"] * 31)])
        with self.assertRaisesRegex(TypeError, "list"):
            batcher.generate("one")

    def test_non_oom_errors_propagate_and_restore_mode(self):
        batcher = self.batcher(batch_size=2)
        with patch.object(
            self.model,
            "generate",
            side_effect=RuntimeError("out of memory is only text"),
        ) as generate:
            with self.assertRaisesRegex(RuntimeError, "only text"):
                batcher.generate(["one", "two"])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(batcher.batch_size, 2)
        self.assertTrue(self.model.training)
        with patch.object(
            self.model,
            "generate",
            side_effect=torch.cuda.OutOfMemoryError("CPU is not CUDA"),
        ):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                batcher.generate(["one"])

    def test_real_probe_is_greedy_and_uses_requested_token_budget(self):
        batcher = self.batcher()
        with patch.object(
            self.model, "generate", wraps=self.model.generate
        ) as generate:
            with batcher._rng():
                batcher._run(["one", "one"], probe_end=time.time() + 10)
        kwargs = generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["num_beams"], 1)
        self.assertEqual(kwargs["min_new_tokens"], 2)
        self.assertEqual(kwargs["max_new_tokens"], 2)

    def test_bounds_and_auto_requires_cuda(self):
        for kwargs in (
            {"batch_size": 0},
            {"max_batch_size": True},
            {"memory_fraction": 0.95},
            {"tuning_seconds": 0},
            {"deadline_unix": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.batcher(**kwargs)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.batcher(auto_batch=True)
        with self.assertRaises(TimeoutError):
            self.batcher(deadline_unix=0).generate(["one"])

    def test_cpu_receipt_has_no_prompt_or_raw_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            batcher = self.batcher(
                receipt_path=path, identity={"revision": "private-revision-secret"}
            )
            batcher.generate(["one two"])
            text = path.read_text()
            self.assertNotIn("one two", text)
            self.assertNotIn("private-revision-secret", text)
            receipt = json.loads(text)
            self.assertEqual(receipt["mode"], "cpu")
            self.assertEqual(receipt["final_batch_size"], 1)
            self.assertNotIn("selected", receipt)


class MockedCUDA(unittest.TestCase):
    """No CUDA allocations: telemetry, device, generation and CUDA RNG are mocked."""

    def setUp(self):
        self.model, self.tokenizer = tiny()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "receipt.json"
        self.spec = dict(
            name="mock-cuda",
            total_bytes=1000,
            free_bytes=900,
            capability=[8, 0],
            torch_version="mock",
            cuda_version="mock",
        )
        for target, kwargs in (
            (
                "draftfit.data.generation_batching.cuda_spec",
                {"side_effect": lambda *args: dict(self.spec)},
            ),
            ("draftfit.data.generation_batching.generation_limit", {"return_value": 4}),
            ("torch.cuda.memory_reserved", {"return_value": 100}),
            ("torch.cuda.reset_peak_memory_stats", {}),
            ("torch.cuda.empty_cache", {}),
            ("torch.cuda.synchronize", {}),
            ("torch.cuda.max_memory_reserved", {"return_value": 500}),
            (
                "torch.cuda.get_rng_state",
                {"side_effect": lambda *args: torch.get_rng_state()},
            ),
            ("torch.cuda.set_rng_state", {}),
        ):
            mocked = patch(target, **kwargs)
            mocked.start()
            self.addCleanup(mocked.stop)

    def batcher(self, **kwargs):
        # Construct on CPU, then replace only device metadata. Never move tensors.
        result = AdaptiveGenerationBatcher(
            self.model,
            self.tokenizer,
            max_batch_size=4,
            identity=kwargs.pop(
                "identity", {"model": "fixture", "revision": "immutable"}
            ),
            receipt_path=self.path,
            generation_kwargs={"max_new_tokens": 2},
            **kwargs,
        )
        result.device = torch.device("cuda:0")
        result.auto_batch = True
        return result

    def run_mock(self, prompts, *, probe_end=None):
        torch.rand(1)
        return None if probe_end is not None else ["continuation"] * len(prompts)

    def test_measured_selection_rng_and_exact_receipt_reuse(self):
        batcher = self.batcher()
        before = torch.get_rng_state().clone()
        with (
            patch.object(batcher, "_run", side_effect=self.run_mock) as run,
            patch(
                "draftfit.data.generation_batching.time.perf_counter",
                side_effect=[0, 4, 10, 12, 20, 21],
            ),
        ):
            batcher._select("one two", 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(batcher.batch_size, 4)
        self.assertEqual([len(call.args[0]) for call in run.call_args_list], [1, 2, 4])
        reused = self.batcher()
        with patch.object(
            reused, "_run", side_effect=AssertionError("receipt should reuse")
        ):
            reused._select("one", 1)
        self.assertEqual(reused.batch_size, 4)
        self.assertTrue(reused.evidence["reused"])
        self.assertNotIn("one two", self.path.read_text())

    def test_receipt_rechecks_memory_length_and_identity(self):
        batcher = self.batcher()
        with patch.object(batcher, "_run", side_effect=self.run_mock):
            batcher._select("one", 1)
        saved = json.loads(self.path.read_text())
        for change in ("memory", "length", "identity", "hardware"):
            with self.subTest(change=change):
                self.path.write_text(json.dumps(saved))
                self.spec.update(free_bytes=900, name="mock-cuda")
                fresh = self.batcher()
                length = 1
                if change == "memory":
                    self.spec["free_bytes"] = 100
                elif change == "length":
                    length = 2
                elif change == "identity":
                    fresh._contract = "different"
                else:
                    self.spec["name"] = "different-device"
                with patch.object(
                    fresh, "_run", side_effect=ValueError("retune reached")
                ) as run:
                    with self.assertRaisesRegex(ValueError, "retune reached"):
                        fresh._select("one", length)
                    self.assertTrue(run.called)
                self.assertEqual(json.loads(self.path.read_text())["status"], "failed")

    def test_model_timeout_is_not_a_tuning_timeout(self):
        batcher = self.batcher()
        with patch.object(
            batcher, "_run", side_effect=TimeoutError("external model error")
        ):
            with self.assertRaisesRegex(TimeoutError, "external model"):
                batcher._select("one", 1)
        self.assertEqual(batcher.evidence["status"], "failed")

    def test_longest_prompt_and_memory_recheck_each_call(self):
        batcher = self.batcher()
        selected = {"status": "ok", "batch_size": 2, "peak_bytes": 500}
        batcher._selected = selected
        _, hardware, _ = batcher._budget()
        batcher.evidence.update(
            status="ok", hardware=hardware, prompt_tokens=3, selected=selected
        )
        with (
            patch.object(batcher, "_budget", wraps=batcher._budget) as budget,
            patch.object(batcher, "_run", side_effect=self.run_mock),
        ):
            batcher.generate(["one", "one two three"])
            batcher.generate(["two"])
        self.assertEqual(budget.call_count, 2)
        batcher = self.batcher()
        batcher.receipt_path = None
        with patch.object(batcher, "_run", side_effect=self.run_mock) as run:
            batcher.generate(["one", "one two three", "two"])
        probes = [call for call in run.call_args_list if "probe_end" in call.kwargs]
        self.assertTrue(probes)
        self.assertTrue(all(set(call.args[0]) == {"one two three"} for call in probes))

    def test_no_valid_measurement_and_expired_budget_fail(self):
        batcher = self.batcher()
        with patch.object(batcher, "_run", side_effect=torch.cuda.OutOfMemoryError()):
            with self.assertRaisesRegex(RuntimeError, "no measured"):
                batcher._select("one", 1)
        self.assertEqual(batcher.evidence["trials"][0]["status"], "oom")
        batcher.deadline_unix = 0
        with patch.object(batcher, "_run", side_effect=AssertionError("deadline")):
            with self.assertRaises(RuntimeError):
                batcher._select("one", 1)
        self.assertNotIn("selected", batcher.evidence)

    def test_sampling_oom_ceiling_survives_next_call_and_receipt_reload(self):
        batcher = self.batcher()
        # Choose four with greedy probes, fail that batch in actual sampling.
        probes = []

        def generate(prompts, *, probe_end=None):
            if probe_end is not None:
                probes.append(len(prompts))
                return None
            if len(prompts) > 2:
                raise torch.cuda.OutOfMemoryError("mock sampling allocation")
            return ["answer"] * len(prompts)

        with (
            patch.object(batcher, "_run", side_effect=generate),
            patch(
                "draftfit.data.generation_batching.time.perf_counter",
                side_effect=[0, 4, 10, 12, 20, 21],
            ),
        ):
            batcher.generate(["one"] * 4)
        self.assertEqual(batcher.evidence["fallback_ceiling"], 2)
        probes.clear()
        with patch.object(batcher, "_run", side_effect=generate):
            batcher.generate(["one"] * 4)
        self.assertTrue(probes)
        self.assertLessEqual(max(probes), 2)
        fresh = self.batcher()
        with patch.object(fresh, "_run", side_effect=generate):
            fresh.generate(["one"] * 4)
        self.assertLessEqual(fresh.batch_size, 2)
        fresh = self.batcher(identity={"model": "fixture", "revision": "changed"})
        probes.clear()
        with patch.object(fresh, "_run", side_effect=generate):
            fresh._select("one", 1)
        self.assertEqual(max(probes), 4)
        self.assertNotIn("fallback_ceiling", fresh.evidence)

    def test_actual_cuda_oom_class_retries_pending_and_restores_rng(self):
        batcher = self.batcher(batch_size=4)
        batcher.auto_batch = False
        seen, samples = [], []

        def attempt(prompts, **kwargs):
            seen.append(list(prompts))
            samples.append(torch.rand(1).item())
            if len(prompts) > 2:
                raise torch.cuda.OutOfMemoryError("mock CUDA allocation failure")
            return ["answer:" + p for p in prompts]

        torch.manual_seed(123)
        with patch.object(batcher, "_run", side_effect=attempt):
            result = batcher.generate(["one", "two", "three", "one two", "two three"])
        self.assertEqual(samples[0], samples[1])
        self.assertNotEqual(samples[1], samples[2])
        self.assertEqual(seen[:2], [["one", "two", "three", "one two"], ["one", "two"]])
        self.assertEqual(
            result,
            ["answer:" + p for p in ["one", "two", "three", "one two", "two three"]],
        )
        receipt = json.loads(self.path.read_text())
        self.assertEqual(receipt["final_batch_size"], 2)
        self.assertEqual(receipt["fallbacks"][0]["reason"], "cuda_oom")
        self.assertNotIn("selected", receipt)
        with patch.object(
            batcher, "_run", side_effect=torch.cuda.OutOfMemoryError()
        ) as run:
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                batcher.generate(["one"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(batcher.batch_size, 2)
        self.assertEqual(json.loads(self.path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
