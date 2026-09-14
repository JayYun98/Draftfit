"""Mocked vLLM boundary tests; no vLLM/GPU parity or throughput claim."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from specforge.offline_capture.vllm import OfflineVLLMCapture, _load_norm


class VLLMCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig(
            num_hidden_layers=4,
            hidden_size=2,
            vocab_size=16,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.teacher = OfflineVLLMCapture()
        self.teacher.config = self.config
        self.teacher._norm = LlamaRMSNorm(2, eps=self.config.rms_norm_eps)
        self.teacher._options = {
            "model": "fixture",
            "dtype": torch.float32,
            "max_model_len": 8,
        }
        self.teacher._llm = None
        self.teacher._directory = None
        self.teacher.capture_layers = None
        names = [
            "vllm",
            "vllm.config",
            "vllm.config.kv_transfer",
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
        ]
        self.modules = {name: ModuleType(name) for name in names}
        self.modules["vllm"].__version__ = "0.22.1"
        self.engine = Mock()
        self.modules["vllm"].LLM = Mock(return_value=self.engine)
        self.modules["vllm"].SamplingParams = Mock(side_effect=lambda **kw: kw)
        self.modules["vllm.config.kv_transfer"].KVTransferConfig = Mock(
            side_effect=lambda **kw: kw
        )
        self.connector = SimpleNamespace(
            load_hidden_states=Mock(side_effect=load_file),
            cleanup_hidden_states=Mock(side_effect=lambda path: Path(path).unlink()),
        )
        self.modules[names[-1]].example_hidden_states_connector = self.connector
        self.patch = patch.dict(sys.modules, self.modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.teacher.close)

    def capture(self):
        return self.teacher.capture(
            input_ids=torch.tensor([[0, 3, 4], [5, 6, 0]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]),
            loss_mask=torch.tensor([[0, 0, 1], [0, 1, 0]]),
        )

    def produce(self, prompts, *args, **kwargs):
        outputs = []
        for i, prompt in enumerate(prompts):
            path = Path(self.teacher._directory.name) / f"{i}.safetensors"
            # traversal slots [1, 3, 4], each slot gets a distinct feature vector.
            states = (
                torch.tensor([[[1.0, 2.0], [3.0, 4.0], [6.0, 8.0]]])
                .expand(2, -1, -1)
                .contiguous()
            )
            save_file(
                {
                    "hidden_states": states,
                    "token_ids": torch.tensor(prompt["prompt_token_ids"]),
                },
                path,
            )
            outputs.append(
                SimpleNamespace(
                    request_id=str(i),
                    finished=True,
                    prompt_token_ids=prompt["prompt_token_ids"],
                    kv_transfer_params={"hidden_states_path": str(path)},
                )
            )
        return outputs

    def test_order_padding_and_normalized_final(self):
        self.teacher.set_capture_layers([0, 2, 3])
        options = self.modules["vllm"].LLM.call_args.kwargs
        self.assertEqual(options["tensor_parallel_size"], 1)
        self.assertEqual(options["pipeline_parallel_size"], 1)
        self.assertEqual(options["distributed_executor_backend"], "mp")
        self.assertFalse(options["enable_prefix_caching"])
        self.assertFalse(options["enable_chunked_prefill"])
        self.assertTrue(options["enforce_eager"])
        self.assertEqual(
            options["speculative_config"]["draft_model_config"]["hf_config"][
                "eagle_aux_hidden_state_layer_ids"
            ],
            [1, 3, 4],
        )
        self.engine.generate.side_effect = self.produce
        batch = self.capture()
        torch.testing.assert_close(
            batch.hidden_states[0, 1], torch.tensor([1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
        )
        torch.testing.assert_close(
            batch.last_hidden_states[0, 1], self.teacher._norm(torch.tensor([6.0, 8.0]))
        )
        self.assertEqual(batch.hidden_states[0, 0].count_nonzero(), 0)
        self.assertEqual(
            len(list(Path(self.teacher._directory.name).glob("*.safetensors"))), 0
        )

    def test_rejects_bad_boundary_evidence(self):
        self.teacher.set_capture_layers([0, 2, 3])
        for mutation in (
            "tokens",
            "shape",
            "nan",
            "dtype",
            "order",
            "count",
            "duplicate",
            "missing",
        ):
            with self.subTest(mutation=mutation):

                def produce(prompts, *args, **kwargs):
                    outputs = self.produce(prompts)
                    if mutation == "order":
                        return list(reversed(outputs))
                    if mutation == "count":
                        return outputs[:1]
                    if mutation == "duplicate":
                        outputs[1].request_id = outputs[0].request_id
                    if mutation == "missing":
                        outputs[0].kv_transfer_params = None
                    return outputs

                self.engine.generate.side_effect = produce

                def load(path):
                    payload = load_file(path)
                    if mutation == "tokens":
                        payload["token_ids"][0] = 9
                    if mutation == "shape":
                        payload["hidden_states"] = payload["hidden_states"][:, :1]
                    if mutation == "nan":
                        payload["hidden_states"][0, 0, 0] = float("nan")
                    if mutation == "dtype":
                        payload["hidden_states"] = payload["hidden_states"].to(
                            torch.int64
                        )
                    return payload

                self.connector.load_hidden_states.side_effect = load
                with self.assertRaises(ValueError):
                    self.capture()

    def test_private_paths_and_masks(self):
        self.teacher.set_capture_layers([0, 2, 3])
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "private.safetensors"
            target.touch()
            for path in (
                target,
                Path(self.teacher._directory.name) / "link.safetensors",
            ):
                if path != target:
                    path.symlink_to(target)
                with self.assertRaises(ValueError):
                    self.teacher._checked_path(str(path))
                self.assertTrue(target.exists())
            lock_target = Path(self.teacher._directory.name) / "states.safetensors"
            Path(str(lock_target) + ".lock").symlink_to(target)
            with self.assertRaises(ValueError):
                self.teacher._checked_path(str(lock_target))
        with self.assertRaises(ValueError):
            self.teacher.capture(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 0]]),
                loss_mask=torch.tensor([[0, 1]]),
            )
        self.engine.generate.assert_not_called()
        with self.assertRaises(ValueError):
            self.teacher.capture(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 0, 1]]),
                loss_mask=torch.tensor([[0, 0, 1]]),
            )

    def test_selective_norm_load(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = torch.tensor([2.0, 3.0])
            save_file(
                {"model.norm.weight": weight},
                str(Path(directory) / "model.safetensors"),
            )
            norm = _load_norm(directory, self.config, None)
            torch.testing.assert_close(norm.weight, weight)
            self.assertEqual(sum(p.numel() for p in norm.parameters()), 2)
            (Path(directory) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.norm.weight": "model.safetensors"}})
            )
            torch.testing.assert_close(
                _load_norm(directory, self.config, None).weight, weight
            )
            (Path(directory) / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"model.norm.weight": "../outside.safetensors"}}
                )
            )
            with self.assertRaises(ValueError):
                _load_norm(directory, self.config, None)

    def test_load_is_lazy_and_settings_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            self.config.save_pretrained(directory)
            save_file(
                {"model.norm.weight": torch.ones(2)},
                str(Path(directory) / "model.safetensors"),
            )
            teacher = OfflineVLLMCapture.from_pretrained(
                directory,
                revision="fixture",
                torch_dtype=torch.float32,
                cache_dir=directory,
                max_model_len=16,
                gpu_memory_utilization=0.5,
            )
            self.addCleanup(teacher.close)
            self.modules["vllm"].LLM.assert_not_called()
            teacher.set_capture_layers([0], capture_method="dflash")
            options = self.modules["vllm"].LLM.call_args.kwargs
            self.assertEqual(options["download_dir"], directory)
            self.assertEqual(options["revision"], "fixture")
            self.assertEqual(options["max_model_len"], 16)

    def test_guards_before_engine_start(self):
        for layers in ([True], [4], [1, 1], [2, 0], []):
            with self.assertRaises(ValueError):
                self.teacher.set_capture_layers(layers, capture_method="dflash")
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
            with self.assertRaises(ValueError):
                self.teacher.set_capture_layers([0], capture_method="dflash")
        with patch.dict("os.environ", {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}):
            with self.assertRaises(ValueError):
                self.teacher.set_capture_layers([0], capture_method="dflash")
        self.modules["vllm"].__version__ = "0.21.0"
        with self.assertRaises(ValueError):
            self.teacher.set_capture_layers([0], capture_method="dflash")
        self.modules["vllm"].LLM.assert_not_called()

    def test_spawn_scope_failure_and_close(self):
        previous = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")

        def fail(**options):
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            raise RuntimeError("mock engine initialization failed")

        self.modules["vllm"].LLM.side_effect = fail
        with self.assertRaises(RuntimeError):
            self.teacher.set_capture_layers([0], capture_method="dflash")
        self.assertEqual(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), previous)
        self.assertIsNone(self.teacher._directory)
        self.modules["vllm"].LLM.side_effect = None
        self.teacher.set_capture_layers([0], capture_method="dflash")
        directory = Path(self.teacher._directory.name)
        self.teacher.close()
        self.engine.llm_engine.engine_core.shutdown.assert_called_once()
        self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
