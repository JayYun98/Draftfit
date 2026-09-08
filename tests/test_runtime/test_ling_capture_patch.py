"""CPU checks execute the actual sink shipped in the pinned Ling patch."""

import ctypes
import itertools
import textwrap
from pathlib import Path
import types
import unittest

import torch

from tests.test_runtime.test_server_capture import _FakeMooncakeStore


def load_sink():
    patch = Path(__file__).resolve().parents[2] / "patches/sglang/ling-8ba213f/spec-capture.patch"
    section = patch.read_text().split(
        "diff --git a/python/sglang/srt/spec_capture_sink.py ", 1
    )[1].split("\ndiff --git ", 1)[0]
    source = "\n".join(
        line[1:] for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    module = types.ModuleType("ling_capture_sink_under_test")
    exec(compile(source, str(patch), "exec"), module.__dict__)
    return module.SpecCaptureSink


class TestLingCapturePatch(unittest.TestCase):
    def test_graph_capture_contract_rejects_null_and_last_without_disabling_full(self):
        patch = Path(__file__).resolve().parents[2] / "patches/sglang/ling-8ba213f/spec-capture.patch"
        section = patch.read_text().split("diff --git a/python/sglang/srt/managers/scheduler.py ", 1)[1].split("\ndiff --git ", 1)[0]
        block = section.split("+            if server_args.return_hidden_states_mode", 1)[1].split("+            from sglang.srt import spec_capture_sink", 1)[0]
        source = "            if server_args.return_hidden_states_mode" + "\n".join(line[1:] if line.startswith("+") else line for line in block.splitlines())
        for mode, decode, prefill in itertools.product((None, "last", "full"), ("disabled", "full"), ("disabled", "full")):
            args = types.SimpleNamespace(return_hidden_states_mode=mode, cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(backend=decode), prefill=types.SimpleNamespace(backend=prefill)))
            with self.subTest(mode=mode, decode=decode, prefill=prefill):
                if mode != "full" and (decode != "disabled" or prefill != "disabled"):
                    with self.assertRaisesRegex(ValueError, "return-hidden-states-mode full"):
                        exec(textwrap.dedent(source), {"server_args": args})
                else:
                    exec(textwrap.dedent(source), {"server_args": args})

    def test_capture_forces_full_even_when_request_asks_for_last(self):
        patch = Path(__file__).resolve().parents[2] / "patches/sglang/ling-8ba213f/spec-capture.patch"
        assignment = next(line[1:].strip() for line in patch.read_text().splitlines()
                          if line.startswith("+        self.return_hidden_states ="))
        for requested in (False, True, "last"):
            for capture in (None, {"sample_id": "test"}):
                obj = types.SimpleNamespace()
                exec(assignment, {"self": obj, "return_hidden_states": requested,
                                  "spec_capture": capture})
                self.assertEqual(obj.return_hidden_states,
                                 True if capture is not None else requested)

    def setUp(self):
        self.sink = load_sink()([3, 7])
        self.store = _FakeMooncakeStore()
        self.sink._store = self.store
        self.addCleanup(self.sink._executor.shutdown, wait=True)
        self.spec = {
            "store_id": "test", "sample_id": "sample", "gen": 1,
            "features": {"aux": "hidden_states", "last_hidden": "target_hidden"},
        }
        self.aux = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        self.last = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

    def test_actual_sink_preserves_tensor_bytes_and_shapes(self):
        result = self.sink.submit_samples([(self.spec, self.aux, self.last)]).result(5)[0]
        self.assertEqual(result["aux_layer_ids"], [3, 7])
        self.assertEqual(result["features"]["hidden_states"]["shape"], [1, 3, 8])
        for name, tensor in (("hidden_states", self.aux), ("target_hidden", self.last)):
            self.assertEqual(self.store._d[f"test/sample/g1/{name}"],
                             ctypes.string_at(tensor.data_ptr(), tensor.numel() * 2))

    def test_missing_requested_feature_never_publishes(self):
        with self.assertRaisesRegex(RuntimeError, "last_hidden"):
            self.sink.put_sample(self.spec, aux=self.aux, last_hidden=None)
        self.assertEqual(self.store._d, {})

    def test_batch_put_failure_removes_partial_objects(self):
        def partial_put(keys, pointers, sizes, config):
            self.store.put_from(keys[0], pointers[0], sizes[0])
            return [0, -1]
        self.store.batch_put_from = partial_put
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.sink.put_sample(self.spec, aux=self.aux, last_hidden=self.last)
        self.assertEqual(self.store._d, {})

    def test_batch_exception_removes_partial_objects(self):
        def partial_put(keys, pointers, sizes, config):
            self.store.put_from(keys[0], pointers[0], sizes[0])
            raise RuntimeError("transport interrupted")
        self.store.batch_put_from = partial_put
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.sink.put_sample(self.spec, aux=self.aux, last_hidden=self.last)
        self.assertEqual(self.store._d, {})


if __name__ == "__main__":
    unittest.main()
