"""CPU API regressions; the compiled bridge uses fake llama functions, no model/GPU."""

import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from draftfit.offline_capture.prism_gguf import OfflinePrismGGUFCapture, PRISM_REVISION
from tests.test_gguf_parts import _fixture


# Execute the actual bridge's validation/parameter routing without a Prism build.
# Native numerical capture remains a separate real-runtime gate.
_FAKE_LLAMA = r"""
#pragma once
#include <cstdint>
#include <cstring>
struct llama_model {};
struct llama_context {};
struct llama_model_params { int n_gpu_layers=-1, split_mode=1, main_gpu=0; };
struct llama_context_params {
    int n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, n_outputs_max=0, pooling_type=0;
    bool embeddings=false, offload_kqv=true, op_offload=true;
};
struct llama_batch {
    int n_tokens; int32_t *token, *pos, *n_seq_id, **seq_id; int8_t *logits;
};
constexpr int LLAMA_SPLIT_MODE_NONE=0, LLAMA_POOLING_TYPE_NONE=0;
inline llama_model model;
inline llama_model_params seen_model;
inline llama_context_params seen_context;
inline int loads=0;
inline void llama_backend_init() {}
inline void ggml_backend_load_all() {}
inline llama_model_params llama_model_default_params() { return {}; }
inline llama_context_params llama_context_default_params() { return {}; }
inline llama_model *llama_model_load_from_file(const char *, llama_model_params p) {
    seen_model=p; ++loads; return &model;
}
inline int llama_model_meta_val_str(llama_model *, const char *, char *out, size_t) {
    std::strcpy(out,"qwen35"); return 6;
}
inline int llama_model_n_embd(llama_model *) { return 5120; }
inline int llama_model_n_layer(llama_model *) { return 64; }
inline void *llama_model_get_vocab(llama_model *) { return nullptr; }
inline int llama_vocab_n_tokens(void *) { return 248320; }
inline llama_context *llama_init_from_model(llama_model *, llama_context_params p) {
    seen_context=p; return nullptr; // stop before any capture/allocation
}
inline void llama_free(llama_context *) {}
inline void llama_model_free(llama_model *) {}
inline void llama_set_capture_layers(llama_context *, const int32_t *, int) {}
inline void llama_set_embeddings_nextn(llama_context *, bool, bool) {}
inline void llama_set_embeddings_layer_inp(llama_context *, int, bool) {}
inline void *llama_get_memory(llama_context *) { return nullptr; }
inline void llama_memory_clear(void *, bool) {}
inline llama_batch llama_batch_init(int, int, int) { return {}; }
inline void llama_batch_free(llama_batch) {}
inline int llama_decode(llama_context *, llama_batch) { return -1; }
inline float *llama_get_embeddings_capture_ith(llama_context *, int) { return nullptr; }
inline float *llama_get_embeddings_ith(llama_context *, int) { return nullptr; }
inline float *llama_get_logits_ith(llama_context *, int) { return nullptr; }
extern "C" int prism_test_observed(int field) {
    switch(field) {
        case 0: return loads;
        case 1: return seen_context.offload_kqv;
        case 2: return seen_context.op_offload;
        case 3: return seen_context.n_batch;
        case 4: return seen_context.n_ubatch;
        case 5: return seen_model.main_gpu;
        case 6: return seen_context.n_ctx;
        default: return seen_model.split_mode;
    }
}
"""


class PrismNativeParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable for fake-native bridge test")
        directory = tempfile.TemporaryDirectory(prefix="prism-native-api-")
        cls.addClassCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "llama.h").write_text(_FAKE_LLAMA)
        for name in ("llama-ext.h", "ggml-backend.h"):
            (root / name).write_text("")
        bridge = Path(__file__).resolve().parents[1] / "benchmarks/native/prism_capture_bridge.cpp"
        library = root / "bridge.so"
        result = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(root),
                                 str(bridge), "-o", str(library)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.draftfit_prism_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.lib.draftfit_prism_open.restype = ctypes.c_void_p
        cls.lib.draftfit_prism_error.restype = ctypes.c_char_p
        cls.lib.prism_test_observed.argtypes = [ctypes.c_int]
        cls.lib.prism_test_observed.restype = ctypes.c_int

    def test_context_63_rejected_before_model_loading(self):
        taps = (ctypes.c_int32 * 2)(1, 5)
        observed = self.lib.prism_test_observed
        loads = observed(0)
        self.assertIsNone(self.lib.draftfit_prism_open(b"fixture.gguf", taps, 2, 63, 32, 0))
        self.assertEqual(observed(0), loads, "reject before native model loading")

    def test_context_floor_and_cpu_gpu_offload_routing(self):
        taps = (ctypes.c_int32 * 2)(1, 5)
        observed = self.lib.prism_test_observed
        loads = observed(0)
        for gpu in (0, 1):
            with self.subTest(gpu=gpu):
                self.assertIsNone(self.lib.draftfit_prism_open(b"fixture.gguf", taps, 2, 64, 32, gpu))
                self.assertIn(b"context creation failed", self.lib.draftfit_prism_error())
                self.assertEqual(observed(0), loads + gpu + 1)
                self.assertEqual((observed(1), observed(2)), (gpu, gpu))
                self.assertEqual((observed(3), observed(4), observed(6)), (64, 64, 64))
                self.assertEqual(observed(5), 0 if gpu else -1)
                self.assertEqual(observed(7), 0)  # explicit split NONE


class PrismCaptureAPITest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="prism-wrapper-api-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.gguf, self.bridge = root / "tiny.gguf", root / "bridge.so"
        _fixture(self.gguf)
        self.bridge.touch()
        self.sha = hashlib.sha256(self.gguf.read_bytes()).hexdigest()
        self.lib = Mock()
        self.lib.draftfit_prism_revision.return_value = PRISM_REVISION.encode()
        self.lib.draftfit_prism_open.return_value = 1

        def geometry(handle, address):
            output = (ctypes.c_int32 * 3).from_address(address)
            output[:] = (5120, 4, 64)
            return 0

        self.lib.draftfit_prism_geometry.side_effect = geometry

    def source(self, **kwargs):
        with (patch("draftfit.offline_capture.prism_gguf.ctypes.CDLL", return_value=self.lib),
              patch("draftfit.modeling.target.gguf_parts._VOCAB_SIZE", 4)):
            result = OfflinePrismGGUFCapture(self.gguf, bridge_path=self.bridge,
                                            expected_sha256=self.sha, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_context_63_rejected_before_library_load_and_64_accepted(self):
        with (patch("draftfit.offline_capture.prism_gguf.ctypes.CDLL") as load,
              patch("draftfit.modeling.target.gguf_parts._VOCAB_SIZE", 4)):
            with self.assertRaisesRegex(ValueError, "context"):
                OfflinePrismGGUFCapture(self.gguf, bridge_path=self.bridge,
                                        expected_sha256=self.sha, context_size=63)
            load.assert_not_called()
        self.assertEqual(self.source(context_size=64).context_size, 64)

    def test_sha_revision_and_profile_fail_before_native_model_open(self):
        with patch("draftfit.offline_capture.prism_gguf.ctypes.CDLL") as load:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                OfflinePrismGGUFCapture(self.gguf, bridge_path=self.bridge,
                                        expected_sha256="0" * 64)
            # The miniature vocabulary is rejected by the unmodified production profile.
            with self.assertRaisesRegex(ValueError, "tokenizer.ggml.tokens"):
                OfflinePrismGGUFCapture(self.gguf, bridge_path=self.bridge,
                                        expected_sha256=self.sha)
            load.assert_not_called()
        self.lib.draftfit_prism_revision.return_value = b"wrong-build"
        with self.assertRaisesRegex(ValueError, "revision"):
            self.source()
        self.lib.draftfit_prism_open.assert_not_called()

    def test_model_or_bridge_drift_after_wrapper_creation_blocks_lazy_open(self):
        source = self.source(context_size=64)
        source.set_capture_layers([1, 5])
        self.gguf.write_bytes(self.gguf.read_bytes() + b"changed")
        with source._lock, self.assertRaisesRegex(ValueError, "SHA256"):
            source._open()
        self.lib.draftfit_prism_open.assert_not_called()
        self.gguf.write_bytes(self.gguf.read_bytes()[:-7])
        self.bridge.write_bytes(b"changed")
        with source._lock, self.assertRaisesRegex(ValueError, "bridge changed"):
            source._open()
        self.lib.draftfit_prism_open.assert_not_called()

    def test_geometry_mismatch_closes_failed_native_handle(self):
        source = self.source(context_size=64)
        source.set_capture_layers([1, 5])
        def wrong_geometry(handle, address):
            output = (ctypes.c_int32 * 3).from_address(address)
            output[:] = (5120, 4, 63)
            return 0
        self.lib.draftfit_prism_geometry.side_effect = wrong_geometry
        with source._lock, self.assertRaisesRegex(ValueError, "geometry"):
            source._open()
        self.lib.draftfit_prism_close.assert_called_once_with(1)
        self.assertIsNone(source._handle)

    def test_wrapper_rejects_invalid_ids_masks_taps_and_nonfinite_native_output(self):
        source = self.source(context_size=64, torch_dtype=torch.float16)
        for taps in ([True], [1, 1], [64], [5, 1], [[1]]):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                source.set_capture_layers(taps)
        source.set_capture_layers([1])
        for ids, attn, loss in (([[4]], [[1]], [[1]]),
                                ([[-1]], [[1]], [[1]]),
                                ([[1, 2, 3]], [[1, 0, 1]], [[1, 0, 1]]),
                                ([[1, 2]], [[1, 0]], [[1, 1]]),
                                ([[1]], [[2]], [[1]])):
            with self.subTest(ids=ids, attn=attn), self.assertRaises(ValueError):
                source.capture(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn),
                               loss_mask=torch.tensor(loss))
        self.lib.draftfit_prism_open.assert_not_called()
        def overflow(handle, tokens, count, aux, hidden, logits):
            (ctypes.c_float * (count * 5120)).from_address(aux)[:] = [70000.0] * (count * 5120)
            ctypes.memset(hidden, 0, count * 5120 * 4)
            return 0
        self.lib.draftfit_prism_capture.side_effect = overflow
        ids = torch.ones(1, 1, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "overflow"):
            source.capture(input_ids=ids, attention_mask=ids, loss_mask=ids)

    def test_tap_change_waits_for_native_open_then_rejects(self):
        source = self.source(context_size=64)
        source.set_capture_layers([1, 5])
        entered, attempted = threading.Event(), threading.Event()
        errors = []

        class ObservedLock:
            def __init__(self):
                self.lock = threading.Lock()

            def __enter__(self):
                if threading.current_thread().name == "prism-tap-change":
                    attempted.set()
                self.lock.acquire()

            def __exit__(self, *args):
                self.lock.release()

        source._lock = ObservedLock()

        def native_open(*args):
            entered.set()
            if not attempted.wait(2):
                raise AssertionError("tap setter did not attempt the shared lock")
            self.assertEqual(source.capture_layers, [1, 5])
            return 1

        self.lib.draftfit_prism_open.side_effect = native_open

        def initialize():
            try:
                with source._lock:
                    source._open()
            except Exception as error:
                errors.append(error)

        def change():
            try:
                source.set_capture_layers([1, 5, 9])
            except Exception as error:
                errors.append(error)

        opener = threading.Thread(target=initialize)
        setter = threading.Thread(target=change, name="prism-tap-change")
        opener.start()
        self.assertTrue(entered.wait(2))
        setter.start()
        opener.join(3)
        setter.join(3)
        self.assertFalse(opener.is_alive() or setter.is_alive())
        self.assertEqual(len(errors), 1, errors)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(source.capture_layers, [1, 5])
        source.set_capture_layers([1, 5])  # same taps remain idempotent

    def test_capture_sizes_outputs_after_acquiring_tap_lock(self):
        source = self.source(context_size=64)
        source.set_capture_layers([1, 5])
        waiting, changed = threading.Event(), threading.Event()
        results, errors = [], []

        class GateLock:
            def __init__(self):
                self.lock = threading.Lock()

            def __enter__(self):
                if threading.current_thread().name == "prism-capture":
                    waiting.set()
                    if not changed.wait(2):
                        raise AssertionError("tap change did not finish")
                self.lock.acquire()

            def __exit__(self, *args):
                self.lock.release()

        source._lock = GateLock()

        def native_capture(handle, tokens, count, aux, hidden, logits):
            ctypes.memset(aux, 0, count * len(source.capture_layers) * 5120 * 4)
            ctypes.memset(hidden, 0, count * 5120 * 4)
            return 0

        self.lib.draftfit_prism_capture.side_effect = native_capture

        def capture():
            try:
                ids = torch.ones(1, 1, dtype=torch.int64)
                results.append(source.capture(input_ids=ids, attention_mask=ids, loss_mask=ids))
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=capture, name="prism-capture")
        worker.start()
        try:
            self.assertTrue(waiting.wait(2))
            source.set_capture_layers([1, 5, 9])
        finally:
            changed.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results[0].hidden_states.shape, (1, 1, 3 * 5120))


if __name__ == "__main__":
    unittest.main()
