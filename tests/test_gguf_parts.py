"""Tiny synthetic GGUF codec, independent transform, gradient and boundary checks."""
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from draftfit.modeling.target import gguf_parts as gguf
from draftfit.modeling.target import quantized_parts


def _string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(typ, value):
    if typ == 8:
        return _string(value)
    if typ == 9:
        element, items = value
        return struct.pack("<IQ", element, len(items)) + b"".join(
            _value(element, item) for item in items)
    return struct.pack("<" + {4: "I", 5: "i", 7: "B", 2: "H"}[typ], value)


def _fixture(path, *, changes=None, duplicate=None, descriptors=None, alignment=64):
    """Four actual-width rows, real block/sign geometry; only vocab is reduced."""
    width, rows = 5120, 4
    signs = [1 if i % 7 < 3 else -1 for i in range(sum(gguf._SIGN_WIDTHS))]
    meta = {
        "general.architecture": (8, "qwen35"),
        "general.alignment": (4, alignment),
        "qwen35.embedding_length": (4, width),
        "qwen35.block_count": (4, 64),
        "tokenizer.ggml.tokens": (9, (8, ["a", "b", "c", "d"])),
        "tokenizer.ggml.merges": (9, (8, ["a b", "b c"])),
        "tokenizer.ggml.eos_token_id": (4, 3),
        "prism.hadamard.version": (4, 1),
        "prism.hadamard.block_size": (4, 1024),
        "prism.hadamard.transform": (8, "normalized-sylvester-walsh-hadamard"),
        "prism.hadamard.axis": (8, "input-last-dimension"),
        "prism.hadamard.sign_mode": (8, "explicit"),
        "prism.hadamard.weight_names": (9, (8, ["output.weight"])),
        "prism.hadamard.inverse_weight_names": (9, (8, ["token_embd.weight"])),
        "prism.hadamard.sign_widths": (9, (5, gguf._SIGN_WIDTHS)),
        "prism.hadamard.sign_values": (9, (5, signs)),
        "prism.hadamard.gdn_v_grouped": (7, True),
    }
    for key, value in (changes or {}).items():
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
    entries = list(meta.items())
    if duplicate:
        entries.append((duplicate, meta[duplicate]))
    # Row/group-dependent scales and nonperiodic bytes exercise both ordering axes.
    data, expected = [], []
    for table in range(2):
        raw, dense = bytearray(), []
        for row in range(rows):
            values = []
            for group in range(width // 128):
                scale = struct.unpack("<e", struct.pack("<e", .0031 * (1 + row + group + table)))[0]
                raw.extend(struct.pack("<e", scale))
                for j in range(32):
                    code = (j * 53 + row * 17 + group * 11 + table * 7) % 256
                    raw.append(code)
                    values.extend([(((code >> shift) & 3) - 1) * scale
                                   for shift in (0, 2, 4, 6)])
            dense.append(values)
        data.append(bytes(raw))
        expected.append(torch.tensor(dense, dtype=torch.float16))
    second = (len(data[0]) + alignment - 1) // alignment * alignment
    desc = descriptors or [("output.weight", (width, rows), 142, 0),
                           ("token_embd.weight", (width, rows), 142, second)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(desc), len(entries))
    header += b"".join(_string(key) + struct.pack("<I", typ) + _value(typ, value)
                       for key, (typ, value) in entries)
    for name, dims, typ, offset in desc:
        header += (_string(name) + struct.pack("<I", len(dims))
                   + struct.pack("<" + "Q" * len(dims), *dims)
                   + struct.pack("<IQ", typ, offset))
    header += b"\0" * (-len(header) % alignment)
    path.write_bytes(header + data[0] + b"\0" * (second - len(data[0])) + data[1])
    return expected, torch.tensor(signs[:width], dtype=torch.float32)


def _reference_transform(rows, signs, inverse=False):
    # Independent explicit Sylvester matrix, five separate 1024-wide blocks.
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < 1024:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0)
    matrix /= math.sqrt(1024)
    value = rows.float() if inverse else rows.float() * signs
    value = (value.reshape(-1, 1024) @ matrix).reshape(rows.shape)
    return (value * signs if inverse else value).half()


class GGUFPartsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.gguf"

    def load(self, **kwargs):
        # Production geometry is fixed. Tiny fixtures only override row count.
        with patch.object(gguf, "_VOCAB_SIZE", 4):
            return gguf.load_target_parts(self.path, **kwargs)

    def test_quantized_loader_routes_local_gguf_without_hf_model_import(self):
        _fixture(self.path)
        upper = self.path.with_suffix(".GGUF")
        self.path.rename(upper)
        # Match the existing family provider's kwargs; only device/dtype belong
        # to the explicit local GGUF loader. No HF model/config import is needed.
        kwargs = dict(embed_key="model.embed_tokens.weight", lm_head_key="lm_head.weight",
                      revision="caller-owned-identity", cache_dir="unused-local-file",
                      trust_remote_code=False, device="cpu", dtype=torch.bfloat16)
        with (patch.object(gguf, "_VOCAB_SIZE", 4),
              patch.dict(sys.modules, {"draftfit.modeling.target.target_utils": None}),
              patch.object(gguf, "load_target_parts", wraps=gguf.load_target_parts) as loader):
            for embedding_only in (False, True):
                with self.subTest(embedding_only=embedding_only):
                    parts = quantized_parts.load_target_parts(
                        upper, embedding_only=embedding_only, **kwargs)
                    loader.assert_called_with(upper, embedding_only=embedding_only,
                                              device="cpu", dtype=torch.bfloat16)
                    self.assertEqual(parts.embed_tokens(torch.tensor([0])).dtype, torch.bfloat16)
                    self.assertEqual(hasattr(parts, "lm_head"), not embedding_only)

    def test_quantized_loader_preserves_nonlocal_and_non_gguf_route(self):
        directory = self.path.parent / "directory.gguf"
        directory.mkdir()
        non_gguf = self.path.with_suffix(".safetensors")
        non_gguf.touch()
        fallback = Mock(return_value=object())
        module = SimpleNamespace(TargetEmbeddingsAndHead=SimpleNamespace(from_pretrained=fallback))
        with (patch.dict(sys.modules, {"draftfit.modeling.target.target_utils": module}),
              patch.object(gguf, "load_target_parts") as loader):
            for path in (directory, non_gguf, self.path, "owner/model.gguf"):
                with self.subTest(path=path):
                    result = quantized_parts.load_target_parts(path, device="cpu", revision="pinned")
                    self.assertIs(result, fallback.return_value)
                    fallback.assert_called_with(path, device="cpu", revision="pinned")
            loader.assert_not_called()

    def test_quantized_loader_rejects_invalid_gguf_without_fallback(self):
        self.path.write_bytes(b"not a GGUF")
        with patch.dict(sys.modules, {"draftfit.modeling.target.target_utils": None}):
            with self.assertRaisesRegex(ValueError, "GGUF"):
                quantized_parts.load_target_parts(self.path)
            for key in ("embed_key", "lm_head_key"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fixed"):
                    quantized_parts.load_target_parts(self.path, **{key: "custom.weight"})

    def test_header_and_exact_codec(self):
        expected, _ = _fixture(self.path)
        header = gguf.read_gguf_metadata(self.path)
        self.assertEqual(header.metadata["tokenizer.ggml.tokens"], 4)
        self.assertNotIn("tokenizer.ggml.merges", header.metadata)
        self.assertEqual(header.metadata["tokenizer.ggml.eos_token_id"], 3)
        tokenizer = gguf.read_gguf_metadata(self.path, include_tokenizer=True).metadata
        self.assertEqual(tokenizer["tokenizer.ggml.tokens"], ["a", "b", "c", "d"])
        self.assertEqual(tokenizer["tokenizer.ggml.merges"], ["a b", "b c"])
        self.assertEqual(header.data_offset % 64, 0)
        with self.path.open("rb") as source:
            raw = gguf._read_rows(source, header, "output.weight", 0, 4)
        torch.testing.assert_close(gguf.dequantize_pq2_rows(raw), expected[0], atol=0, rtol=0)
        # Fixed byte 0xe4 must yield -1, 0, 1, 2 consecutively, not planes.
        block = torch.tensor(list(struct.pack("<e", .25)) + [0xe4] * 32,
                             dtype=torch.uint8).reshape(1, 34)
        torch.testing.assert_close(gguf.dequantize_pq2_rows(block),
                                   torch.tensor([[-.25, 0, .25, .5]]).repeat(1, 32).half())

    def test_independent_head_inverse_embedding_and_gradient(self):
        (head, embedding), signs = _fixture(self.path)
        parts = self.load()
        ids = torch.tensor([[3, 0, 3], [1, 2, 0]])
        expected = _reference_transform(embedding[ids], signs, inverse=True)
        torch.testing.assert_close(parts.embed_tokens(ids), expected, atol=3e-3, rtol=2e-3)
        torch.manual_seed(29)
        x = (torch.randn(2, 3, 5120) * .1).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        ref_logits = F.linear(_reference_transform(ref_x.half(), signs), head).float()
        logits = parts.lm_head(x)
        torch.testing.assert_close(logits, ref_logits, atol=4e-3, rtol=4e-3)
        logits.square().mean().backward()
        ref_logits.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
        torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-4, rtol=1e-2)
        self.assertEqual(list(parts.parameters()), [])
        self.assertFalse(parts.training)
        # A missing/inverted transform must be detectable by this fixture.
        self.assertGreater((logits - F.linear(x.half(), head).float()).abs().max().item(), .1)
        wrong = _reference_transform(embedding[ids], signs, inverse=False)
        self.assertGreater((expected - wrong).abs().max().item(), .1)

    def test_frozen_precision_device_cast_autocast_and_empty_ids(self):
        _fixture(self.path)
        parts = self.load()
        original = {name: value.clone() for name, value in parts.named_buffers()}
        ids = torch.tensor([1, 3])
        embedding = parts.embed_tokens(ids)
        x = torch.linspace(-.1, .2, 5120).reshape(1, -1)
        logits = parts.lm_head(x)
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            parts.to(device="cpu", dtype=dtype)
            for name, value in parts.named_buffers():
                self.assertEqual(value.dtype, original[name].dtype)
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)
            self.assertEqual(parts.embed_tokens(ids).dtype, dtype)
            torch.testing.assert_close(parts.embed_tokens(ids), embedding.to(dtype), atol=0, rtol=0)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                torch.testing.assert_close(parts.lm_head(x), logits, atol=0, rtol=0)
        self.assertEqual(parts.embed_tokens(torch.empty(2, 0, dtype=torch.long)).shape,
                         (2, 0, 5120))
        for ids in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([.1])):
            with self.assertRaises(ValueError):
                parts.embed_tokens(ids)
        only = self.load(embedding_only=True)
        self.assertFalse(hasattr(only, "lm_head"))

    def test_exact_profile_rejects_tiny_vocab_without_override(self):
        _fixture(self.path)
        with self.assertRaisesRegex(ValueError, "tokenizer.ggml.tokens"):
            gguf.load_target_parts(self.path)

    def test_reject_mismatched_missing_or_unknown_transform_metadata(self):
        invalid = {
            "general.architecture": (8, "qwen3"),
            "qwen35.embedding_length": (4, 4096),
            "qwen35.block_count": (4, 63),
            "qwen35.vocab_size": (4, 7),
            "prism.hadamard.version": (4, 2),
            "prism.hadamard.block_size": (4, 512),
            "prism.hadamard.transform": (8, "unnormalized"),
            "prism.hadamard.axis": (8, "output"),
            "prism.hadamard.sign_mode": (8, "identity"),
            "prism.hadamard.tied_output": (7, True),
            "prism.hadamard.gdn_v_grouped": (7, False),
            "qwen35.tie_word_embeddings": (7, True),
            "prism.hadamard.sign_widths": (9, (5, [5120, 17408, 6144])),
            "prism.hadamard.sign_values": (9, (5, [0] * 28672)),
            "prism.hadamard.weight_names": (9, (8, ["token_embd.weight"])),
            "prism.hadamard.inverse_weight_names": (9, (8, ["output.weight"])),
            "prism.hadamard.future_required": (4, 1),
            "split.count": (2, 2),
            "general.alignment": (4, 3),
        }
        for key, value in invalid.items():
            with self.subTest(key=key):
                _fixture(self.path, changes={key: value})
                with self.assertRaises(ValueError):
                    self.load()
        for key in ("prism.hadamard.version", "prism.hadamard.sign_values",
                    "prism.hadamard.inverse_weight_names"):
            with self.subTest(missing=key):
                _fixture(self.path, changes={key: None})
                with self.assertRaises(ValueError):
                    self.load()
        _fixture(self.path, changes={"prism.hadamard.version": (7, True)})
        with self.assertRaisesRegex(ValueError, "wrong metadata type"):
            self.load()

    def test_reject_bad_tensor_spans_types_duplicates_and_truncation(self):
        good = ("output.weight", (5120, 4), 142, 0)
        for desc in (
            [good], [good, good],
            [good, ("token_embd.weight", (5120, 4), 142, 0)],
            [good, ("token_embd.weight", (5120, 4), 142, 1 << 60)],
            [good, ("token_embd.weight", (5120, 4), 142, 1)],
            [("output.weight", (5120, 4), 42, 0)],
            [("output.weight", (5119, 4), 142, 0)],
            [("output.weight", (1 << 40, 4), 142, 0)],
        ):
            with self.subTest(descriptors=desc):
                _fixture(self.path, descriptors=desc)
                with self.assertRaises(ValueError):
                    self.load()
        _fixture(self.path, duplicate="general.architecture")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()
        _fixture(self.path)
        raw = self.path.read_bytes()
        for cut in (3, 20, 100, len(raw) - 1):
            self.path.write_bytes(raw[:cut])
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                self.load()

    def test_bounded_malformed_header_rejected_before_tensor_allocation(self):
        prefix = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
        samples = [b"GGUF" + struct.pack("<IQQ", 2, 1, 1),
                   b"GGUF" + struct.pack("<IQQ", 3, 1 << 60, 1),
                   prefix + struct.pack("<Q", 1 << 60),
                   prefix + _string("x") + struct.pack("<IIQ", 9, 8, 1 << 60),
                   prefix + _string("x") + struct.pack("<IIQ", 9, 9, 1),
                   prefix + _string("x") + struct.pack("<IB", 7, 2),
                   prefix + _string("x") + struct.pack("<I", 99)]
        for raw in samples:
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with patch.object(torch, "empty", side_effect=AssertionError("allocated")):
                    with self.assertRaises(ValueError):
                        self.load()

    def test_nonfinite_and_fp16_overflow_scales_fail(self):
        for scale in (math.nan, math.inf, 65504.):
            raw = struct.pack("<e", scale) + bytes([255] * 32)
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                gguf.dequantize_pq2_rows(torch.tensor(list(raw), dtype=torch.uint8).reshape(1, 34))

    def test_cpu_numpy_exact_bits_chunk_boundary_strides_and_owned_output(self):
        import numpy as np

        # Signed zeros, subnormals, normal boundaries and largest finite values.
        bits = [0, 0x8000, 1, 0x8001, 0x3ff, 0x83ff, 0x400, 0x8400,
                0x3555, 0xb555, 0x77ff, 0xf7ff, 0x7800, 0xf800, 0x7bff, 0xfbff]
        raw = bytearray()
        for row in range(513):
            for group in range(2):
                scale = bits[(row + group) % len(bits)]
                raw.extend(struct.pack("<H", scale))
                # A large scale is valid when the overflowing code 3 is unused.
                raw.extend(bytes([0x24] * 32) if (scale & 0x7fff) >= 0x7800
                           else bytes((row * 17 + group * 11 + j) % 256 for j in range(32)))
        packed = torch.frombuffer(raw, dtype=torch.uint8).reshape(513, 68)
        backing = torch.empty(514, 136, dtype=torch.uint8)
        strided = backing[1:, ::2]
        strided.copy_(packed)
        original = strided.clone()
        reference = gguf._dequantize_pq2_rows_torch(strided)
        actual = gguf._dequantize_pq2_rows_numpy(strided, np)
        torch.testing.assert_close(actual.view(torch.int16), reference.view(torch.int16),
                                   atol=0, rtol=0)
        self.assertTrue(torch.equal(strided, original))
        saved = actual.clone()
        strided.zero_()
        self.assertTrue(torch.equal(actual.view(torch.int16), saved.view(torch.int16)))

    def test_cpu_numpy_rejects_same_nonfinite_scales_and_overflow(self):
        import numpy as np

        for scale in (math.nan, math.inf, -math.inf, 32768., -32768., 65504., -65504.):
            raw = bytearray(struct.pack("<e", scale) + bytes([255] * 32))
            packed = torch.frombuffer(raw, dtype=torch.uint8).reshape(1, 34)
            for decode in (gguf._dequantize_pq2_rows_torch,
                           lambda p: gguf._dequantize_pq2_rows_numpy(p, np)):
                with self.subTest(scale=scale, decode=decode), self.assertRaises(ValueError):
                    decode(packed)

    def test_cpu_dispatch_fallback_and_accelerator_route(self):
        raw = bytearray(struct.pack("<e", .25) + bytes([0xe4] * 32))
        packed = torch.frombuffer(raw, dtype=torch.uint8).reshape(1, 34)
        expected = gguf._dequantize_pq2_rows_torch(packed)
        with patch.object(gguf, "_dequantize_pq2_rows_numpy",
                          wraps=gguf._dequantize_pq2_rows_numpy) as numpy_decode:
            self.assertTrue(torch.equal(gguf.dequantize_pq2_rows(packed), expected))
            self.assertEqual(numpy_decode.call_count, 1)
            with patch.object(torch, "get_num_threads", return_value=4):
                gguf.dequantize_pq2_rows(packed.repeat(2, 1))
                self.assertEqual(numpy_decode.call_count, 1)
            with patch.object(torch, "get_num_threads", return_value=1):
                gguf.dequantize_pq2_rows(packed.repeat(2, 1))
                self.assertEqual(numpy_decode.call_count, 1)
        with patch.dict(sys.modules, {"numpy": None}):
            self.assertTrue(torch.equal(gguf.dequantize_pq2_rows(packed), expected))
        # Dispatch evidence only: no GPU execution is needed for this assertion.
        noncpu = torch.empty(1, 34, dtype=torch.uint8, device="meta")
        with (patch.object(gguf, "_dequantize_pq2_rows_numpy") as numpy_decode,
              patch.object(gguf, "_dequantize_pq2_rows_torch", return_value=expected) as decode):
            self.assertIs(gguf.dequantize_pq2_rows(noncpu), expected)
            decode.assert_called_once_with(noncpu)
            numpy_decode.assert_not_called()

    def test_cpu_numpy_loader_readout_gradient_and_frozen_precision(self):
        import numpy as np

        _fixture(self.path)
        reference = self.load()
        # Exercise the bounded NumPy codec through the real tiny loader as
        # well; production batches deliberately retain the faster Torch route.
        with patch.object(gguf, "dequantize_pq2_rows",
                          side_effect=lambda p: gguf._dequantize_pq2_rows_numpy(p, np)) as decode:
            actual = self.load()
            ids = torch.tensor([3, 1, 0])
            torch.testing.assert_close(actual.embed_tokens(ids), reference.embed_tokens(ids),
                                       atol=0, rtol=0)
            self.assertGreaterEqual(decode.call_count, 3)
        for name, value in actual.named_buffers():
            torch.testing.assert_close(value, dict(reference.named_buffers())[name], atol=0, rtol=0)
        x = torch.linspace(-.1, .2, 5120).reshape(1, -1).requires_grad_()
        y = x.detach().clone().requires_grad_()
        logits, ref_logits = actual.lm_head(x), reference.lm_head(y)
        torch.testing.assert_close(logits, ref_logits, atol=0, rtol=0)
        logits.square().sum().backward()
        ref_logits.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
        saved = {name: value.clone() for name, value in actual.named_buffers()}
        actual.to(device="cpu", dtype=torch.bfloat16)
        for name, value in actual.named_buffers():
            self.assertEqual(value.dtype, saved[name].dtype)
            torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
        self.assertEqual(list(actual.parameters()), [])


if __name__ == "__main__":
    unittest.main()
