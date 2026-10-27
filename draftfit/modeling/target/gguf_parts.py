"""Explicit local Bonsai GGUF v3 readout; no model loader or GGUF dependency.

PQ2_0 layout follows Prism llama.cpp e3de715222b2667c54a267441159a5729f11155e,
ggml-common.h/ggml-quants.c: 128 values = little-endian FP16 scale + 32
bytes, four consecutive low-to-high 2-bit codes per byte, decoded as q - 1.
Head weights are dequantized once to FP16; non-finite scales/overflow fail.
The head casts inputs to FP16, runs the existing FP32-accumulating normalized
Hadamard helper with FP16 output rounding, then performs an FP16 linear readout.
Embedding rows are dequantized to FP16 before inverse Hadamard and FP16 rounding.
These are training rounding choices, not a claim of native F32/Metal parity.

Usage: load_target_parts('/immutable/path/model.gguf', device='cpu').
The caller owns immutable file identity and provenance; parsing verifies neither
SHA nor authenticity. No remote paths, repository model code, or capture bridge
are loaded. Only the two frozen readout modules are constructed.
"""

from __future__ import annotations

import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .quantized_parts import FrozenPacked, hadamard

_VOCAB_SIZE = 248320
_HIDDEN_SIZE = 5120
_SIGN_WIDTHS = [5120, 6144, 17408]
_PREFIX = "prism.hadamard."
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}
_META_TYPES = {
    "general.architecture": 8,
    "general.alignment": 4,
    "split.count": 2,
    "qwen35.embedding_length": 4,
    "qwen35.vocab_size": 4,
    "qwen35.block_count": 4,
    "qwen35.tie_word_embeddings": 7,
    "tokenizer.ggml.tokens": (9, 8),
    "tokenizer.ggml.merges": (9, 8),
    "tokenizer.ggml.eos_token_id": 4,
    "tokenizer.ggml.bos_token_id": 4,
    "tokenizer.ggml.padding_token_id": 4,
    **{
        _PREFIX + key: typ
        for key, typ in {
            "version": 4,
            "block_size": 4,
            "transform": 8,
            "axis": 8,
            "sign_mode": 8,
            "weight_names": (9, 8),
            "sign_widths": (9, 5),
            "sign_values": (9, 5),
            "inverse_weight_names": (9, 8),
            "tied_output": 7,
            "gdn_v_grouped": 7,
        }.items()
    },
}
# Only storage layouts needed by this pinned family; unknown tensor types fail.
_STORAGE = {0: (1, 4), 1: (1, 2), 30: (1, 2), 142: (128, 34), 143: (128, 28)}


@dataclass(frozen=True)
class TensorInfo:
    dims: tuple[int, ...]  # GGUF order: contiguous width first
    type: int
    offset: int  # relative to data_offset
    nbytes: int


@dataclass(frozen=True)
class GGUFMetadata:
    metadata: dict
    tensors: dict[str, TensorInfo]
    data_offset: int
    file_size: int


class _Reader:
    """Bounded stdlib parser; irrelevant metadata is consumed without retention."""

    def __init__(self, source, include_tokenizer):
        self.source = source
        self.size = os.fstat(source.fileno()).st_size
        self.limit = min(self.size, 64 << 20)
        self.items_left = 1_000_000
        self.include_tokenizer = include_tokenizer

    def read(self, count):
        if count < 0 or count > self.limit - self.source.tell():
            raise ValueError("truncated GGUF or metadata exceeds 64 MiB limit")
        value = self.source.read(count)
        if len(value) != count:
            raise ValueError("truncated GGUF")
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, *, name=False):
        count = self.number("Q")
        if count > (1024 if name else 8 << 20):
            raise ValueError("GGUF string length exceeds limit")
        try:
            value = self.read(count).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid GGUF UTF-8") from exc
        if name and (not value or "\0" in value):
            raise ValueError("invalid GGUF name")
        return value

    def scalar(self, typ):
        if typ == 8:
            return self.string()
        if typ not in _SCALARS:
            raise ValueError(f"unsupported GGUF metadata type {typ}")
        value = self.number(_SCALARS[typ])
        if typ == 7:
            if value not in (0, 1):
                raise ValueError("invalid GGUF bool")
            return bool(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite GGUF metadata")
        return value

    def value(self, key, typ):
        expected = _META_TYPES.get(key)
        if typ != 9:
            if expected is not None and typ != expected:
                raise ValueError(f"wrong metadata type for {key}")
            return self.scalar(typ)
        element, count = self.number("I"), self.number("Q")
        if element not in (*_SCALARS, 8):
            raise ValueError("nested/unknown GGUF array type")
        if expected is not None and expected != (9, element):
            raise ValueError(f"wrong metadata array type for {key}")
        self.items_left -= count
        if self.items_left < 0:
            raise ValueError("GGUF metadata array count exceeds limit")
        # Normally only vocabulary count is needed. Retention is explicitly opt-in.
        tokenizer = key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
        keep = expected is not None and (not tokenizer or self.include_tokenizer)
        if keep and count > (262144 if tokenizer else 32768):
            raise ValueError("readout metadata array exceeds limit")
        values = [] if keep else None
        for _ in range(count):
            value = self.scalar(element)
            if keep:
                values.append(value)
        return values if keep else count


def _read_metadata(source, *, include_tokenizer=False):
    reader = _Reader(source, include_tokenizer)
    if reader.read(4) != b"GGUF" or reader.number("I") != 3:
        raise ValueError("only little-endian GGUF v3 is supported")
    tensors_count, metadata_count = reader.number("Q"), reader.number("Q")
    if not 1 <= tensors_count <= 8192 or not 1 <= metadata_count <= 4096:
        raise ValueError("GGUF tensor/metadata count exceeds bounds")
    metadata, seen = {}, set()
    for _ in range(metadata_count):
        key, typ = reader.string(name=True), reader.number("I")
        if key in seen:
            raise ValueError(f"duplicate GGUF metadata: {key}")
        seen.add(key)
        if key.startswith("prism.") and key not in _META_TYPES:
            raise ValueError(f"unrecognized mandatory Prism metadata: {key}")
        value = reader.value(key, typ)
        if key in _META_TYPES and (key != "tokenizer.ggml.merges" or include_tokenizer):
            metadata[key] = value
    alignment = metadata.get("general.alignment", 32)
    if not 1 <= alignment <= 4096 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    tensors = {}
    for _ in range(tensors_count):
        name, rank = reader.string(name=True), reader.number("I")
        if name in tensors or not 1 <= rank <= 4:
            raise ValueError("duplicate GGUF tensor or invalid rank")
        dims = tuple(reader.number("Q") for _ in range(rank))
        typ, offset = reader.number("I"), reader.number("Q")
        if any(not 1 <= dim <= 1 << 24 for dim in dims) or math.prod(dims) > 1 << 40:
            raise ValueError("GGUF tensor dimensions exceed bounds")
        if typ not in _STORAGE:
            raise ValueError(f"unsupported GGUF tensor storage type {typ}")
        group, size = _STORAGE[typ]
        if dims[0] % group or offset % alignment:
            raise ValueError("invalid GGUF tensor block/alignment")
        tensors[name] = TensorInfo(dims, typ, offset, math.prod(dims) // group * size)
    data_offset = (source.tell() + alignment - 1) // alignment * alignment
    end = data_offset
    for info in sorted(tensors.values(), key=lambda t: t.offset):
        start = data_offset + info.offset
        if start < end or info.nbytes > reader.size - start:
            raise ValueError("overlapping or out-of-file GGUF tensor data")
        end = start + info.nbytes
    return GGUFMetadata(metadata, tensors, data_offset, reader.size)


def read_gguf_metadata(model_path, *, include_tokenizer=False):
    """Inspect metadata/spans without allocating weights or asserting identity.

    By default ``metadata['tokenizer.ggml.tokens']`` is the vocabulary count.
    With ``include_tokenizer=True``, tokens and merges are full ordered string
    lists, suitable for the caller's exact tokenizer comparison. BOS/EOS/padding
    IDs are retained when present; callers must prefer target EOS over draft EOS.
    """
    with Path(model_path).open("rb") as source:
        return _read_metadata(source, include_tokenizer=include_tokenizer)


def _validate_bonsai(header):
    meta = header.metadata
    required = {
        "general.architecture": "qwen35",
        "qwen35.embedding_length": _HIDDEN_SIZE,
        "qwen35.block_count": 64,
        "tokenizer.ggml.tokens": _VOCAB_SIZE,
        _PREFIX + "version": 1,
        _PREFIX + "block_size": 1024,
        _PREFIX + "transform": "normalized-sylvester-walsh-hadamard",
        _PREFIX + "axis": "input-last-dimension",
        _PREFIX + "sign_mode": "explicit",
        _PREFIX + "sign_widths": _SIGN_WIDTHS,
        _PREFIX + "gdn_v_grouped": True,
        _PREFIX + "inverse_weight_names": ["token_embd.weight"],
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"unsupported Bonsai metadata: {key}")
    if (
        meta.get(_PREFIX + "tied_output", False)
        or meta.get("qwen35.tie_word_embeddings", False)
        or meta.get("qwen35.vocab_size", _VOCAB_SIZE) != _VOCAB_SIZE
        or meta.get("split.count", 1) != 1
    ):
        raise ValueError("tied, split or mismatched Bonsai readout is unsupported")
    names = meta.get(_PREFIX + "weight_names", [])
    if (
        not names
        or len(set(names)) != len(names)
        or "output.weight" not in names
        or "token_embd.weight" in names
        or any(n not in header.tensors for n in names)
    ):
        raise ValueError("invalid explicit forward Hadamard weights")
    values = meta.get(_PREFIX + "sign_values", [])
    if len(values) != sum(_SIGN_WIDTHS) or any(v not in (-1, 1) for v in values):
        raise ValueError("invalid explicit Hadamard signs")
    for name in ("output.weight", "token_embd.weight"):
        info = header.tensors.get(name)
        if info is None or info.type != 142 or info.dims != (_HIDDEN_SIZE, _VOCAB_SIZE):
            raise ValueError(f"invalid PQ2_0 readout: {name}")
    return values[:_HIDDEN_SIZE]


def dequantize_pq2_rows(packed):
    """Decode uint8 [rows, groups * 34] to FP16 [rows, groups * 128].

    NumPy avoids Torch dispatch overhead for a single CPU embedding row.
    Batched decoding remains on Torch: bounded probes did not show a stable
    NumPy advantage for 512 rows. Accelerator decoding is unchanged.
    """
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] % 34:
        raise ValueError("invalid PQ2_0 row storage")
    if sys.byteorder != "little":
        raise ValueError("PQ2_0 Torch decoding requires a little-endian host")
    if packed.device.type == "cpu" and packed.shape[0] == 1:
        try:
            import numpy as np
        except ImportError:
            pass  # Keep the Torch decoder usable without NumPy.
        else:
            return _dequantize_pq2_rows_numpy(packed, np)
    return _dequantize_pq2_rows_torch(packed)


def _dequantize_pq2_rows_numpy(packed, np):
    """Bound scratch to 512 rows; FP32 multiply then one FP16 rounding.

    The result owns its storage and never aliases the compact input. This
    decoder does not change Torch's process-wide thread or precision settings.
    """
    rows, groups = packed.shape[0], packed.shape[1] // 34
    dense = np.empty((rows, groups * 128), dtype=np.float16)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    for start in range(0, rows, 512):
        blocks = packed[start : start + 512].numpy().reshape(-1, groups, 34)
        scales = blocks[..., :2].copy().view("<f2").astype(np.float32)
        if not np.isfinite(scales).all():
            raise ValueError("non-finite PQ2_0 scale")
        codes = ((blocks[..., 2:, None] >> shifts) & 3).reshape(-1, groups, 128)
        values = codes.astype(np.float32)
        values -= np.float32(1)
        values *= scales
        with np.errstate(over="ignore"):
            chunk = values.reshape(len(blocks), groups * 128).astype(np.float16)
        if not np.isfinite(chunk).all():
            raise ValueError("PQ2_0 values overflow FP16 readout")
        dense[start : start + len(blocks)] = chunk
    return torch.from_numpy(dense)


def _dequantize_pq2_rows_torch(packed):
    # Preserve the original accelerator/batched decoder and its FP16 contract.
    blocks = packed.reshape(packed.shape[0], -1, 34)
    scales = blocks[..., :2].contiguous().view(torch.float16).float()
    if not torch.isfinite(scales).all():
        raise ValueError("non-finite PQ2_0 scale")
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    codes = ((blocks[..., 2:, None] >> shifts) & 3).reshape(*blocks.shape[:2], 128)
    dense = ((codes.float() - 1) * scales).reshape(packed.shape[0], -1).half()
    if not torch.isfinite(dense).all():
        raise ValueError("PQ2_0 values overflow FP16 readout")
    return dense


class GGUFEmbedding(FrozenPacked):
    def __init__(self, packed, signs):
        super().__init__()
        self.register_buffer("packed_weight", packed)
        self.register_buffer("signs", signs)
        self.num_embeddings = packed.shape[0]
        self.embedding_dim = packed.shape[1] // 34 * 128

    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("embedding IDs must be integers")
        if torch.any((ids < 0) | (ids >= self.num_embeddings)):
            raise ValueError("embedding IDs outside vocabulary")
        # Preserve empty batches without an ambiguous reshape in the shared helper.
        if not ids.numel():
            return torch.empty(
                *ids.shape,
                self.embedding_dim,
                device=ids.device,
                dtype=getattr(self, "output_dtype", torch.float16),
            )
        rows = dequantize_pq2_rows(self.packed_weight[ids.reshape(-1).long()])
        result = hadamard(rows, 1024, self.signs, inverse=True)
        return result.reshape(*ids.shape, self.embedding_dim).to(
            getattr(self, "output_dtype", torch.float16)
        )


class GGUFHead(FrozenPacked):
    def __init__(self, weight, signs):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("signs", signs)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]

    def forward(self, hidden):
        transformed = hadamard(hidden.half(), 1024, self.signs)
        # Autocast must not silently change the explicitly FP16 readout contract.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(transformed, self.weight).to(hidden.dtype)


def _read_rows(source, header, name, start, count):
    info = header.tensors[name]
    row_bytes = info.nbytes // info.dims[1]
    source.seek(header.data_offset + info.offset + start * row_bytes)
    raw = bytearray(source.read(count * row_bytes))
    if len(raw) != count * row_bytes:
        raise ValueError("GGUF changed/truncated during readout load")
    return torch.frombuffer(raw, dtype=torch.uint8).reshape(count, row_bytes)


def load_target_parts(
    model_path, *, embedding_only=False, device="cpu", dtype=torch.float16
):
    """Load exact untied Bonsai v1 PQ2_0 embedding/head from an explicit GGUF file.

    No SHA is inferred or checked. Caller must keep this local file immutable.
    CPU storage is ~338 MB packed embedding plus ~2.54 GB dense FP16 head;
    head construction uses bounded 512-row decode chunks, never a full F32 head.
    ``embedding_only`` still validates both descriptors and the metadata contract.
    FrozenPacked preserves buffers across trainer .to(dtype/device) calls.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported GGUF readout output dtype")
    path = Path(model_path)
    if not path.is_file():
        raise ValueError("model_path must be an explicit local GGUF file")
    with path.open("rb") as source:
        header = _read_metadata(source)
        signs = torch.tensor(_validate_bonsai(header), dtype=torch.float16)
        packed = _read_rows(source, header, "token_embd.weight", 0, _VOCAB_SIZE)
        # Validate every scale/value in bounded chunks before exposing the module.
        for start in range(0, _VOCAB_SIZE, 512):
            dequantize_pq2_rows(packed[start : start + 512])
        parts = nn.Module()
        parts.embed_tokens = GGUFEmbedding(packed, signs.clone())
        if not embedding_only:
            dense = torch.empty(_VOCAB_SIZE, _HIDDEN_SIZE, dtype=torch.float16)
            for start in range(0, _VOCAB_SIZE, 512):
                count = min(512, _VOCAB_SIZE - start)
                dense[start : start + count] = dequantize_pq2_rows(
                    _read_rows(source, header, "output.weight", start, count)
                )
            parts.lm_head = GGUFHead(dense, signs.clone())
    return parts.requires_grad_(False).eval().to(device=device, dtype=dtype)
