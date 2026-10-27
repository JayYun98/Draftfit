"""Frozen Bonsai readout for the existing PyTorch family trainer.

Only the embedding and output projection are loaded. The teacher's packed
backbone remains in its capture engine. No bare rotated weight fallback.
CUDA numerical parity and GGUF serving remain separate validation gates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def hadamard(x, block, signs, *, inverse=False):
    if not block:
        return x
    shape, dtype = x.shape, x.dtype
    if (
        type(block) is not int
        or block < 1
        or block & (block - 1)
        or not shape
        or shape[-1] % block
        or signs is None
        or tuple(signs.shape) != (shape[-1],)
    ):
        raise ValueError("invalid Hadamard block/sign geometry")
    x = x.float()
    if not inverse:
        x = x * signs.float()
    x = x.reshape(-1, block)
    step = 1
    while step < block:
        paired = x.reshape(-1, block // (2 * step), 2, step)
        a, b = paired.unbind(-2)
        x = torch.stack((a + b, a - b), dim=-2).reshape(-1, block)
        step *= 2
    x = (x / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs.float()
    return x.to(dtype)


def dequantize_rows(weight, scales, biases):
    if (
        weight.ndim != 2
        or weight.dtype != torch.uint32
        or weight.shape[1] % 8
        or tuple(scales.shape) != (weight.shape[0], weight.shape[1] // 8)
        or biases.shape != scales.shape
    ):
        raise ValueError("invalid packed affine row geometry/storage")
    codes = (
        (
            weight.to(torch.int64)[..., None]
            >> (2 * torch.arange(16, device=weight.device))
        )
        & 3
    ).reshape(weight.shape[0], weight.shape[1] // 8, 128)
    dense = (
        (codes.float() * scales.float()[..., None] + biases.float()[..., None])
        .reshape(weight.shape[0], weight.shape[1] * 16)
        .half()
    )
    if not torch.isfinite(dense).all():
        raise ValueError("non-finite packed affine values or FP16 readout overflow")
    return dense


class FrozenPacked(nn.Module):
    """Keep pack storage/rounding in FP16 when the trainer casts its modules."""

    def _apply(self, fn, recurse=True):
        self.output_dtype = fn(torch.empty(0, dtype=torch.float16)).dtype
        for name, value in self._buffers.items():
            if value is not None:
                destination = fn(
                    torch.empty(0, device=value.device, dtype=torch.float16)
                ).device
                self._buffers[name] = value.to(device=destination)
        return self


class HadamardEmbedding(FrozenPacked):
    def __init__(self, arrays, *, block, signs):
        super().__init__()
        for name, value in zip(("packed_weight", "scales", "biases"), arrays):
            self.register_buffer(name, value)
        self.register_buffer("signs", signs)
        self.block = block
        self.num_embeddings = arrays[0].shape[0]
        self.embedding_dim = arrays[0].shape[1] * 16

    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("embedding IDs must be integers")
        if torch.any((ids < 0) | (ids >= self.num_embeddings)):
            raise ValueError("embedding IDs outside vocabulary")
        if not ids.numel():
            return torch.empty(
                *ids.shape,
                self.embedding_dim,
                device=ids.device,
                dtype=getattr(self, "output_dtype", torch.float16),
            )
        flat = ids.reshape(-1)
        # uint32 advanced indexing is not implemented by all PyTorch devices.
        selected = self.packed_weight.view(torch.int32)[flat].view(torch.uint32)
        rows = dequantize_rows(selected, self.scales[flat], self.biases[flat])
        out = hadamard(rows, self.block, self.signs, inverse=True).reshape(
            *ids.shape, self.embedding_dim
        )
        return out.to(getattr(self, "output_dtype", torch.float16))


class HadamardHead(FrozenPacked):
    def __init__(self, arrays, *, block, signs):
        super().__init__()
        rows, words = arrays[0].shape
        dense = torch.empty(rows, words * 16, dtype=torch.float16)
        for start in range(0, rows, 512):
            stop = min(rows, start + 512)
            dense[start:stop] = dequantize_rows(*(x[start:stop] for x in arrays))
        self.register_buffer("weight", dense)
        self.register_buffer("signs", signs)
        self.block = block
        self.in_features, self.out_features = dense.shape[1], dense.shape[0]

    def forward(self, hidden):
        # Preserve the source's activation transform and its FP16 rounding.
        transformed = hadamard(hidden.half(), self.block, self.signs)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(transformed, self.weight).to(hidden.dtype)


def load_target_parts(model_path, *, embedding_only=False, **kwargs):
    """Normal target loading remains unchanged; Bonsai is an explicit local lane."""
    path = Path(model_path)
    if path.suffix.lower() == ".gguf" and path.is_file():
        if kwargs.get("embed_key") not in (
            None,
            "model.embed_tokens.weight",
        ) or kwargs.get("lm_head_key") not in (None, "lm_head.weight"):
            raise ValueError("Bonsai readout keys are fixed by the pack contract")
        from .gguf_parts import load_target_parts as load_gguf_parts

        return load_gguf_parts(
            model_path,
            embedding_only=embedding_only,
            device=kwargs.get("device", "cpu"),
            dtype=kwargs.get("dtype", torch.float16),
        )
    from .target_utils import TargetEmbeddingsAndHead

    config_path = path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get("model_type") != "prism_hadamard_qwen35":
        return TargetEmbeddingsAndHead.from_pretrained(model_path, **kwargs)
    if kwargs.get("embed_key") not in (None, "model.embed_tokens.weight") or kwargs.get(
        "lm_head_key"
    ) not in (None, "lm_head.weight"):
        raise ValueError("Bonsai readout keys are fixed by the pack contract")
    revision = kwargs.get("revision")
    if revision and path.parent.name == "snapshots" and path.name != revision:
        raise ValueError("Bonsai target revision differs from snapshot")
    from draftfit.offline_capture.prism_config import validate_prism_config

    validate_prism_config(config)
    if kwargs.get("dtype", torch.float16) not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("unsupported packed readout output dtype")
    records = {r["path"]: r for r in config["modules"]}
    if len(records) != len(config["modules"]):
        raise ValueError("duplicate packed records")
    prefix = "language_model." if config.get("components", {}).get("vision") else ""
    from safetensors import safe_open

    modules = []
    with safe_open(
        str(path / "model.safetensors"), framework="pt", device="cpu"
    ) as source:
        for name, kind, cls in (
            ("model.embed_tokens", True, HadamardEmbedding),
            ("lm_head", False, HadamardHead),
        ):
            if embedding_only and not kind:
                continue
            record = records[name]
            block = record["block"]
            if (
                record["embedding"] is not kind
                or record["dtype"] != "float16"
                or type(block) is not int
                or block not in (0, 512, 1024, 2048, 4096)
            ):
                raise ValueError("invalid packed readout metadata")
            keys = [prefix + name + "." + k for k in ("weight", "scales", "biases")]
            if any(key not in source.keys() for key in keys):
                raise ValueError(f"missing packed readout tensor for {name}")
            arrays = [source.get_tensor(key) for key in keys]
            text = config["text_config"]
            rows, width = text["vocab_size"], text["hidden_size"]
            if (
                width % 128
                or [tuple(x.shape) for x in arrays]
                != [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
                or arrays[0].dtype != torch.uint32
            ):
                raise ValueError("invalid packed readout shapes/storage")
            if any(
                x.dtype != torch.float16 or not torch.isfinite(x).all()
                for x in arrays[1:]
            ):
                raise ValueError("invalid packed affine dtype/values")
            sign_key = prefix + name + ".signs"
            signs = source.get_tensor(sign_key) if sign_key in source.keys() else None
            if block and (
                width % block
                or signs is None
                or tuple(signs.shape) != (width,)
                or not torch.all((signs == 1) | (signs == -1))
            ):
                raise ValueError("invalid readout Hadamard signs")
            if not block and signs is not None:
                raise ValueError("unexpected readout signs")
            if kind:
                for start in range(0, rows, 512):
                    dequantize_rows(*(x[start : start + 512] for x in arrays))
            modules.append(cls(arrays, block=block, signs=signs))
    parts = nn.Module()
    parts.embed_tokens = modules[0]
    if not embedding_only:
        parts.lm_head = modules[1]
    parts.requires_grad_(False).eval().to(
        device=kwargs.get("device", "cpu"), dtype=kwargs.get("dtype", torch.float16)
    )
    return parts


@torch.no_grad()
def export_embedding(model_path):
    embedding = load_target_parts(
        model_path, embedding_only=True, device="cpu"
    ).embed_tokens
    result = torch.empty(
        embedding.num_embeddings, embedding.embedding_dim, dtype=torch.float16
    )
    for start in range(0, embedding.num_embeddings, 512):
        stop = min(embedding.num_embeddings, start + 512)
        result[start:stop] = embedding(torch.arange(start, stop))
    return result
