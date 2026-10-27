"""Reviewed Bonsai text loader, adapted from Prism's pinned MLX runtime.

Packed/fwht: Copyright © 2023 Apple Inc., MIT (PRISM_RUNTIME_LICENSE).
Source: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit runtime/runtime.py,
revision fcba37d2117a7077eac6b613b2668d14d9779edd.
Changes: strict text-only loading, shape/finite checks, no remote Python execution.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def fwht(x, block, signs, inverse=False):
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = arrays
        self.block, self.signs, self.embedding, self.dtype = block, signs, embedding, dtype

    def __call__(self, x):
        if self.embedding:
            out = mx.dequantize(self.weight[x.reshape(-1)], self.scales[x.reshape(-1)],
                                self.biases[x.reshape(-1)], group_size=128, bits=2)
            out = out.reshape(*x.shape, -1).astype(self.dtype)
            return fwht(out, self.block, self.signs, inverse=True) if self.block else out
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                  transpose=True, group_size=128, bits=2)

    def dense_rows(self, start, stop):
        """Undo the weight basis; embedding lookup is exact, linear readout approximate.

        The packed head rounds transformed activations before its matmul. A dense
        reconstructed head does not preserve that rounding and needs a parity gate.
        """
        out = mx.dequantize(self.weight[start:stop], self.scales[start:stop],
                            self.biases[start:stop], group_size=128, bits=2)
        return fwht(out, self.block, self.signs, inverse=True) if self.block else out


def install_packed(model, records, weights):
    seen = set()
    for record in records:
        name = record["path"]
        if not isinstance(name, str) or name in seen:
            raise ValueError("duplicate/invalid packed module path")
        seen.add(name)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        leaf = parts[-1]
        original = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        if not isinstance(original, (nn.Linear, nn.Embedding)):
            raise ValueError("unsupported packed module target")
        if type(record["embedding"]) is not bool or record["embedding"] != isinstance(original, nn.Embedding):
            raise ValueError("packed embedding/linear kind mismatch")
        rows, width = original.weight.shape
        arrays = [weights[name + "." + k] for k in ("weight", "scales", "biases")]
        expected = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
        if width % 128 or [x.shape for x in arrays] != expected or arrays[0].dtype != mx.uint32:
            raise ValueError("invalid packed tensor shape or dtype")
        if any(x.dtype not in (mx.float16, mx.float32, mx.bfloat16) or not mx.all(mx.isfinite(x)).item() for x in arrays[1:]):
            raise ValueError("invalid/non-finite packed affine parameters")
        block, signs = record["block"], weights.get(name + ".signs")
        if record["dtype"] != "float16" or type(block) is not int or block not in (0, 512, 1024, 2048, 4096):
            raise ValueError("unsupported packed activation dtype or block")
        if block and (width % block or signs is None or signs.shape != (width,)
                      or not mx.all((signs == 1) | (signs == -1)).item()):
            raise ValueError("invalid Hadamard sign vector/dimensions")
        if not block and signs is not None:
            raise ValueError("unexpected sign vector")
        replacement = Packed(arrays, block, signs, record["embedding"])
        if leaf.isdigit():
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)


def load_bonsai_text(path):
    """Retain the native VLM text stack and all text tensors, omit vision only."""
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    from .prism_config import validate_prism_config
    validate_prism_config(config)
    model = LanguageModel(TextConfig.from_dict(copy.deepcopy(config["text_config"])))
    weights = mx.load(str(path / "model.safetensors"))
    if config.get("components", {}).get("vision"):
        unknown = [key for key in weights if not key.startswith(("language_model.", "vision_tower."))]
        if unknown:
            raise ValueError(f"unknown non-text weight prefixes: {unknown[:3]}")
        weights = {key.removeprefix("language_model."): value for key, value in weights.items()
                   if key.startswith("language_model.")}
    install_packed(model, config["modules"], weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
