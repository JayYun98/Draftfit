"""Checks required before training from official DSpark target-tensor copies."""

from __future__ import annotations

import os
import json


def is_official_taichu_source(source: str) -> bool:
    if source.startswith("TaichuAI/ZDTaichu5.0-9B-DSpark"):
        return True
    if source.startswith("file://"):
        source = source[len("file://"):]
    if not os.path.isdir(source):
        return False
    config_path = os.path.join(source, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path, encoding="utf-8") as stream:
        return "transformer_layer_config" in json.load(stream)


def verify_taichu_target_copies(source: str, target_parts, *, cache_dir=None) -> None:
    """Compare the official draft's extra embedding/head with loaded target weights.

    Safetensors slices avoid a second multi-gigabyte copy of either matrix.
    A mismatch means the training run would not start from the official model.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if source.startswith("file://"):
        source = source[len("file://"):]
    if os.path.isdir(source):
        path = os.path.join(source, "model.safetensors")
    else:
        path = hf_hub_download(source, "model.safetensors", cache_dir=cache_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"official Taichu draft weights not found: {path}")

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key, target in (
            ("embed_tokens.weight", target_parts.embed_tokens.weight),
            ("lm_head.weight", target_parts.lm_head.weight),
        ):
            if key not in checkpoint.keys():
                raise ValueError(f"official Taichu draft is missing {key}")
            source_tensor = checkpoint.get_slice(key)
            if tuple(source_tensor.get_shape()) != tuple(target.shape):
                raise ValueError(f"official Taichu {key} shape differs from target")
            for start in range(0, target.shape[0], 512):
                end = min(start + 512, target.shape[0])
                actual = source_tensor[start:end]
                expected = target[start:end].detach().cpu()
                if not torch.equal(actual.to(expected.dtype), expected):
                    raise ValueError(
                        f"official Taichu {key} differs from target at rows {start}:{end}"
                    )
