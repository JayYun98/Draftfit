"""Identify genuine Taichu exports and verify published target-tensor copies."""

from __future__ import annotations

import json
import os


def is_taichu_config(payload: dict) -> bool:
    """A nested Qwen config alone does not identify the Taichu release."""
    if payload.get("official_dspark_format") == "taichu_speculators":
        return True
    transformer = payload.get("transformer_layer_config") or {}
    return (
        payload.get("architectures") == ["DSparkDraftModel"]
        and payload.get("aux_hidden_state_layer_ids") == [1, 8, 15, 22, 29]
        and transformer.get("hidden_size") == 4096
        and transformer.get("vocab_size") == 248320
        and transformer.get("num_hidden_layers") == 5
        and transformer.get("head_dim") == 256
    )


def is_official_taichu_source(source: str) -> bool:
    if source.startswith("TaichuAI/ZDTaichu5.0-9B-DSpark"):
        return True
    source = os.path.expanduser(source.removeprefix("file://"))
    config_path = (
        os.path.join(source, "config.json") if os.path.isdir(source) else source
    )
    if not os.path.isfile(config_path):
        return False
    with open(config_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    # A native re-save contains only draft tensors. It is not the original
    # nested export carrying the mandatory target copies.
    return "transformer_layer_config" in payload and is_taichu_config(payload)


def verify_target_copies(
    source: str, target_parts, *, cache_dir=None, require_both: bool = False
) -> None:
    """Check every saved target copy exactly, without lossy dtype conversion.

    Generic exports may omit either copy. Genuine Taichu exports require both.
    Safetensors slices avoid another full embedding/head allocation.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    source = os.path.expanduser(source.removeprefix("file://"))
    if os.path.isdir(source):
        path = os.path.join(source, "model.safetensors")
    else:
        path = hf_hub_download(source, "model.safetensors", cache_dir=cache_dir)
    label = "official Taichu" if require_both else "published speculators"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} draft weights not found: {path}")

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key, target in (
            ("embed_tokens.weight", target_parts.embed_tokens.weight),
            ("lm_head.weight", target_parts.lm_head.weight),
        ):
            if key not in checkpoint.keys():
                if require_both:
                    raise ValueError(f"{label} draft is missing {key}")
                continue
            source_tensor = checkpoint.get_slice(key)
            if tuple(source_tensor.get_shape()) != tuple(target.shape):
                raise ValueError(f"{label} {key} shape differs from target")
            for start in range(0, target.shape[0], 512):
                end = min(start + 512, target.shape[0])
                actual = source_tensor[start:end]
                expected = target[start:end].detach().cpu()
                dtype = torch.promote_types(actual.dtype, expected.dtype)
                if not torch.equal(actual.to(dtype), expected.to(dtype)):
                    raise ValueError(
                        f"{label} {key} differs from target at rows {start}:{end}"
                    )


def verify_taichu_target_copies(source: str, target_parts, *, cache_dir=None) -> None:
    """Keep the genuine Taichu release's mandatory embedding/head checks."""
    verify_target_copies(source, target_parts, cache_dir=cache_dir, require_both=True)
