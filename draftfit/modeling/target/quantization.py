"""Quantized teacher identity shared by capture, training, and resume checks."""

from __future__ import annotations

import json
from pathlib import Path

TARGET_QUANTIZATION_RECIPES = ("auto", "bitsandbytes-4bit", "bitsandbytes-8bit")


def require_floating_target_weight(tensor, key):
    import torch

    if tensor.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(
            f"Target embedding/head {key} must be floating-point weights; "
            "packed embeddings or output heads are not supported"
        )
    return tensor


def target_quantization_identity(config, recipe="auto"):
    """Describe the actual teacher load without importing a quantization runtime.

    ``auto`` preserves checkpoint metadata; it never requests dequantization.
    Explicit recipes apply to floating checkpoints and quantize decoder linears.
    Embeddings and the output head must remain floating point for draft training.
    """
    if recipe not in TARGET_QUANTIZATION_RECIPES:
        raise ValueError(f"unsupported target quantization recipe: {recipe!r}")
    checkpoint = getattr(config, "quantization_config", None)
    if checkpoint is None:
        text = getattr(config, "text_config", None)
        checkpoint = (
            text.get("quantization_config")
            if isinstance(text, dict)
            else getattr(text, "quantization_config", None)
        )
    if hasattr(checkpoint, "to_dict"):
        checkpoint = checkpoint.to_dict()
    if checkpoint is not None and not isinstance(checkpoint, dict):
        raise ValueError("target quantization_config must be an object")
    if recipe != "auto" and checkpoint:
        raise ValueError("target is already quantized; use target_quantization=auto")
    if recipe == "auto" and not checkpoint:
        return None
    return {
        "recipe": recipe,
        "checkpoint_config": json.loads(json.dumps(checkpoint, default=str)),
    }


def validate_quantized_feature_source(cfg, target_config):
    """Reject stale or unlabelled features before allocating a draft or optimizer."""
    from draftfit.inference.capture_manifest import load_capture_manifest

    expected = target_quantization_identity(
        target_config, cfg.model.target_quantization
    )
    for source in (cfg.data.hidden_states_path, cfg.data.eval_hidden_states_path):
        if not source:
            continue
        root = Path(source).expanduser()
        if root.is_file():
            root = root.parent
        path = root / "capture_manifest.json"
        if not path.is_file():
            if expected is not None:
                raise ValueError(
                    "quantized-target training requires capture_manifest.json; recapture features"
                )
            continue
        manifest = load_capture_manifest(str(path))
        actual = manifest["target"].get("quantization")
        if actual != expected:
            raise ValueError(
                "capture target quantization differs from training; recapture features"
            )
        if expected is None:
            continue
        target = manifest["target"]
        revision = (
            getattr(target_config, "_commit_hash", None) or cfg.model.target_revision
        )
        if target["model"] != cfg.model.target_model_path or (
            revision is not None and target["revision"] != revision
        ):
            raise ValueError(
                "quantized capture target model/revision differs from training"
            )
        if manifest.get("producer", {}).get("backend") != cfg.model.target_backend:
            raise ValueError("quantized capture backend differs from training")
        settings = manifest.get("producer", {}).get("settings", {})
        if settings.get("dtype") != cfg.model.torch_dtype:
            raise ValueError("quantized capture compute dtype differs from training")
        if (
            cfg.model.target_backend == "transformers"
            and settings.get("attention_implementation")
            != cfg.model.hf_attn_implementation
        ):
            raise ValueError(
                "quantized capture attention implementation differs from training"
            )
