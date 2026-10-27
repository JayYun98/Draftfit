"""Reviewed Prism pack metadata shared by MLX capture and Torch readout."""
from __future__ import annotations

import re


def validate_prism_config(config):
    if (not isinstance(config, dict)
            or config.get("model_type") != "prism_hadamard_qwen35"
            or type(config.get("schema_version")) is not int
            or config["schema_version"] not in (1, 2)):
        raise ValueError("unsupported Bonsai pack schema")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("Bonsai text_config must be an object")
    for key in ("hidden_size", "vocab_size"):
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ValueError(f"Bonsai {key} must be a positive integer")
    if text["hidden_size"] % 128:
        raise ValueError("Bonsai hidden_size must be divisible by 128")
    if "num_hidden_layers" in text and (
        type(text["num_hidden_layers"]) is not int or text["num_hidden_layers"] <= 0
    ):
        raise ValueError("Bonsai num_hidden_layers must be a positive integer")
    if text.get("tie_word_embeddings", False) is not False:
        raise ValueError("Bonsai packed tied readout is unsupported")
    records = config.get("modules")
    if not isinstance(records, list) or not 1 <= len(records) <= 8192:
        raise ValueError("Bonsai modules must be a nonempty bounded list")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid packed module record")
        name = record.get("path")
        if (not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.(?:[A-Za-z][A-Za-z0-9_]*|[0-9]+))*", name)
                or name in seen):
            raise ValueError("duplicate/invalid packed module path")
        seen.add(name)
        if (type(record.get("embedding")) is not bool
                or record.get("dtype") != "float16"
                or type(record.get("block")) is not int
                or record["block"] not in (0, 512, 1024, 2048, 4096)):
            raise ValueError("invalid packed module metadata")
    if not {"model.embed_tokens", "lm_head"}.issubset(seen):
        raise ValueError("Bonsai modules must declare embedding and untied readout")
    if config["schema_version"] == 2:
        if (config.get("tensor_namespace") != "mlx-vlm-qwen3_5"
                or config.get("gdn_activation_layout") != "grouped"
                or config.get("base_model_type") != "qwen3_5"
                or config.get("quantization") != {"bits": 2, "group_size": 128, "mode": "affine"}
                or config.get("components") != {"text": True, "vision": True, "mtp": False}):
            raise ValueError("unsupported Bonsai schema 2 tensor/layout contract")
