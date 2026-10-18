#!/usr/bin/env python3
"""Extract a local official Taichu snapshot as a Qwen3.5 text-only target.

Usage: python scripts/export_taichu_text_target.py --source LOCAL --output NEW
Requires torch and safetensors. No downloads or remote code execution. Processes
one source shard at a time (official shards are approximately 5 GB); does not
cast, transpose or normalize tensors. CUDA/serving correctness is not verified.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "language_model."


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _expected_keys(config):
    """Exact dense Qwen3.5 text schema used by the official Taichu snapshot."""
    count = config.get("num_hidden_layers")
    types = config.get("layer_types")
    if type(count) is not int or count < 1 or not isinstance(types, list) or len(types) != count:
        raise ValueError("Invalid layer_types / num_hidden_layers")
    if config.get("attention_bias", False) or config.get("tie_word_embeddings") is not False:
        raise ValueError("Only official bias-free, untied Taichu text layout is supported")
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    common = ["input_layernorm.weight", "post_attention_layernorm.weight",
              "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    for i, kind in enumerate(types):
        if kind == "full_attention":
            suffixes = [f"self_attn.{x}.weight" for x in
                        ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        elif kind == "linear_attention":
            suffixes = [f"linear_attn.{x}.weight" for x in
                        ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d", "norm")]
            suffixes += ["linear_attn.A_log", "linear_attn.dt_bias"]
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
        keys.update(f"model.layers.{i}.{suffix}" for suffix in common + suffixes)
    return keys


def export_text_target(source: Path, output: Path) -> Path:
    source, output = Path(source).absolute(), Path(output).absolute()
    if not source.is_dir():
        raise ValueError("--source must be an existing local snapshot directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {output}")
    if source.resolve() in output.resolve().parents:
        raise ValueError("Output must be outside the source snapshot")
    outer = _json(source / "config.json")
    if outer.get("model_type") != "zdtaichu5_0" or outer.get("architectures") != ["ZDTaichu5_0_ForConditionalGeneration"]:
        raise ValueError("Expected official zdtaichu5_0 target architecture")
    config = copy.deepcopy(outer.get("llm_config"))
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5_text" or config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("Expected Qwen3_5ForCausalLM / qwen3_5_text llm_config")
    if config.get("auto_map"):
        raise ValueError("Unexpected remote-code mapping in text config")
    expected = _expected_keys(config)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if config.get(key) is None and outer.get(key) is not None:
            config[key] = outer[key]
    if config.get("eos_token_id") is None:
        raise ValueError("No text or outer EOS token ID")
    index = _json(source / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing weight_map")
    grouped = {}
    text_keys = set()
    for key, shard in mapping.items():
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"Unsafe shard path: {shard!r}")
        if not key.startswith((PREFIX, "vision_model.", "mlp1.")):
            raise ValueError(f"Unexpected tensor namespace: {key}")
        if key.startswith(PREFIX):
            text_keys.add(key[len(PREFIX):])
        grouped.setdefault(shard, set()).add(key)
    if text_keys != expected:
        raise ValueError(f"Text key mismatch: missing={sorted(expected-text_keys)}, extra={sorted(text_keys-expected)}")
    # Validate all shards, including vision-only files, before creating output.
    for shard, indexed_keys in grouped.items():
        path = source / shard
        if not path.is_file():
            raise ValueError(f"Missing shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != indexed_keys:
                raise ValueError(f"Shard/index key mismatch: {shard}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        _json(source / name)
    generation = None
    if (source / "generation_config.json").is_file():
        generation = _json(source / "generation_config.json")
        if generation.get("eos_token_id") is None:
            generation["eos_token_id"] = config["eos_token_id"]
    metadata_files = []
    for path in source.iterdir():
        name = path.name
        if (name.startswith(("tokenizer", "special_tokens_map", "added_tokens", "chat_template"))
            or name in ("vocab.json", "merges.txt", "THIRD_PARTY_LICENSES.md")
            or name.lower().startswith(("license", "licence", "notice", "copying", "copyright", "readme"))):
            if path.is_file() or path.is_dir():
                metadata_files.append(path)
    # Atomic mkdir guarantees even an empty pre-existing output is never reused.
    output.mkdir(parents=False, exist_ok=False)
    try:
        output_map, total_size = {}, 0
        text_shards = [name for name in sorted(grouped) if any(k.startswith(PREFIX) for k in grouped[name])]
        for number, shard in enumerate(text_shards, 1):
            filename = f"model-{number:05d}-of-{len(text_shards):05d}.safetensors"
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                tensors = {key[len(PREFIX):]: handle.get_tensor(key)
                           for key in sorted(grouped[shard]) if key.startswith(PREFIX)}
                total_size += sum(t.numel() * t.element_size() for t in tensors.values())
                save_file(tensors, output / filename, metadata=handle.metadata() or {"format": "pt"})
                output_map.update({key: filename for key in tensors})
                del tensors
        _write_json(output / "config.json", config)
        _write_json(output / "model.safetensors.index.json",
                    {"metadata": {"total_size": total_size}, "weight_map": output_map})
        for path in metadata_files:
            if path.is_dir():
                shutil.copytree(path, output / path.name)
            else:
                shutil.copy2(path, output / path.name)
        if generation is not None:
            _write_json(output / "generation_config.json", generation)
        (output / "TEXT_EXPORT_NOTICE.md").write_text(
            "Text-only derivative of TaichuAI/ZDTaichu5.0-9B.\n"
            "config.json was extracted from llm_config with outer special-token fallbacks; "
            "weight keys had language_model. removed, preserving tensor values.\n"
            "Vision/projector tensors were omitted. Original notices are retained.\n"
            "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n"
            "CUDA serving has not been verified by this exporter.\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export_text_target(args.source, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Text-only target exported to {result}; CUDA/serving unverified.")


if __name__ == "__main__":
    main()
