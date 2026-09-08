"""Weight-free Hugging Face target inspection for DSpark planning.

Only metadata/configuration files are read.  Remote modeling code and weight
files are never imported or downloaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_ALLOWED_REMOTE_FILES = {
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "README.md",
    "model.safetensors.index.json",
}
_LAYER_KEYS = {"layer_types", "layers_block_type", "block_types", "attn_types"}


def _read_json_bytes(data: bytes, name: str) -> Dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _remote_json(repo: str, filename: str, revision: str) -> Optional[Dict[str, Any]]:
    if filename not in _ALLOWED_REMOTE_FILES:
        raise ValueError(f"refusing to fetch non-metadata file {filename!r}")
    url = "https://huggingface.co/{}/resolve/{}/{}?download=false".format(
        quote(repo, safe="/"), quote(revision, safe=""), filename
    )
    try:
        with urlopen(Request(url, headers={"User-Agent": "dspark-inspector/0.1"}), timeout=20) as response:
            return _read_json_bytes(response.read(), filename)
    except (HTTPError, URLError, TimeoutError):
        return None


def _remote_text(repo: str, filename: str, revision: str) -> Optional[str]:
    if filename not in _ALLOWED_REMOTE_FILES:
        raise ValueError(f"refusing to fetch non-metadata file {filename!r}")
    url = "https://huggingface.co/{}/resolve/{}/{}?download=false".format(
        quote(repo, safe="/"), quote(revision, safe=""), filename
    )
    try:
        with urlopen(Request(url, headers={"User-Agent": "dspark-inspector/0.1"}), timeout=20) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError):
        return None


def _remote_api(repo: str, revision: str) -> Optional[Dict[str, Any]]:
    url = "https://huggingface.co/api/models/{}/revision/{}".format(
        quote(repo, safe="/"), quote(revision, safe="")
    )
    try:
        with urlopen(Request(url, headers={"User-Agent": "dspark-inspector/0.1"}), timeout=20) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def _local_metadata(root: Path, filename: str) -> Optional[Any]:
    path = root / filename
    if not path.is_file():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _walk_values(value: Any, key: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield child_key, child
            yield from _walk_values(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child, key)


def _first(config: Dict[str, Any], names: Iterable[str]) -> Any:
    wanted = set(names)
    for key, value in _walk_values(config):
        if key in wanted and isinstance(value, (str, int, float, bool)):
            return value
    return None


def _layer_types(config: Dict[str, Any]) -> List[str]:
    for key, value in _walk_values(config):
        if key in _LAYER_KEYS and isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return list(value)
    return []


def _text_blob(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False).lower()


def _classify(config: Dict[str, Any], layer_types: List[str]) -> Tuple[str, str]:
    blob = _text_blob(config)
    model_type = str(_first(config, {"model_type"}) or "").lower()
    moe = any(token in blob for token in ("num_local_experts", "num_experts", "moe"))
    linear = any(token in blob for token in ("linear_attention", "gated_delta", "deltanet"))
    recurrent = linear or any(token in blob for token in ("rwkv", "mamba", "recurrent"))
    conv = "conv" in blob or "shortconv" in blob
    mixed = bool(layer_types) and len(set(layer_types)) > 1
    if moe and (recurrent or conv or mixed):
        return "moe_hybrid", "mixed_expert_and_stateful_layers"
    if moe:
        return "moe", "expert_routing"
    if (recurrent or conv) and (conv or mixed):
        return "hybrid_stateful", "recurrent_or_conv_plus_attention"
    if recurrent or model_type in {"rwkv7", "mamba", "mamba2", "gdn"}:
        return "linear_recurrent", "constant_or_linear_recurrent_state"
    if mixed:
        return "attention_hybrid", "mixed_attention_layer_types"
    return "dense", "standard_attention_kv_state"


def _parameter_count(config: Dict[str, Any], api: Optional[Dict[str, Any]]) -> Optional[int]:
    if api:
        safetensors = api.get("safetensors")
        if isinstance(safetensors, dict) and isinstance(safetensors.get("total"), int):
            return safetensors["total"]
    value = _first(config, {"num_parameters", "n_params", "parameter_count"})
    return int(value) if isinstance(value, (int, float)) else None


def _tap_candidates(layer_types: List[str], n_layers: Optional[int]) -> List[int]:
    if layer_types:
        preferred = [
            i
            for i, kind in enumerate(layer_types)
            if kind.lower() == "full_attention" or kind.lower().startswith("full_")
        ]
        if preferred:
            return preferred[:8]
    if not n_layers or n_layers < 1:
        return []
    points = {0, n_layers - 1, n_layers // 2, n_layers // 4, (3 * n_layers) // 4}
    return sorted(x for x in points if 0 <= x < n_layers)


def inspect_target(source: str, revision: str = "main", local_only: bool = False) -> Dict[str, Any]:
    root = Path(source).expanduser()
    is_local = root.is_dir()
    if local_only and not is_local:
        raise ValueError(f"local-only inspection requires a directory: {source}")
    if is_local:
        loaded = {name: _local_metadata(root, name) for name in _ALLOWED_REMOTE_FILES}
        api = None
        repo = str(root.resolve())
    else:
        api = _remote_api(source, revision)
        metadata_revision = api.get("sha") if isinstance(api, dict) else None
        metadata_revision = metadata_revision or revision
        loaded = {
            name: _remote_json(source, name, metadata_revision)
            for name in _ALLOWED_REMOTE_FILES
            if name.endswith(".json")
        }
        loaded["chat_template.jinja"] = _remote_text(source, "chat_template.jinja", metadata_revision)
        loaded["README.md"] = _remote_text(source, "README.md", metadata_revision)
        repo = source

    raw_config = loaded.get("config.json")
    if not isinstance(raw_config, dict):
        raise ValueError(f"{source!r} has no readable config.json")
    config = raw_config.get("text_config") if isinstance(raw_config.get("text_config"), dict) else raw_config
    tokenizer = loaded.get("tokenizer_config.json") or {}
    if not isinstance(tokenizer, dict):
        tokenizer = {}
    chat_template = tokenizer.get("chat_template")
    chat_source = "tokenizer_config.json" if chat_template else None
    if not chat_template and loaded.get("chat_template.jinja"):
        chat_template = loaded["chat_template.jinja"]
        chat_source = "chat_template.jinja"
    layer_types = _layer_types(config)
    lane, state_reason = _classify(config, layer_types)
    n_layers = _first(config, {"num_hidden_layers", "n_layer", "num_layers"})
    n_layers = int(n_layers) if isinstance(n_layers, (int, float)) else None
    blob = _text_blob(config)
    if layer_types:
        # Prefer text-model layer types; multimodal configs may contain a vision
        # encoder whose convolution keys must not change the LM state contract.
        has_conv = any("conv" in kind.lower() for kind in layer_types)
        has_linear = any("linear" in kind.lower() or "delta" in kind.lower() for kind in layer_types)
    else:
        has_conv = "conv" in blob or "shortconv" in blob
        has_linear = any(token in blob for token in ("linear_attention", "gated_delta", "deltanet"))
    if lane == "linear_recurrent":
        state_kind = "recurrent"
    elif lane == "hybrid_stateful":
        state_kind = "conv_plus_kv" if has_conv else "linear_plus_kv" if has_linear else "hybrid_kv"
    else:
        state_kind = "kv"
    if lane == "moe_hybrid":
        state_kind = "moe_plus_conv" if has_conv else "moe_plus_linear" if has_linear else "moe_plus_kv"
    recommendations = {
        "draft_depth_candidates": [x for x in (3, 5, 7) if not n_layers or x <= n_layers],
        "target_tap_candidates": _tap_candidates(layer_types, n_layers),
        "train_block_candidates": [7, 16],
        "decode_block_candidates": [4, 7, 12, 16],
        "anchor_sampling_candidates": ["uniform", "random"],
        "required_gates": ["chat_template_fixture", "target_only_greedy_token_parity", "hidden_feature_parity"],
    }
    index = loaded.get("model.safetensors.index.json") or {}
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    names = [key for key in weight_map if isinstance(key, str)] if isinstance(weight_map, dict) else []
    weight_keys = {
        "embedding_candidates": sorted(key for key in names if key.endswith((
            "embed_tokens.weight", "word_embeddings.weight", "wte.weight",
        ))),
        "head_candidates": sorted(key for key in names if key.endswith("lm_head.weight")),
    }
    if state_kind != "kv":
        recommendations["required_gates"].append("state_snapshot_rollback_replay")
    if not chat_template:
        recommendations["required_gates"].append("explicit_renderer_required")
    evidence = [
        {"field": "architecture", "source": "config.json", "confidence": "high"},
        {"field": "state_kind", "source": "config.json", "confidence": "medium"},
        {
            "field": "parameter_count",
            "source": "huggingface_api.safetensors.total" if api else "config.json",
            "confidence": "high" if api else "medium",
        },
        {
            "field": "chat_template",
            "source": chat_source or "missing",
            "confidence": "high" if chat_template else "low",
        },
    ]
    recommendations["provenance"] = {
        "draft_depth_candidates": "bounded by num_hidden_layers from config.json",
        "target_tap_candidates": "full-attention layer types when declared; otherwise evenly spaced config layers",
        "train_block_candidates": "DSpark baseline candidates; validate against target state and memory",
        "anchor_sampling_candidates": "DSpark baseline ablation; measure accepted-tokens/sec",
    }
    resolved_revision = (
        api.get("sha")
        if isinstance(api, dict) and isinstance(api.get("sha"), str) and api.get("sha")
        else revision
    )
    return {
        "schema_version": "target_inspect_v1",
        "source": repo,
        "revision": resolved_revision,
        "requested_revision": revision,
        "config": raw_config,
        "weight_keys": weight_keys,
        "facts": {
            "model_type": _first(config, {"model_type"}),
            "architectures": raw_config.get("architectures", []),
            "parameter_count": _parameter_count(config, api),
            "num_hidden_layers": n_layers,
            "layer_types": layer_types,
            "architecture_lane": lane,
            "state_kind": state_kind,
            "state_reason": state_reason,
            "chat_template_present": bool(chat_template),
            "chat_template_has_generation": bool(re.search(r"\{%[-+]?\s*generation\s*[-+]?%\}", chat_template)) if isinstance(chat_template, str) else None,
            "chat_template_source": chat_source,
            "vocab_size": _first(config, {"vocab_size"}),
            "context_length": _first(config, {"max_position_embeddings", "max_sequence_length", "context_length"}),
        },
        "recommendations": recommendations,
        "evidence": evidence,
        "unknown": [
            "exact capture boundary and normalization are not inferable from config alone",
            "backend support must be verified against the pinned SGLang revision",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a HF target without downloading weights")
    parser.add_argument("source", help="HF repo id or local model directory")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = inspect_target(args.source, revision=args.revision, local_only=args.local_only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
