# SPDX-License-Identifier: MIT
# Adapted from LeanModels/Osprey f62cd488; inherited Transformers components retain their licenses.
"""Osprey source-backbone conversion into a registered, function-preserving draft."""
import json
import shutil
from pathlib import Path
from typing import Dict
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_config(source: Path) -> Dict:
    with open(source / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_state_dict(source: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = source / "model.safetensors"
    if safetensors_path.exists():
        out = {}
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
        return out

    safetensors_index = source / "model.safetensors.index.json"
    if safetensors_index.exists():
        with open(safetensors_index, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        shards = sorted(set(index["weight_map"].values()))
        out = {}
        for shard in shards:
            with safe_open(str(source / shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    out[key] = f.get_tensor(key)
        return out

    bin_path = source / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    bin_index = source / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        shards = sorted(set(index["weight_map"].values()))
        out = {}
        for shard in shards:
            out.update(torch.load(source / shard, map_location="cpu", weights_only=True))
        return out

    raise FileNotFoundError(
        f"No model.safetensors[.index.json] or pytorch_model.bin[.index.json] under {source}"
    )

def expand_qkv(old: torch.Tensor, hidden_size: int) -> torch.Tensor:
    new = old.new_zeros((old.shape[0], hidden_size * 2))
    new[:, hidden_size:] = old
    return new

def make_fc_weight(config: Dict) -> torch.Tensor:
    """Initialize ``fc``, which maps the three concatenated target aux hidden
    states (``3 * target_hidden_size``) to the draft hidden size.

    Identity on the last aux block, the paper's setting: at initialization
    ``fc(aux)`` is the deepest tapped target hidden state (truncated to the draft
    width). Function preservation does not depend on this -- it comes from the
    zero target-feature half of Q/K/V, which multiplies ``fc(aux)`` by zero. What
    the identity block buys is a gradient path: with a non-zero ``fc(aux)`` the
    zero Q/K/V half receives gradient from step 0 and the drafter starts reading
    the target. An all-zero ``fc`` together with that zero half is a stationary
    point: ``RMSNorm(0) = 0``, so both ``dL/dW_qkv[:, :h]`` and ``dL/dfc`` vanish
    identically and the target-feature path never trains.
    """
    hidden_size = config["hidden_size"]
    target_hidden_size = config.get("target_hidden_size", hidden_size)
    dtype = getattr(torch, config.get("torch_dtype", "bfloat16"), torch.bfloat16)
    fc = torch.zeros((hidden_size, target_hidden_size * 3), dtype=dtype)
    block = 2  # the deepest of the three tapped layers
    shared = min(hidden_size, target_hidden_size)
    start = block * target_hidden_size
    fc[:shared, start : start + shared] = torch.eye(shared, dtype=dtype)
    return fc

def convert_state(
    source_state: Dict[str, torch.Tensor],
    config: Dict,
    draft_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    target_prefix = "layers" if num_layers > 1 else "midlayer"
    out: Dict[str, torch.Tensor] = {}

    embed = source_state["model.embed_tokens.weight"][:draft_vocab_size].contiguous()
    out["embed_tokens.weight"] = embed

    if "lm_head.weight" in source_state:
        source_head = source_state["lm_head.weight"]
    else:
        source_head = embed
    out["lm_head.weight"] = source_head[:draft_vocab_size].clone().contiguous()

    out["norm.weight"] = source_state["model.norm.weight"].contiguous()
    out["fc.weight"] = make_fc_weight(config)

    for layer_idx in range(num_layers):
        src = f"model.layers.{layer_idx}"
        dst = f"{target_prefix}.{layer_idx}" if num_layers > 1 else target_prefix

        out[f"{dst}.self_attn.q_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.q_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.k_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.k_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.v_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.v_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.o_proj.weight"] = source_state[
            f"{src}.self_attn.o_proj.weight"
        ].contiguous()

        # Qwen3-specific: per-head RMSNorm on Q and K (shape: (head_dim,)).
        # Copied verbatim from source. The eagle3 attention applies these
        # AFTER projection and reshape to per-head; the (head_dim,) weight
        # is invariant to the 2x QKV input expansion done above.
        for name in ("q_norm", "k_norm"):
            key = f"{src}.self_attn.{name}.weight"
            if key in source_state:
                out[f"{dst}.self_attn.{name}.weight"] = source_state[key].contiguous()

        input_norm = source_state[f"{src}.input_layernorm.weight"].contiguous()
        out[f"{dst}.hidden_norm.weight"] = input_norm.clone()
        out[f"{dst}.input_layernorm.weight"] = input_norm.clone()
        out[f"{dst}.post_attention_layernorm.weight"] = source_state[
            f"{src}.post_attention_layernorm.weight"
        ].contiguous()

        for name in ("gate_proj", "up_proj", "down_proj"):
            out[f"{dst}.mlp.{name}.weight"] = source_state[
                f"{src}.mlp.{name}.weight"
            ].contiguous()

    t2d = torch.zeros(config["vocab_size"], dtype=torch.bool)
    d2t = torch.zeros(draft_vocab_size, dtype=torch.int64)
    t2d[:draft_vocab_size] = True
    out["t2d"] = t2d
    out["d2t"] = d2t

    return out

def make_eagle_config(
    config: Dict,
    draft_vocab_size: int,
    num_draft_layers: int,
    aux_layer_ids,
    target_hidden_size: int,
) -> Dict:
    eagle = dict(config)
    eagle["num_hidden_layers"] = num_draft_layers
    eagle["architectures"] = ["OspreyLlamaDraftModel" if config["model_type"] == "llama" else "OspreyQwen3DraftModel"]
    eagle["draft_vocab_size"] = draft_vocab_size
    eagle["tie_word_embeddings"] = False
    eagle["use_cache"] = True
    eagle.setdefault("pretraining_tp", 1)
    if target_hidden_size is not None:
        eagle["target_hidden_size"] = target_hidden_size
    if aux_layer_ids is not None:
        eagle["eagle_config"] = {
            "eagle_aux_hidden_state_layer_ids": aux_layer_ids,
            "use_aux_hidden_state": True,
        }
    eagle["swap_h_and_emb"] = True
    return eagle

def copy_tokenizer_files(source: Path, output: Path):
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)

def convert(source: str, output: str, num_layers: int, target_hidden_size: int, aux_layer_ids: list[int]):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = load_config(source)
    if config.get("model_type") not in {"qwen3", "llama"}:
        raise ValueError("Osprey conversion requires a Qwen3 or Llama source backbone")
    layer_types = config.get("layer_types")
    if layer_types is None and config["model_type"] == "qwen3":
        from transformers import Qwen3Config
        # Resolve legacy window settings using the source model's configuration rules.
        layer_types = Qwen3Config.from_dict(config).layer_types
    if layer_types is not None and any(kind != "full_attention" for kind in layer_types):
        raise ValueError("Osprey source conversion supports full attention only; sliding attention is unsupported")
    if config.get("attention_bias") or config.get("mlp_bias"):
        raise ValueError("Osprey source conversion currently requires bias-free attention/MLP")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    supported = {"default", "llama3"} if config["model_type"] == "llama" else {"default"}
    if rope.get("rope_type", rope.get("type", "default")) not in supported:
        raise ValueError("Osprey source conversion supports default RoPE or Llama3 scaling for Llama")
    if not 1 <= num_layers <= config["num_hidden_layers"]:
        raise ValueError("requested layers must fit the source backbone")
    if len(aux_layer_ids) != 3 or len(set(aux_layer_ids)) != 3 or any(i < 0 for i in aux_layer_ids):
        raise ValueError("Osprey requires three distinct nonnegative target layer taps")
    config = make_eagle_config(config, config["vocab_size"], num_layers, aux_layer_ids, target_hidden_size)
    source_state = load_state_dict(source)
    if not source_state["model.embed_tokens.weight"].dtype.is_floating_point:
        raise ValueError("Osprey conversion requires unquantized floating-point source weights")
    config["torch_dtype"] = str(source_state["model.embed_tokens.weight"].dtype).split(".")[-1]
    state = convert_state(source_state, config, config["vocab_size"])
    from draftfit.training.model_loading import _draft_config_from_dict
    from draftfit.modeling.auto import AutoDraftModel
    # Check actual model keys before producing a checkpoint.
    with torch.device("meta"):
        model = AutoDraftModel.from_config(_draft_config_from_dict(config))
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("converted Osprey tensors do not match the registered draft")
    output.mkdir(parents=True)
    save_file(state, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    copy_tokenizer_files(source, output)
    return {"architecture": config["architectures"][0], "layers": num_layers, "tensors": len(state)}
