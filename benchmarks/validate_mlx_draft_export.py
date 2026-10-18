"""Local format and optional MLX tensor-load probe for Draftfit exports.

This does not implement a draft decoder or test speculative inference.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


PROFILES = {
    "taichu-9b": (4096, 8, 248077, [1, 8, 15, 22, 29]),
    "lfm-2.6b": (2048, 9, 125017, [2, 9, 17, 21, 27]),
    "lfm-8b-a1b": (2048, 9, 125017, [2, 6, 10, 14, 18]),
}
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1,
               "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2,
               "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
               "I64": 8, "U64": 8, "F64": 8}
REQUIRED = {"fc.weight", "hidden_norm.weight", "norm.weight"}
DFLASH2_REQUIRED = {
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
    "candidate_selector.hidden_projection.weight",
}


def check_config(root: Path, target: str, strategy: str) -> dict:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    expected = PROFILES[target]
    nested = config.get("dflash_config") or {}
    for name, value in (("hidden_size", expected[0]), ("block_size", expected[1])):
        actual = config.get(name)
        if actual is None and name == "hidden_size":
            actual = (config.get("transformer_layer_config") or {}).get(name)
        if actual != value:
            raise ValueError(f"config {name}: expected {value}, got {actual!r}")
    for name, value in (("mask_token_id", expected[2]),
                        ("target_layer_ids", expected[3])):
        actual = nested.get(name, config.get(name))
        if actual is None and name == "target_layer_ids":
            actual = config.get("aux_hidden_state_layer_ids")
        if actual != value:
            raise ValueError(f"config {name}: expected {value}, got {actual!r}")
    if strategy == "dflash2" and config.get("model_type") not in {"qwen3", "qwen3_dflash2"}:
        raise ValueError("DFlash2 export has unexpected model_type")
    return config


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > path.stat().st_size - 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(handle.read(header_size))
    payload_size = path.stat().st_size - 8 - header_size
    spans = []
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        dtype = entry["dtype"]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"{path}: invalid dtype or shape for {key}")
        if not (0 <= start <= end <= payload_size) or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"{path}: invalid byte span for {key}")
        spans.append((start, end, key))
    spans.sort()
    if not spans or spans[0][0] != 0 or spans[-1][1] != payload_size or any(
        left[1] != right[0] for left, right in zip(spans, spans[1:])
    ):
        raise ValueError(f"{path}: noncontiguous or missing tensor data")
    return {key: value for key, value in header.items() if key != "__metadata__"}


def check_export(root: Path, target: str, strategy: str, mlx_load: bool = False) -> dict:
    if not root.is_dir():
        raise ValueError(f"export directory absent: {root}")
    check_config(root, target, strategy)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        names = set(weight_map.values())
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError("index contains invalid shard name")
        shards = sorted(root / name for name in names)
    else:
        shards = sorted(root.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError("expected one safetensors file or a shard index")
        weight_map = None
    tensors = {}
    for shard in shards:
        header = read_header(shard)
        for key, entry in header.items():
            if key in tensors:
                raise ValueError(f"duplicate tensor: {key}")
            if weight_map is not None and weight_map.get(key) != shard.name:
                raise ValueError(f"index mismatch: {key}")
            tensors[key] = entry
        if mlx_load:
            try:
                import mlx.core as mx
            except ImportError as exc:
                raise RuntimeError("--mlx-load requires mlx in this Mac Python environment") from exc
            loaded = mx.load(str(shard))
            if set(loaded) != set(header):
                raise ValueError(f"MLX key mismatch in {shard}")
            for key, tensor in loaded.items():
                if list(tensor.shape) != header[key]["shape"]:
                    raise ValueError(f"MLX shape mismatch: {key}")
                if tensor.size:
                    mx.eval(tensor.reshape(-1)[:1])
            del loaded
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError("index tensor list does not match shards")
    required = REQUIRED | (DFLASH2_REQUIRED if strategy == "dflash2" else set())
    missing = sorted(required - tensors.keys())
    if missing:
        raise ValueError(f"missing required {strategy} tensors: {missing}")
    return {"target": target, "strategy": strategy, "tensor_count": len(tensors),
            "shard_count": len(shards), "format_config": "pass",
            "mlx_tensor_load": "pass" if mlx_load else "not_run",
            "token_inference": "not_run", "speed_inference": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--target", choices=PROFILES, required=True)
    parser.add_argument("--strategy", choices=("dspark", "dflash", "dflash2"), required=True)
    parser.add_argument("--mlx-load", action="store_true", help="load each safetensors shard in MLX and evaluate one element per tensor")
    args = parser.parse_args()
    try:
        result = check_export(args.export_dir, args.target, args.strategy, args.mlx_load)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"validation failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
