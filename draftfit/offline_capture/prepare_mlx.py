"""Capture local hybrid MLX features from revision-bound, pretokenized JSONL.

Run without torchrun. The trainer remains PyTorch; this is an offline producer.
The first JSONL row must identify target_revision and tokenizer_sha256. Each
following row has a unique id, input_ids and an explicit assistant loss_mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.target_revision):
        raise ValueError("target revision must be an immutable 40-character commit")
    if min(args.max_length, args.num_samples) < 1:
        raise ValueError("sample and length limits must be positive")
    path = args.target.expanduser().resolve()
    if not path.is_dir():
        raise ValueError("MLX capture requires a local model directory")
    fingerprints = input_fingerprints(args)
    if path.parent.name == "snapshots" and path.name != args.target_revision:
        raise ValueError("target revision differs from local snapshot")
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") not in {"qwen3_5", "qwen3_5_text", "prism_hadamard_qwen35"}:
        raise ValueError("unsupported MLX target; GGUF must use its exact capture runtime")
    if config.get("model_file") or config.get("auto_map"):
        raise ValueError("custom model code is not accepted by MLX capture")
    if config["model_type"] == "prism_hadamard_qwen35":
        from .prism_config import validate_prism_config
        validate_prism_config(config)
    target = config.get("text_config", config)
    if not isinstance(target, dict) or any(
        type(target.get(key)) is not int or target[key] <= 0
        for key in ("hidden_size", "vocab_size", "num_hidden_layers")
    ):
        raise ValueError("target hidden_size, vocab_size and depth must be positive integers")
    draft = json.loads(args.draft_config.read_text())
    if not isinstance(draft, dict) or not isinstance(draft.get("dflash_config", {}), dict):
        raise ValueError("draft config must be an object with a valid dflash_config")
    layers = draft.get("dflash_config", {}).get("target_layer_ids", draft.get("target_layer_ids"))
    if (not isinstance(layers, list) or not layers
            or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in layers)
            or layers != sorted(set(layers))):
        raise ValueError("draft must declare unique ascending target taps within target depth")
    for key in ("hidden_size", "vocab_size"):
        if type(draft.get(key)) is not int or draft[key] != target[key]:
            raise ValueError(f"draft/target {key} differs")
    tokenizer_hash = fingerprints["tokenizer_sha256"]
    with args.data.open() as stream:
        first = next(stream, None)
        if first is None:
            raise ValueError("empty capture dataset: provenance header is required")
        header = json.loads(first)
        if (not isinstance(header, dict) or header.get("kind") != "provenance" or header.get("target_revision") != args.target_revision
                or header.get("tokenizer_sha256") != tokenizer_hash):
            raise ValueError("tokenized dataset target revision/tokenizer fingerprint differs")
        rows, seen = [], set()
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("dataset records must be objects")
            sample_id, ids, mask = row.get("id"), row.get("input_ids"), row.get("loss_mask")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError("dataset IDs must be nonempty and unique")
            if (not isinstance(ids, list) or not 1 <= len(ids) <= args.max_length
                    or any(type(i) is not int or not 0 <= i < target["vocab_size"] for i in ids)):
                raise ValueError("dataset token IDs/length violate target contract; no silent truncation")
            if (not isinstance(mask, list) or len(mask) != len(ids)
                    or any(type(i) is not int or i not in (0, 1) for i in mask) or not any(mask[1:])):
                raise ValueError("dataset requires explicit binary loss_mask with supervised successor tokens")
            rows.append(row)
            seen.add(sample_id)
            if len(rows) == args.num_samples:
                break
    if not rows:
        raise ValueError("empty capture dataset")
    from draftfit.inference.capture_manifest import build_capture_manifest
    manifest = build_capture_manifest(
        strategy="dspark", capture_method="dspark", capture_layers=layers,
        feature_names=("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"),
        target_model=str(path), target_revision=args.target_revision,
        target_hidden_size=target["hidden_size"], target_vocab_size=target["vocab_size"],
        draft_vocab_size=draft["vocab_size"], target_repr="hidden_state",
        target_feature="target_last_hidden_states", tokenizer_version=tokenizer_hash,
        chat_template="pretokenized-explicit-loss-mask", max_length=args.max_length,
        is_preformatted=True, draft_config=draft, capture_backend="mlx",
        capture_settings={**fingerprints, "precision_bridge": "float32",
                          "target_num_hidden_layers": target["num_hidden_layers"],
                          "model_type": config["model_type"], "offline_only": True},
    )
    verify_inputs(args, manifest)
    return manifest, rows


def input_fingerprints(args):
    """Bind actual local contents, including every safetensors shard.

    A revision on an ordinary local directory is caller-supplied provenance;
    these hashes distinguish its contents without asserting Hub authenticity.
    """
    path = args.target.expanduser().resolve()
    weights = sorted(path.glob("*.safetensors"))
    if not weights or any(not p.is_file() for p in weights):
        raise ValueError("MLX capture requires local safetensors weights")
    return {"config_sha256": sha256(path / "config.json"),
            "tokenizer_sha256": sha256(path / "tokenizer.json"),
            "draft_config_sha256": sha256(args.draft_config),
            "data_sha256": sha256(args.data),
            "weights_sha256": {p.name: sha256(p) for p in weights}}


def verify_inputs(args, manifest):
    settings = manifest["producer"]["settings"]
    for name, value in input_fingerprints(args).items():
        if value != settings[name]:
            raise ValueError(f"MLX capture input fingerprint changed: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.target = args.target.expanduser()
    args.draft_config = args.draft_config.expanduser()
    args.data = args.data.expanduser()
    args.output = args.output.expanduser()
    manifest, rows = resolve(args)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("use a new output directory for each target/split/revision")
    if args.dry_run:
        print(json.dumps({"manifest": manifest, "samples": len(rows), "weights_loaded": False,
                          "feature_bytes_estimate": feature_bytes(manifest, rows)}, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estimated = feature_bytes(manifest, rows)
    if shutil.disk_usage(args.output.parent).free < estimated + max(256*1024**2, estimated//20):
        raise ValueError("insufficient disk for this bounded FP32 feature capture; reduce --num-samples")
    import importlib.metadata
    import torch
    from .mlx import OfflineMLXCapture
    from draftfit.inference.capture_manifest import write_capture_manifest
    verify_inputs(args, manifest)
    teacher = OfflineMLXCapture.from_pretrained(args.target, revision=args.target_revision)
    contract = manifest["contract"]
    if (teacher.hidden_size != contract["target_hidden_size"]
            or teacher.vocab_size != contract["target_vocab_size"]
            or teacher.num_hidden_layers != manifest["producer"]["settings"]["target_num_hidden_layers"]):
        raise ValueError("loaded MLX target geometry differs from capture manifest")
    teacher.set_capture_layers(manifest["contract"]["capture_layers"])
    # Stage the entire bounded batch before exposing any feature directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlx-capture-", dir=args.output.parent) as temporary:
        folder = Path(temporary)
        for i, row in enumerate(rows):
            ids = torch.tensor([row["input_ids"]])
            batch = teacher.capture(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    loss_mask=torch.tensor([row["loss_mask"]]))
            torch.save({"input_ids": batch.input_ids[0], "loss_mask": batch.loss_mask[0],
                        "hidden_states": batch.hidden_states[0],
                        "target_last_hidden_states": batch.last_hidden_states[0]}, folder / f"{i:06d}.ckpt")
        verify_inputs(args, manifest)
        write_capture_manifest(str(folder / "capture_manifest.json"), manifest)
        versions = {}
        for package in ("mlx", "mlx-lm", "mlx-vlm", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        (folder / "capture_complete.json").write_text(json.dumps({
            "samples": len(rows), "ids": [r["id"] for r in rows],
            "model_sha256": manifest["producer"]["settings"]["weights_sha256"],
            "versions": versions,
        }, indent=2))
        args.output.mkdir()  # exclusive reservation, also guards concurrent captures
        try:
            folder.replace(args.output)
        except OSError:
            # Only remove our empty reservation; never remove another writer's data.
            try:
                args.output.rmdir()
            except OSError:
                pass
            raise
    print(json.dumps({"output": str(args.output), "samples": len(rows), "status": "complete"}))


def feature_bytes(manifest, rows):
    contract = manifest["contract"]
    width = contract["target_hidden_size"] * (len(contract["capture_layers"]) + 1)
    return sum(len(row["input_ids"])*(width*4 + 16) + 4096 for row in rows)


if __name__ == "__main__":
    main()
