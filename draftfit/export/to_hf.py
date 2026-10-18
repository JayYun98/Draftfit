# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Export a DataFlow training checkpoint to an HF-format draft directory.

The output loads back through ``AutoDraftModel.from_pretrained`` (e.g. to
finetune from it). EAGLE-family embeddings may be absent by design — training
reloads frozen embeddings from the target — while DFlash-family drafts do not
own an embedding at all.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from draftfit.export.checkpoint_io import (
    apply_legacy_rope_scaling,
    materialize_draft,
    resolve_training_state,
)


def _load_embedding_tensor(source: str, key: str) -> torch.Tensor:
    """Read one target embedding without materializing the target lm_head."""
    root = source if os.path.exists(source) else snapshot_download(repo_id=source)
    index_paths = glob.glob(os.path.join(root, "*.index.json"))
    if len(index_paths) > 1:
        raise FileNotFoundError(f"multiple index files found under {root}")
    if index_paths:
        with open(index_paths[0], encoding="utf-8") as handle:
            weight_file = json.load(handle).get("weight_map", {}).get(key)
        if not weight_file:
            raise KeyError(f"embedding key {key!r} is absent from {index_paths[0]}")
        path = os.path.join(root, weight_file)
    else:
        candidates = glob.glob(os.path.join(root, "*.safetensors"))
        candidates += glob.glob(os.path.join(root, "*.bin"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one model weight file under {root}, found {len(candidates)}"
            )
        path = candidates[0]

    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt") as handle:
            if key not in handle.keys():
                raise KeyError(f"embedding key {key!r} is absent from {path}")
            return handle.get_tensor(key)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if key not in state:
        raise KeyError(f"embedding key {key!r} is absent from {path}")
    return state[key]


def _add_dspark_sglang_config(output_dir: str) -> bool:
    """Mirror DSpark runtime fields into the config namespace SGLang reads.

    Draftfit keeps family-specific training settings under ``dflash_config``.
    SGLang's DSpark loader reads Markov settings from ``dspark_config`` instead,
    so an otherwise valid HF export would silently resolve ``markov_rank`` to 0.
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    method = config.get("dflash_config") or {}
    if method.get("projector_type") != "dspark":
        return False

    dspark_config = config.get("dspark_config") or {}
    if not isinstance(dspark_config, dict):
        raise ValueError("exported dspark_config must be an object")
    dspark_config = dict(dspark_config)
    dspark_config["markov_rank"] = int(method.get("markov_rank", 0) or 0)
    dspark_config["markov_head_type"] = str(
        method.get("markov_head_type") or "vanilla"
    ).lower()
    # SGLang 0.5.19's DSpark model constructor parses the nested namespace
    # above for validation, then builds the Markov head from these top-level
    # attributes on the Transformers config. Keep both representations in
    # sync so a valid rank is not reduced to the default zero at load time.
    config["markov_rank"] = dspark_config["markov_rank"]
    config["markov_head_type"] = dspark_config["markov_head_type"]
    if method.get("mask_token_id") is not None:
        dspark_config["mask_token_id"] = int(method["mask_token_id"])
    config["dspark_config"] = dspark_config

    temporary = config_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, config_path)
    return True



def export_to_hf(
    checkpoint_path: str,
    draft_config_path: str,
    output_dir: str,
    *,
    vocab_mapping_path: Optional[str] = None,
    embedding_source: Optional[str] = None,
    embedding_key: str = "model.embed_tokens.weight",
) -> str:
    """Write the checkpoint's draft as an HF directory; returns ``output_dir``.

    The exported directory is SELF-CONTAINED (reloads via ``from_pretrained``
    with no missing keys). Draft checkpoints deliberately exclude the frozen
    embedding, so it must come from somewhere real: pass ``embedding_source``
    (the target model path/dir — the embedding is deterministic from it) unless
    the checkpoint itself carries ``embed_tokens.weight``. A randomly
    initialized embedding would silently break serving, so its absence raises.
    """
    state = resolve_training_state(checkpoint_path)
    model = materialize_draft(
        state, draft_config_path, vocab_mapping_path=vocab_mapping_path
    )
    full_state = dict(model.state_dict())
    owns_embedding = hasattr(model, "embed_tokens")
    if owns_embedding and "embed_tokens.weight" not in state["draft_state_dict"]:
        if not embedding_source:
            raise ValueError(
                "checkpoint has no embed_tokens.weight (draft checkpoints "
                "exclude the frozen embedding); pass embedding_source=<target "
                "model path> so the export ships the real embedding"
            )
        load_embedding = getattr(model, "load_embedding", None)
        if load_embedding is not None:
            load_embedding(embedding_source, embedding_key=embedding_key)
    if (
        owns_embedding
        and "embed_tokens.weight" not in full_state
        and "embed_tokens.weight" not in state["draft_state_dict"]
    ):
        if not embedding_source:
            raise ValueError(
                "checkpoint does not contain embed_tokens.weight; pass "
                "embedding_source=<target model path>"
            )
        full_state["embed_tokens.weight"] = _load_embedding_tensor(
            embedding_source, embedding_key
        )
    full_state.update(state["draft_state_dict"])  # trained keys win
    model.save_pretrained(output_dir, state_dict=full_state)
    apply_legacy_rope_scaling(output_dir)
    _add_dspark_sglang_config(output_dir)
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=export_to_hf.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-mapping", default=None)
    parser.add_argument(
        "--embedding-source",
        default=None,
        help="target model path/dir supplying the frozen embedding "
        "(required unless the checkpoint carries embed_tokens.weight)",
    )
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    args = parser.parse_args(argv)
    out = export_to_hf(
        args.checkpoint,
        args.draft_config,
        args.output_dir,
        vocab_mapping_path=args.vocab_mapping,
        embedding_source=args.embedding_source,
        embedding_key=args.embedding_key,
    )
    print(f"exported HF draft to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
