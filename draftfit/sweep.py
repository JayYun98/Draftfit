"""Deterministic, sequential model-option sweep manifest generation."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping


def build_sweep(spec: Mapping[str, Any], *, max_runs: int = 24) -> list[dict[str, Any]]:
    rec = spec.get("recommendations", {})
    depths = list(rec.get("draft_depth_candidates", [3, 5, 7]))
    tap_ids = [int(value) for value in rec.get("target_tap_candidates", [0])]
    tap_counts = [count for count in (3, 5, 8) if len(tap_ids) >= count]
    taps = [(count, tap_ids[:count]) for count in tap_counts] or [
        (len(tap_ids), tap_ids)
    ]
    blocks = list(rec.get("train_block_candidates", [7, 16]))
    decode_blocks = [
        int(value) for value in rec.get("decode_block_candidates", [4, 7, 12, 16])
    ]
    anchors = list(rec.get("anchor_sampling_candidates", ["uniform", "random"]))
    if max_runs < 1:
        raise ValueError("max_runs must be positive")
    rows = []
    index = 0
    for depth, (tap_count, selected_taps), block, anchor in product(
        depths, taps, blocks, anchors
    ):
        valid_decode_blocks = [value for value in decode_blocks if value <= int(block)]
        if not valid_decode_blocks:
            valid_decode_blocks = [int(block)]
        for decode_block in valid_decode_blocks:
            index += 1
            rows.append(
                {
                    "run_index": index,
                    "draft_depth": int(depth),
                    "target_tap_count": tap_count,
                    "target_taps": selected_taps,
                    "train_block_size": int(block),
                    "decode_block_size": decode_block,
                    "anchor_sampling": str(anchor),
                }
            )
            if len(rows) >= max_runs:
                return rows
    return rows


__all__ = ["build_sweep"]
