#!/usr/bin/env python3
"""Create a deterministic 1K train/holdout JSONL split without datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def make_split(source: str, output_dir: str, *, train_size: int = 1000, holdout_size: int = 200, seed: int = 20260830) -> dict[str, int | str]:
    if train_size < 1 or holdout_size < 1:
        raise ValueError("train_size and holdout_size must be positive")
    rows = [line for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < train_size + holdout_size:
        raise ValueError(f"need at least {train_size + holdout_size} records, got {len(rows)}")
    for line_number, line in enumerate(rows, 1):
        json.loads(line)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train = [rows[index] for index in indices[:train_size]]
    holdout = [rows[index] for index in indices[train_size : train_size + holdout_size]]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_path, holdout_path = root / "train.jsonl", root / "holdout.jsonl"
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    holdout_path.write_text("\n".join(holdout) + "\n", encoding="utf-8")
    manifest = {"source": str(Path(source).resolve()), "seed": seed, "source_rows": len(rows), "train_rows": len(train), "holdout_rows": len(holdout), "train": str(train_path), "holdout": str(holdout_path)}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output_dir")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--holdout-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    print(json.dumps(make_split(args.source, args.output_dir, train_size=args.train_size, holdout_size=args.holdout_size, seed=args.seed), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
