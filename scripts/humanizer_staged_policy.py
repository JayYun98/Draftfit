"""Immutable journal slices and held-out-loss decisions for staged Humanizer runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

FAMILIES = ("dflash", "dspark", "dflash2")


def token_content_hash(row: dict) -> str:
    ids = row["input_ids"]
    if "attention_mask" in row:
        ids = ids[:sum(row["attention_mask"])]
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".staged-", delete=False) as out:
        temporary = Path(out.name)
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def accepted_slice(journal: Path, start: int, count: int) -> tuple[list[dict], list[str]]:
    """Read the append-only generation journal; do not mutate its 30k target."""
    if start < 0 or count < 1:
        raise ValueError("invalid stage slice")
    tokens, ids = [], []
    with journal.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("status") != "accepted":
                continue
            if len(ids) >= start and len(ids) < start + count:
                if not isinstance(row.get("id"), str) or not isinstance(row.get("tokens"), dict):
                    raise ValueError("accepted journal row lacks id or tokens")
                ids.append(row["id"])
                tokens.append(row["tokens"])
            elif len(ids) < start:
                ids.append(row["id"])
            if len(ids) >= start + count:
                break
    if len(tokens) != count:
        raise ValueError(f"need {start + count} accepted rows; found {len(ids)}")
    return tokens, ids[start:]


def publish_slice(journal: Path, folder: Path, start: int, count: int,
                  excluded_ids: set[str], excluded_hashes: set[str] | None = None) -> dict:
    """Bind an immutable, disjoint stage dataset to exact journal content."""
    tokens, ids = accepted_slice(journal, start, count)
    if len(ids) != len(set(ids)) or set(ids) & excluded_ids:
        raise ValueError("training stages or held-out set overlap by id")
    if excluded_hashes is not None:
        hashes = [token_content_hash(row) for row in tokens]
        if len(hashes) != len(set(hashes)) or set(hashes) & excluded_hashes:
            raise ValueError("new journal shard overlaps prior train or held-out tokens")
    folder.mkdir(parents=True, exist_ok=True)
    target, receipt = folder / "train.tokens.jsonl", folder / "data.json"
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in tokens).encode()
    digest = hashlib.sha256(payload).hexdigest()
    desired = {"journal": str(journal.resolve()), "start": start, "count": count,
               "ids": ids, "sha256": digest, "tokens": str(target.resolve())}
    if receipt.exists() or target.exists():
        if not receipt.exists() or not target.exists() or json.loads(receipt.read_text()) != desired \
                or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("existing stage dataset differs; refusing overwrite")
        return desired
    with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".stage-", delete=False) as out:
        temporary = Path(out.name)
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(receipt, desired)
    return desired


def publish_pool_slice(pool: Path, folder: Path, start: int, count: int,
                       heldout_hashes: set[str]) -> dict:
    """Snapshot a disjoint range of an already tokenized immutable train pool."""
    if start < 0 or count < 1:
        raise ValueError("invalid pool slice")
    source_sha = hashlib.sha256(pool.read_bytes()).hexdigest()
    with pool.open(encoding="utf-8") as source:
        selected = [json.loads(line) for index, line in enumerate(source)
                    if start <= index < start + count]
    if len(selected) != count:
        raise ValueError("train pool has fewer rows than the requested stage")
    hashes = [hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()
              for row in selected]
    if set(hashes) & heldout_hashes:
        raise ValueError("train pool overlaps fixed held-out tokens")
    folder.mkdir(parents=True, exist_ok=True)
    target, receipt = folder / "train.tokens.jsonl", folder / "data.json"
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in selected).encode()
    desired = {"pool": str(pool.resolve()), "pool_sha256": source_sha, "start": start,
               "count": count, "ids": [f"pool:{start + i}" for i in range(count)],
               "sha256": hashlib.sha256(payload).hexdigest(), "tokens": str(target.resolve())}
    if receipt.exists() or target.exists():
        if not receipt.exists() or not target.exists() or json.loads(receipt.read_text()) != desired \
                or hashlib.sha256(target.read_bytes()).hexdigest() != desired["sha256"]:
            raise ValueError("existing stage pool slice differs")
        return desired
    with tempfile.NamedTemporaryFile("wb", dir=folder, prefix=".stage-", delete=False) as out:
        temporary = Path(out.name)
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(receipt, desired)
    return desired


def decide(history: list[dict], losses: dict[str, float], *, step: int,
           patience: int = 3, min_delta: float = 0.0) -> dict:
    """Only fixed held-out eval/avg_loss informs patience; all arms share a stop."""
    if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("invalid patience policy")
    if set(losses) != set(FAMILIES) or any(not math.isfinite(v) or v < 0 for v in losses.values()):
        raise ValueError("missing or nonfinite held-out loss")
    if history and step <= history[-1]["step"]:
        raise ValueError("evaluation steps must increase")
    best = {}
    meaningful_best = {}
    stale = {}
    for name in FAMILIES:
        prior = [row for row in history if name in row["losses"]]
        # Replay losses so old receipts need no migration. Small raw minima
        # accumulate against the last meaningful improvement, not each other.
        baseline, stale[name] = math.inf, 0
        values = [row["losses"][name] for row in prior] + [losses[name]]
        for value in values:
            if value < baseline - min_delta:
                baseline, stale[name] = value, 0
            else:
                stale[name] += 1
        best[name] = min(values)
        meaningful_best[name] = baseline
    improved = {name: stale[name] == 0 for name in FAMILIES}
    stop = all(stale[name] >= patience for name in FAMILIES)
    return {"step": step, "losses": losses, "best": best, "meaningful_best": meaningful_best, "stale": stale,
            "improved": improved, "stop": stop, "reason": "heldout_plateau" if stop else None}
