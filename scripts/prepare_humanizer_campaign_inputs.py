"""Prepare *input drafts*, never target rewrites, from PerfectBlend-dedup.

Run from the repository root with a Python environment containing datasets,
transformers and huggingface_hub. Only small source parquet shards and pinned
tokenizer assets are fetched; model weights are not requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from draftfit.data import prepare_presets
from draftfit.data.humanizer import MODEL_ID, REVISION

SOURCE = "tepirale/open-perfectblend-deduplicate-mlabonne"
WORD = re.compile(r"\b[\w'-]+\b", re.UNICODE)
CODE = re.compile(r"```|<\/?(?:code|script|pre)>|\b(?:def |class |import |return |SELECT |function |console\.log|#include)\b|[{};]{3,}", re.I)
MATH = re.compile(r"\$\$|\\\[|\\begin\{|\\frac|\\boxed|\b(?:equation|integral|derivative|solve for x|calculate|compute)\b|\d\s*[=+*/^×÷-]\s*\d|[=+*/^]{3,}", re.I)
TOOL = re.compile(r"\b(?:tool call|function call|browser\.search|python\.exec|<\|tool|<function)\b", re.I)
MATH_CODE_SOURCE = re.compile(r"math|code|program|python|leetcode|gsm8k|algebra|arithmetic", re.I)
DOMAINS = {
    "email": re.compile(r"\b(?:email|e-mail|subject line|dear recipient)\b", re.I),
    "blog": re.compile(r"\b(?:blog|article|newsletter|post for readers)\b", re.I),
    "report": re.compile(r"\b(?:report|summary for management|executive summary)\b", re.I),
    "essay": re.compile(r"\b(?:essay|personal narrative|reflection)\b", re.I),
    "forum": re.compile(r"\b(?:forum|reddit|discussion post|community post)\b", re.I),
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(text: str) -> str:
    return " ".join(text.split())


def candidate(raw: dict, normalized: dict, tokenizer, counts: Counter) -> dict | None:
    """Use a complete human/assistant exchange and one prose paragraph."""
    messages = normalized.get("conversations") or []
    raw_messages = raw.get("conversations") or []
    if MATH_CODE_SOURCE.search(str(raw.get("source", ""))):
        counts["math_or_code_source"] += 1
        return None
    if not raw_messages or str(raw_messages[-1].get("from", "")).lower() not in {"gpt", "chatgpt", "bing", "bard"}:
        counts["tool_or_nonassistant_ending"] += 1
        return None
    if len(messages) < 2 or len(messages) % 2 or any(
        not isinstance(m.get("content"), str) or not m["content"].strip()
        or m.get("role") != ("user" if i % 2 == 0 else "assistant")
        for i, m in enumerate(messages)
    ):
        counts["invalid_conversation"] += 1
        return None
    if any(CODE.search(m["content"]) or MATH.search(m["content"]) or TOOL.search(m["content"]) for m in messages):
        counts["code_math_or_tool_conversation"] += 1
        return None
    parent = digest(json.dumps(
        [(m["role"], canonical(m["content"])) for m in messages],
        ensure_ascii=False, separators=(",", ":"),
    ))
    answer = messages[-1]["content"]
    instruction = messages[-2]["content"]
    domain = next((name for name, pattern in DOMAINS.items() if pattern.search(instruction)), "general_prose")
    paragraphs = re.split(r"\n\s*\n", answer)
    for paragraph in paragraphs:
        prose = canonical(paragraph)
        words = len(WORD.findall(prose))
        if not 40 <= words <= 100 or CODE.search(prose) or MATH.search(prose) or TOOL.search(prose):
            continue
        if len(re.findall(r"[.!?](?:\s|$)", prose)) < 2:
            continue
        tokens = len(tokenizer.encode(prose, add_special_tokens=False))
        if tokens > 150:
            counts["over_150_tokens"] += 1
            continue
        return {
            "id": digest(f"{SOURCE}:{parent}:{digest(prose)}")[:24],
            "draft": prose,
            "input_kind": "source_assistant_prose_draft_not_target_rewrite",
            "source": SOURCE,
            "source_id": str(normalized["id"]),
            "source_collection": str(raw.get("source", "")),
            "domain": domain,
            "parent_hash": parent,
            "draft_hash": digest(prose),
            "word_count": words,
            "token_count": tokens,
        }
    counts["no_short_prose_paragraph"] += 1
    return None


def collect(rows, processor, tokenizer, wanted: int, counts: Counter) -> list[dict]:
    selected, parents, drafts = [], set(), set()
    for raw in rows:
        counts["source_rows_seen"] += 1
        if counts["source_rows_seen"] % 10000 == 0:
            print(f"scanned={counts['source_rows_seen']} usable={len(selected)}", flush=True)
        try:
            normalized, skipped = processor(raw, SOURCE)
            counts["source_messages_skipped_by_normalizer"] += skipped
            item = candidate(raw, normalized, tokenizer, counts)
        except (KeyError, TypeError, ValueError, AttributeError):
            counts["malformed_source_row"] += 1
            continue
        if item is None:
            continue
        if item["parent_hash"] in parents:
            counts["duplicate_parent"] += 1
            continue
        if item["draft_hash"] in drafts:
            counts["duplicate_draft"] += 1
            continue
        parents.add(item["parent_hash"])
        drafts.add(item["draft_hash"])
        selected.append(item)
        if len(selected) >= wanted:
            break
    counts["unique_usable"] = len(selected)
    return selected


def distribute_reserve(rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep the primary splits fixed; assign replacement rows to their own split."""
    reserve = rows.pop("replacement_candidates")
    counts = [len(rows[name]) for name in ("train", "validation", "test")]
    total = sum(counts)
    exact = [len(reserve) * count / total for count in counts]
    allocated = [int(value) for value in exact]
    for i in sorted(range(3), key=lambda i: (-(exact[i] - allocated[i]), i))[:len(reserve) - sum(allocated)]:
        allocated[i] += 1
    start = 0
    for name, count in zip(("train", "validation", "test"), allocated):
        rows[f"{name}_replacements"] = reserve[start:start + count]
        start += count
    return rows


def split_rows(rows: list[dict], train: int, validation: int, test: int, reserve: int, seed: int) -> dict[str, list[dict]]:
    import random

    needed = train + validation + test
    if len(rows) < needed:
        raise ValueError(f"insufficient usable drafts: {len(rows)} < {needed}")
    # One selected paragraph per parent; deterministic shuffle only after grouping.
    rows = sorted(rows, key=lambda row: row["parent_hash"])
    random.Random(seed).shuffle(rows)
    return distribute_reserve({
        "train": rows[:train],
        "validation": rows[train:train + validation],
        "test": rows[train + validation:needed],
        "replacement_candidates": rows[needed:needed + reserve],
    })


def write_outputs(output: Path, splits: dict[str, list[dict]], counts: Counter, *, source_revision: str, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "input_drafts_only_no_target_generated_rewrites",
        "source": SOURCE,
        "source_revision": source_revision,
        "source_split": "train",
        "tokenizer_model": MODEL_ID,
        "tokenizer_revision": REVISION,
        "selection": "last assistant message; 40-100 word prose paragraph; <=150 tokenizer tokens; one per normalized original conversation hash",
        "seed": seed,
        "counts": dict(counts),
        "splits": {},
    }
    for name, records in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][name] = {"count": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if len(splits["train"]) >= 48:
        pilot = output / "pilot48.jsonl"
        with pilot.open("w", encoding="utf-8") as target:
            for record in splits["train"][:48]:
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["pilot"] = {"count": 48, "source": "first 48 train rows", "sha256": hashlib.sha256(pilot.read_bytes()).hexdigest()}
    materialize_inputs(output, manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def materialize_inputs(output: Path, manifest: dict) -> None:
    """Append only the replacement pool assigned to each fixed primary split."""
    manifest["inputs"] = {}
    for name in ("train", "validation", "test"):
        primary = output / f"{name}.jsonl"
        replacements = output / f"{name}_replacements.jsonl"
        destination = output / f"{name}.inputs.jsonl"
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("wb") as target:
            target.write(primary.read_bytes())
            target.write(replacements.read_bytes())
        manifest["inputs"][name] = {
            "count": manifest["splits"][name]["count"] + manifest["splits"][f"{name}_replacements"]["count"],
            "primary_count": manifest["splits"][name]["count"],
            "replacement_count": manifest["splits"][f"{name}_replacements"]["count"],
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".planning/humanizer-run-3usd/data"))
    parser.add_argument("--train", type=int, default=30000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--reserve", type=int, default=6200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) <= 0 or args.reserve < 0:
        parser.error("split counts must be positive; reserve may be zero")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    source_revision = HfApi().dataset_info(SOURCE).sha
    tokenizer_path = hf_hub_download(MODEL_ID, "tokenizer.json", revision=REVISION)
    hf_hub_download(MODEL_ID, "tokenizer_config.json", revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(tokenizer_path).parent), local_files_only=True, use_fast=True, trust_remote_code=False)
    rows, processor = prepare_presets.load_dataset_preset("perfectblend-dedup", streaming=True)
    counts = Counter()
    selected = collect(rows, processor, tokenizer, args.train + args.validation + args.test + args.reserve, counts)
    splits = split_rows(selected, args.train, args.validation, args.test, args.reserve, args.seed)
    manifest = write_outputs(args.output, splits, counts, source_revision=source_revision, seed=args.seed)
    print(json.dumps({"splits": {k: v["count"] for k, v in manifest["splits"].items()}, "source_rows_seen": counts["source_rows_seen"], "source_revision": source_revision}))


if __name__ == "__main__":
    main()
