"""Pinned Humanizer completion preparation; never applies a chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

MODEL_ID = "jialinyyzz/humanizer"
REVISION = "7232532cb8466bd393544e97826e12e6b5e471ac"
PROMPT_FINGERPRINT = "cc51d66b4c593fbe"


def load_prompt_format(path: str | Path) -> dict:
    """Verify the official X probe (SHA256 of UTF-8 prompt, first 16 hex)."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_prompt_format(value)
    return value


def validate_prompt_format(value: Mapping) -> None:
    if not isinstance(value, Mapping) or not all(
        isinstance(value.get(key), str) for key in ("instr", "sep")
    ):
        raise ValueError("prompt format needs string instr and sep")
    probe = value["instr"] + "\n\nX" + value["sep"]
    fingerprint = hashlib.sha256(probe.encode("utf-8")).hexdigest()[:16]
    if (
        value.get("chat") is not False
        or value.get("style") != "legacy"
        or value.get("fingerprint") != PROMPT_FINGERPRINT
        or fingerprint != PROMPT_FINGERPRINT
    ):
        raise ValueError("prompt format does not match the pinned official X fingerprint")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def render_prompt(draft: str, prompt_format: Mapping) -> str:
    validate_prompt_format(prompt_format)
    return prompt_format["instr"] + "\n\n" + _text(draft, "draft").strip() + prompt_format["sep"]


def prepare_record(
    record: Mapping, tokenizer: Any, prompt_format: Mapping, *, max_length: int = 8192
) -> dict[str, list[int]]:
    """Encode once; supervise only tokens wholly inside the explicit rewrite.

    Offsets are character positions in the concatenated text. Tokens spanning
    the prompt/rewrite boundary are masked out, as are inserted special tokens.
    No separately encoded prefix length, decoding heuristic or chat renderer is
    used. Rewrites retain their supplied whitespace; only drafts are stripped.
    """
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a JSON object with draft and rewrite")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    prompt = render_prompt(record.get("draft"), prompt_format)
    rewrite = _text(record.get("rewrite"), "rewrite")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with character offsets is required")
    text = prompt + rewrite
    encoded = tokenizer(
        text, add_special_tokens=True, truncation=False,
        return_offsets_mapping=True, return_special_tokens_mask=True,
    )
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    specials = encoded["special_tokens_mask"]
    if not ids or len(ids) != len(offsets) or len(ids) != len(specials):
        raise ValueError("tokenizer returned inconsistent ids/offsets/special mask")
    if len(ids) > max_length:
        raise ValueError(f"record has {len(ids)} tokens, exceeds max_length={max_length}; no truncation")
    mask = []
    for (start, end), special in zip(offsets, specials):
        if not 0 <= start <= end <= len(text):
            raise ValueError("tokenizer returned invalid character offsets")
        mask.append(int(not special and len(prompt) <= start < end))
    if not any(mask):
        raise ValueError("rewrite has no wholly contained supervised tokens")
    return {"input_ids": ids, "loss_mask": mask}


def prepare_jsonl(
    input_path: str | Path, output_path: str | Path, tokenizer: Any,
    prompt_format: Mapping, *, max_length: int = 8192,
) -> int:
    """Stream records to a new file; errors leave no partial published output."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    temporary = None
    try:
        with input_path.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=".humanizer-", suffix=".jsonl", delete=False,
        ) as target:
            temporary = Path(target.name)
            count = 0
            for line_number, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                    prepared = prepare_record(record, tokenizer, prompt_format, max_length=max_length)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
                target.write(json.dumps(prepared) + "\n")
                count += 1
            if not count:
                raise ValueError("input JSONL is empty")
        # Same-directory hard link publishes atomically without overwriting a
        # concurrently created output (unlike replace()).
        os.link(temporary, output_path)
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL with explicit draft and rewrite strings")
    parser.add_argument("output", type=Path, help="new pretokenized JSONL (never overwritten)")
    parser.add_argument("--max-length", type=int, default=8192, help="reject longer rows; never truncate")
    parser.add_argument("--local-files-only", action="store_true", help="use the pinned HF cache only")
    args = parser.parse_args(argv)
    try:
        if args.max_length < 1:
            raise ValueError("max_length must be positive")
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        # Fetch only these assets, never model weights or remote Python code.
        paths = {
            name: hf_hub_download(
                MODEL_ID, name, revision=REVISION, local_files_only=args.local_files_only,
            )
            for name in ("prompt_format.json", "tokenizer.json", "tokenizer_config.json")
        }
        prompt_format = load_prompt_format(paths["prompt_format.json"])
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(paths["tokenizer.json"]).parent), use_fast=True,
            local_files_only=True, trust_remote_code=False,
        )
        count = prepare_jsonl(args.input, args.output, tokenizer, prompt_format, max_length=args.max_length)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"humanizer: {exc}\n")
    print(f"Prepared {count} records; revision={REVISION}; prompt_fingerprint={PROMPT_FINGERPRINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
