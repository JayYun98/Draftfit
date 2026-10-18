"""Bounded, resumable Humanizer rewrite generation from one split of input drafts.

Example (run in the provisioned single-GPU environment)::

  python scripts/generate_humanizer_campaign_pairs.py --input data/train.inputs.jsonl \
    --output-dir out --required-count 30000 --batch-size 16 --max-batches 3 \
    --hourly-usd 1.50 --deadline-unix 1791900000

Exit 0 means the exact requested count was published. Exit 2 means an explicit
stage or stop-file request left a resumable journal prefix at a batch boundary.
Exit 3 means a pilot, source exhaustion, deadline, or budget stop left partial
progress. A stopped stage never publishes the final tokens/pairs or ready manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Also support `python scripts/generate_humanizer_campaign_pairs.py` from a
# checkout before an editable install has been performed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draftfit.data.humanizer import (
    MODEL_ID,
    PROMPT_FINGERPRINT,
    REVISION,
    load_prompt_format,
    prepare_record,
    render_prompt,
)

MAX_LENGTH = 512
MAX_DRAFT_TOKENS = 150
MAX_NEW_TOKENS = 192
SEED = 42


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".humanizer-", delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"existing final output differs: {path}")
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=".humanizer-", delete=False
    ) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_inputs(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows, ids = [], set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ),
            )
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"]
            ):
                raise ValueError("id must be a nonempty string")
            if row["id"] in ids:
                raise ValueError(f'duplicate id {row["id"]}')
            if not isinstance(row.get("draft"), str) or not row["draft"].strip():
                raise ValueError("draft must be nonempty text")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"input line {line_number}: {exc}") from exc
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("input is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def read_sources(paths: list[Path]) -> tuple[list[dict], str]:
    rows, ids, hashes = [], set(), []
    for path in paths:
        part, digest = read_inputs(path)
        for row in part:
            if row["id"] in ids:
                raise ValueError(f'duplicate id across inputs: {row["id"]}')
            ids.add(row["id"])
        rows.extend(part)
        hashes.append(digest)
    return rows, hashlib.sha256(json.dumps(hashes).encode("utf-8")).hexdigest()


def load_journal(path: Path, rows: list[dict]) -> list[dict]:
    attempts = []
    if not path.exists():
        return attempts
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                if entry["id"] != rows[line_number - 1]["id"] or entry[
                    "status"
                ] not in {"accepted", "rejected"}:
                    raise ValueError("journal differs from input order")
                if entry["status"] == "accepted" and not all(
                    k in entry for k in ("pair", "tokens")
                ):
                    raise ValueError("accepted entry lacks pair/tokens")
            except (IndexError, KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"journal line {line_number}: {exc}") from exc
            attempts.append(entry)
    return attempts


def assess_completion(
    row: dict,
    generated_ids: list[int],
    tokenizer,
    prompt_format: dict,
    *,
    eos_token_id: int,
    vocab_size: int,
) -> tuple[dict | None, str | None]:
    if not generated_ids or any(
        type(i) is not int or i < 0 or i >= vocab_size for i in generated_ids
    ):
        return None, "invalid_tokens"
    if eos_token_id not in generated_ids:
        return None, "no_eos"
    generated_ids = generated_ids[: generated_ids.index(eos_token_id)]
    if not generated_ids or any(
        i in set(tokenizer.all_special_ids) for i in generated_ids
    ):
        return None, "invalid_tokens"
    rewrite = tokenizer.decode(generated_ids, skip_special_tokens=False)
    if not rewrite.strip():
        return None, "empty"
    pair = {**row, "rewrite": rewrite}
    try:
        tokens = prepare_record(pair, tokenizer, prompt_format, max_length=MAX_LENGTH)
    except (ValueError, TypeError) as exc:
        return None, (
            "over_512" if "exceeds max_length" in str(exc) else "invalid_tokens"
        )
    return {"pair": pair, "tokens": tokens}, None


def estimate(
    attempts: list[dict],
    elapsed_seconds: float,
    generation_seconds: float,
    required_count: int,
    hourly_usd: float,
    *,
    warmup_generation_seconds: float = 0.0,
    warmup_accepted: int = 0,
) -> dict:
    accepted = sum(a["status"] == "accepted" for a in attempts)
    measured_seconds = generation_seconds - warmup_generation_seconds
    measured_accepted = accepted - warmup_accepted
    rate = measured_accepted / measured_seconds if measured_seconds > 0 else 0.0
    remaining = max(0, required_count - accepted)
    projected_seconds = remaining / rate if rate else None
    if projected_seconds is not None and not math.isfinite(projected_seconds):
        projected_seconds = None
    projected_cost = (
        (elapsed_seconds + projected_seconds) * hourly_usd / 3600
        if projected_seconds is not None
        else None
    )
    if projected_cost is not None and not math.isfinite(projected_cost):
        projected_cost = None
    return {
        "processed": len(attempts),
        "accepted": accepted,
        "accepted_per_second": rate,
        "elapsed_seconds": elapsed_seconds,
        "generation_seconds": generation_seconds,
        "warmup_generation_seconds": warmup_generation_seconds,
        "warmup_accepted": warmup_accepted,
        "projected_remaining_seconds": projected_seconds,
        "projected_total_usd": projected_cost,
    }


def record_pause(progress: Path, reason: str) -> None:
    if progress.exists():
        receipt = json.loads(progress.read_text(encoding="utf-8"))
        receipt["pause_reason"] = reason
        receipt["resumable"] = True
        atomic_json(progress, receipt)


def run(args, *, tokenizer=None, prompt_format=None, model=None) -> int:
    if (
        args.required_count < 1
        or args.batch_size < 1
        or (args.max_batches is not None and args.max_batches < 1)
    ):
        raise ValueError("counts and batch size must be positive")
    stop_after = getattr(args, "stop_after_accepted", None)
    if stop_after is not None and (stop_after < 1 or stop_after >= args.required_count):
        raise ValueError(
            "stop-after-accepted must be positive and below required-count"
        )
    stop_request_file = getattr(args, "stop_request_file", None)
    if not all(
        math.isfinite(x) and x > 0 for x in (args.hourly_usd, args.max_projected_usd)
    ):
        raise ValueError("hourly-usd and max-projected-usd must be positive")
    input_paths = [args.input, *getattr(args, "reserve_input", [])]
    rows, input_sha = read_sources(input_paths)
    split = args.input.name.split(".")[0]
    if split not in {"train", "validation", "val", "test"}:
        raise ValueError("input name must begin train, validation, val, or test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / f"{split}.partial.jsonl"
    progress = args.output_dir / f"{split}.progress.json"
    tokens_path = args.output_dir / f"{split}.tokens.jsonl"
    pairs_path = args.output_dir / f"{split}.pairs.jsonl"
    manifest_path = args.output_dir / f"{split}.manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{split}: ready manifest already exists")
    if progress.exists():
        previous = json.loads(progress.read_text(encoding="utf-8"))
        if (
            previous["input_sha256"] != input_sha
            or previous["required_count"] != args.required_count
        ):
            raise ValueError("resume input or required count differs")
        generation_seconds = float(previous["generation_seconds"])
        prior_elapsed_seconds = float(previous["elapsed_seconds"])
        warmup_generation_seconds = float(
            previous.get("warmup_generation_seconds", 0.0)
        )
        warmup_accepted = int(previous.get("warmup_accepted", 0))
        warmup_done = bool(previous.get("warmup_done", False))
    elif journal.exists():
        raise ValueError(
            "journal exists without progress metadata; refusing unsafe resume"
        )
    else:
        generation_seconds = 0.0
        prior_elapsed_seconds = 0.0
        warmup_generation_seconds = 0.0
        warmup_accepted = 0
        warmup_done = False
    attempts = load_journal(journal, rows)
    run_started = time.monotonic()
    if progress.exists() and previous["processed"] > len(attempts):
        raise ValueError("progress is ahead of journal")
    if sum(a["status"] == "accepted" for a in attempts) > args.required_count:
        raise ValueError("journal has too many accepted rows")
    accepted_count = sum(a["status"] == "accepted" for a in attempts)
    if stop_after is not None and accepted_count >= stop_after:
        record_pause(progress, "stop_after_accepted")
        print(
            f"stop_after_accepted: resumable journal; accepted={accepted_count}/{args.required_count}",
            flush=True,
        )
        return 2
    if stop_request_file is not None and stop_request_file.is_file():
        record_pause(progress, "stop_requested")
        print(
            f"stop_requested: resumable journal; accepted={accepted_count}/{args.required_count}",
            flush=True,
        )
        return 2
    if tokenizer is None or prompt_format is None or model is None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one CUDA GPU is required")
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            print("deadline reached before model load; pilot incomplete", flush=True)
            return 3
        assets = {
            name: hf_hub_download(
                MODEL_ID,
                name,
                revision=REVISION,
                local_files_only=args.local_files_only,
            )
            for name in (
                "prompt_format.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        }
        prompt_format = load_prompt_format(assets["prompt_format.json"])
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(assets["tokenizer.json"]).parent),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        ).eval()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError("fast tokenizer and EOS token are required")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.model_type != "gemma4_unified":
        raise ValueError("expected Gemma4Unified target")
    batches = 0
    stop_reason = None
    while (
        len(attempts) < len(rows)
        and sum(a["status"] == "accepted" for a in attempts) < args.required_count
    ):
        accepted_count = sum(a["status"] == "accepted" for a in attempts)
        if stop_after is not None and accepted_count >= stop_after:
            stop_reason = "stop_after_accepted"
            break
        if stop_request_file is not None and stop_request_file.is_file():
            stop_reason = "stop_requested"
            break
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            stop_reason = "deadline"
            break
        if args.max_batches is not None and batches >= args.max_batches:
            stop_reason = "pilot_incomplete"
            break
        start_index = len(attempts)
        remaining = args.required_count - sum(
            a["status"] == "accepted" for a in attempts
        )
        batch = rows[start_index : start_index + min(args.batch_size, remaining)]
        entries, selected, prompts = [], [], []
        for row in batch:
            draft_ids = tokenizer.encode(row["draft"].strip(), add_special_tokens=False)
            if len(draft_ids) > MAX_DRAFT_TOKENS:
                entries.append(
                    {"id": row["id"], "status": "rejected", "reason": "draft_over_150"}
                )
                continue
            prompt = render_prompt(row["draft"], prompt_format)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(prompt_ids) + MAX_NEW_TOKENS > MAX_LENGTH:
                entries.append(
                    {
                        "id": row["id"],
                        "status": "rejected",
                        "reason": "prompt_budget_over_512",
                    }
                )
                continue
            selected.append(row)
            prompts.append(prompt)
            entries.append(None)
        if prompts:
            import torch

            encoded = tokenizer(
                prompts, padding=True, return_tensors="pt", add_special_tokens=True
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=0,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            batch_generation_seconds = time.monotonic() - started
            generation_seconds += batch_generation_seconds
            prompt_width = encoded["input_ids"].shape[1]
            completed = iter(zip(selected, output[:, prompt_width:].tolist()))
            for index, entry in enumerate(entries):
                if entry is not None:
                    continue
                row, generated = next(completed)
                result, reason = assess_completion(
                    row,
                    generated,
                    tokenizer,
                    prompt_format,
                    eos_token_id=tokenizer.eos_token_id,
                    vocab_size=model.config.get_text_config().vocab_size,
                )
                entries[index] = (
                    {"id": row["id"], "status": "accepted", **result}
                    if result
                    else {"id": row["id"], "status": "rejected", "reason": reason}
                )
            if not warmup_done:
                warmup_generation_seconds = batch_generation_seconds
                warmup_accepted = sum(e["status"] == "accepted" for e in entries)
                warmup_done = True
        with journal.open("a", encoding="utf-8") as f:
            for entry in entries:
                f.write(
                    json.dumps(
                        entry,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            f.flush()
            os.fsync(f.fileno())
        attempts.extend(entries)
        batches += 1
        projection_target = (
            stop_after if stop_after is not None else args.required_count
        )
        metrics = estimate(
            attempts,
            prior_elapsed_seconds + time.monotonic() - run_started,
            generation_seconds,
            projection_target,
            args.hourly_usd,
            warmup_generation_seconds=warmup_generation_seconds,
            warmup_accepted=warmup_accepted,
        )
        receipt = {
            "model": MODEL_ID,
            "revision": REVISION,
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "inputs": [str(p.resolve()) for p in input_paths],
            "input_sha256": input_sha,
            "required_count": args.required_count,
            "projection_target_accepted": projection_target,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "processed": len(attempts),
            "rejections": dict(
                Counter(a.get("reason") for a in attempts if a["status"] == "rejected")
            ),
            "hourly_usd": args.hourly_usd,
            "warmup_done": warmup_done,
            "sampling_resume": "seed_restarts_each_process; exact RNG replay is not guaranteed",
            **metrics,
        }
        atomic_json(progress, receipt)
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)
        if stop_after is not None and metrics["accepted"] >= stop_after:
            stop_reason = "stop_after_accepted"
            break
        if stop_request_file is not None and stop_request_file.is_file():
            stop_reason = "stop_requested"
            break
        projected = metrics["projected_total_usd"]
        if batches >= 3 and (projected is None or projected > args.max_projected_usd):
            stop_reason = "projected_budget_exceeded"
            break
        if metrics["accepted"] >= args.required_count:
            break
    accepted = [a for a in attempts if a["status"] == "accepted"][: args.required_count]
    if stop_reason in {"stop_after_accepted", "stop_requested"}:
        record_pause(progress, stop_reason)
        print(
            f"{stop_reason}: resumable journal; accepted={len(accepted)}/{args.required_count}",
            flush=True,
        )
        return 2
    if len(accepted) != args.required_count:
        print(
            f'{stop_reason or "source_exhausted"}: pilot incomplete; accepted={len(accepted)}/{args.required_count}',
            flush=True,
        )
        return 3
    publish_jsonl(pairs_path, [a["pair"] for a in accepted])
    publish_jsonl(tokens_path, [a["tokens"] for a in accepted])
    atomic_json(
        manifest_path,
        {
            "status": "ready",
            "count": len(accepted),
            "input_sha256": input_sha,
            "model": MODEL_ID,
            "revision": REVISION,
            "sampling_resume": "seed_restarts_each_process; exact RNG replay is not guaranteed",
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "pairs": pairs_path.name,
            "tokens": tokens_path.name,
            "ids": [a["id"] for a in accepted],
            "provenance": [
                {k: v for k, v in a["pair"].items() if k not in {"draft", "rewrite"}}
                for a in accepted
            ],
        },
    )
    print(f"ready: {tokens_path}; count={len(accepted)}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--reserve-input",
        type=Path,
        action="append",
        default=[],
        help="additional unique candidates consumed after the primary split",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--stop-after-accepted",
        type=int,
        help="cumulative accepted-row threshold; stop after persisting the crossing batch (exit 2)",
    )
    parser.add_argument(
        "--stop-request-file",
        type=Path,
        help="if this file appears, stop after the current batch is persisted (exit 2)",
    )
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--max-projected-usd", type=float, default=3.0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, FileExistsError) as exc:
        parser.exit(2, f"humanizer generation: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
