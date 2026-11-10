"""Batched text conversation regeneration with one resident target model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _recover_journal(path):
    """Discard only an interrupted last record; reject corruption inside a journal."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                return
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError(f"corrupt completed journal record: {path}")
                handle.truncate(start)
                handle.flush()
                os.fsync(handle.fileno())
                return
            if not line.endswith(b"\n"):
                handle.seek(0, 2)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                return


def _load_target(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from draftfit.modeling.target.quantization import target_quantization_identity

    source = Path(args.model).expanduser()
    if not source.is_dir() and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise ValueError("remote local-generation models require an exact revision")
    options = dict(
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(args.model, **options)
    quantization = target_quantization_identity(config)
    dtype = "auto" if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    placement = {"device_map": {"": args.device}} if quantization else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        **options,
        **placement,
    )
    if quantization is None:
        model.to(args.device)
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    if not tokenizer.chat_template:
        raise ValueError("local regeneration requires a tokenizer chat template")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    files = []
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in {
                ".safetensors",
                ".bin",
                ".json",
                ".model",
                ".jinja",
                ".txt",
                ".py",
            }:
                stat = path.stat()
                files.append(
                    (str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns)
                )
    identity = dict(
        model=str(source.resolve()) if source.is_dir() else args.model,
        revision=getattr(config, "_commit_hash", None) or args.revision,
        local_file_metadata=hashlib.sha256(json.dumps(files).encode()).hexdigest(),
        quantization=quantization,
        dtype=str(model.dtype),
        tokenizer_sha256=hashlib.sha256(
            json.dumps(
                {
                    "template": tokenizer.chat_template,
                    "special_tokens": tokenizer.special_tokens_map,
                    "tokenizer": (
                        tokenizer.backend_tokenizer.to_str()
                        if tokenizer.is_fast
                        else tokenizer.get_vocab()
                    ),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    )
    return model, tokenizer, identity


def _regenerate_rows(rows, args, tokenizer, batcher):
    from draftfit.data.regenerate import (
        add_regeneration_provenance,
        set_skipped,
        validate_regen_input,
    )

    states = []
    for row in rows:
        invalid = validate_regen_input(row)
        if invalid is not None:
            set_skipped(row, invalid)
            continue
        states.append(dict(row=row, source=row["conversations"], messages=[], index=0))
    while states:
        active, prompts = [], []
        for state in states:
            messages, source = state["messages"], state["source"]
            while state["index"] < len(source):
                message = source[state["index"]]
                state["index"] += 1
                if message["role"] == "assistant":
                    continue
                messages.append(dict(message))
                if message["role"] == "user":
                    template_options = (
                        {"enable_thinking": False}
                        if args.reasoning == "disable"
                        else {}
                    )
                    prompts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            **template_options,
                        )
                    )
                    active.append(state)
                    break
            else:
                state["row"]["conversations"] = messages
                state["row"]["status"] = "success"
                add_regeneration_provenance(state["row"], args)
                state["row"]["regeneration"]["backend"] = "transformers"
        if not active:
            break
        outputs = batcher.generate(prompts)
        if len(outputs) != len(active):
            raise RuntimeError("local generation returned a different row count")
        states = []
        for state, output in zip(active, outputs):
            if not isinstance(output, str) or not output.strip():
                set_skipped(
                    state["row"], "Local generation returned empty assistant content"
                )
                continue
            if args.reasoning == "disable":
                from draftfit.data.conversation_validation import has_think_marker

                if has_think_marker(output):
                    set_skipped(
                        state["row"], "Local response contains a thinking marker"
                    )
                    continue
            state["messages"].append(dict(role="assistant", content=output))
            states.append(state)
    return rows


def regenerate(args):
    """Commit complete rows only; interrupted pending rows remain reproducible inputs."""
    from draftfit.data.generation_batching import AdaptiveGenerationBatcher
    from draftfit.data.regenerate import load_processed_ids

    if args.reasoning == "save" or args.is_gpt_oss:
        raise ValueError(
            "local regeneration supports text-only none/disable reasoning modes"
        )
    for name in ("batch_size", "max_auto_batch"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not math.isfinite(args.tuning_seconds) or args.tuning_seconds <= 0:
        raise ValueError("tuning_seconds must be finite and positive")
    if not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 0.9:
        raise ValueError("memory_fraction must keep at least 10% headroom")
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        raise ValueError("deadline_unix must be finite")
    output = Path(args.output_file_path)
    if (
        output.suffix != ".jsonl"
        or output.resolve() == Path(args.input_file_path).resolve()
    ):
        raise ValueError("use a separate .jsonl output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = dict(
        success=output,
        error=output.with_name(output.stem + "_error.jsonl"),
        skipped=output.with_name(output.stem + "_skipped.jsonl"),
    )
    input_path = Path(args.input_file_path)
    destinations = [
        *paths.values(),
        output.with_suffix(".regeneration.json"),
        output.with_suffix(".progress.json"),
        output.with_suffix(".autotune.json"),
    ]
    if any(
        path.resolve() == input_path.resolve()
        or (
            path.exists() and input_path.exists() and os.path.samefile(path, input_path)
        )
        for path in destinations
    ):
        raise ValueError("use separate input and output journal paths")
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("use separate output journal paths")
    digest = hashlib.sha256()
    with Path(args.input_file_path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    contract = {
        name: getattr(args, name)
        for name in (
            "model",
            "revision",
            "torch_dtype",
            "reasoning",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "repetition_penalty",
            "regenerated_by",
            "trust_remote_code",
        )
    }
    contract["input_sha256"] = digest.hexdigest()
    manifest = output.with_suffix(".regeneration.json")
    existing = any(path.exists() and path.stat().st_size for path in paths.values())
    if args.resume and existing:
        if (
            not manifest.is_file()
            or json.loads(manifest.read_text()).get("contract") != contract
        ):
            raise ValueError(
                "resume input, target or generation contract differs; use a new output"
            )
    elif not args.resume or not manifest.exists():
        _write_json(manifest, dict(contract=contract))
    if args.resume:
        for path in paths.values():
            _recover_journal(path)
    processed = load_processed_ids(paths.values()) if args.resume else set()
    progress = output.with_suffix(".progress.json")
    counts = dict(success=0, skipped=0, error=0)
    pending, seen, submitted = [], set(), 0
    model = tokenizer = batcher = None
    mode = "a" if args.resume else "w"
    handles = {}
    try:
        for status, path in paths.items():
            handles[status] = path.open(mode, encoding="utf-8")

        def commit():
            nonlocal model, tokenizer, batcher
            if args.deadline_unix is not None and time.time() >= args.deadline_unix:
                raise TimeoutError("local generation deadline")
            if batcher is None:
                model, tokenizer, identity = _load_target(args)
                saved = json.loads(manifest.read_text())
                if (
                    args.resume
                    and existing
                    and saved.get("target_identity") != identity
                ):
                    raise ValueError("resume target files changed; use a new output")
                _write_json(manifest, dict(contract=contract, target_identity=identity))
                generation = dict(
                    max_new_tokens=args.max_tokens,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if args.temperature > 0:
                    generation.update(
                        temperature=args.temperature,
                        top_k=args.top_k if args.top_k is not None else 0,
                    )
                    if args.top_p is not None:
                        generation["top_p"] = args.top_p
                if args.repetition_penalty is not None:
                    generation["repetition_penalty"] = args.repetition_penalty
                batcher = AdaptiveGenerationBatcher(
                    model,
                    tokenizer,
                    batch_size=args.batch_size,
                    auto_batch=args.auto_batch,
                    max_batch_size=args.max_auto_batch,
                    tuning_seconds=args.tuning_seconds,
                    memory_fraction=args.memory_fraction,
                    deadline_unix=args.deadline_unix,
                    receipt_path=output.with_suffix(".autotune.json"),
                    identity=identity,
                    generation_kwargs=generation,
                )
            for row in _regenerate_rows(pending, args, tokenizer, batcher):
                status = row["status"]
                handles[status].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[status] += 1
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
            _write_json(
                progress,
                dict(
                    status="running",
                    committed_this_run=counts,
                    submitted_this_run=submitted,
                    backend="transformers",
                ),
            )
            pending.clear()

        with Path(args.input_file_path).open(encoding="utf-8") as source:
            for line in source:
                if args.num_samples is not None and submitted >= args.num_samples:
                    break
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get("id")) not in (str, int):
                    raise ValueError(
                        "local regeneration requires a stable string/integer id per row"
                    )
                key = row["id"]
                if key in seen:
                    raise ValueError("input contains duplicate row ids")
                seen.add(key)
                if key in processed:
                    continue
                pending.append(row)
                submitted += 1
                if len(pending) >= (
                    args.max_auto_batch if args.auto_batch else args.batch_size
                ):
                    commit()
            if pending:
                commit()
        _write_json(
            progress,
            dict(
                status="complete",
                committed_this_run=counts,
                submitted_this_run=submitted,
                backend="transformers",
            ),
        )
        return 0
    except BaseException as exc:
        _write_json(
            progress,
            dict(
                status="deadline" if isinstance(exc, TimeoutError) else "failed",
                error_type=type(exc).__name__,
                committed_this_run=counts,
                submitted_this_run=submitted,
                backend="transformers",
            ),
        )
        raise
    finally:
        for handle in handles.values():
            handle.close()
