"""Benchmark a running SGLang server.

Request scheduling is adapted from z-lab/dflash's MIT benchmark:
https://github.com/z-lab/dflash/blob/main/dflash/benchmark.py

The runner is speculative-algorithm agnostic and consumes SGLang's optional
speculative-decoding metadata when the server returns it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['question']}\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['problem']}\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{row['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: row["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda row: row["prompt"],
        "multi_turn": True,
    },
}
_JSONL_LINE_LIMIT = 8 * 1024 * 1024
_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_UNSUPPORTED_TOOL_FIELDS = frozenset({"tool_calls", "function_call"})


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    dataset: str
    samples: int
    output_tokens: int
    latency_seconds: float
    throughput_tokens_per_second: float
    average_acceptance_length: Optional[float] = None
    spec_verify_count: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _limit_samples(prompts: list[Any], max_samples: Optional[int]) -> list[Any]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if max_samples is not None and len(prompts) > max_samples:
        random.Random(42).shuffle(prompts)
        prompts = prompts[:max_samples]
    return prompts


def _load_prompts(name: str, max_samples: Optional[int]) -> list[list[str]]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    from datasets import load_dataset

    descriptor = DATASETS[name]
    dataset = load_dataset(
        *descriptor["load_args"],
        **descriptor["load_kwargs"],
    )
    prompts: list[list[str]] = []
    for row in dataset:
        formatted = descriptor["format"](row)
        if descriptor.get("multi_turn"):
            if not isinstance(formatted, list) or not formatted or not all(
                isinstance(turn, str) for turn in formatted
            ):
                raise ValueError(f"dataset {name!r} contains invalid multi-turn prompt")
            prompts.append(formatted)
        else:
            if not isinstance(formatted, str):
                raise ValueError(f"dataset {name!r} contains an invalid prompt")
            prompts.append([formatted])
    if not prompts:
        raise ValueError(f"dataset {name!r} did not contain any prompts")
    return _limit_samples(prompts, max_samples)


def _load_messages_jsonl(
    path: str | Path, max_samples: Optional[int]
) -> list[list[dict[str, Any]]]:
    """Load held-out conversations, omitting each row's final answer."""

    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"messages JSONL does not exist: {path}")

    prompts: list[list[dict[str, Any]]] = []
    with path.open("rb") as input_file:
        line_number = 0
        while True:
            raw_line = input_file.readline(_JSONL_LINE_LIMIT + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line) > _JSONL_LINE_LIMIT:
                raise ValueError(
                    f"messages JSONL line {line_number} exceeds the {_JSONL_LINE_LIMIT} byte limit"
                )
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid JSON in messages JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"messages JSONL line {line_number} must be an object"
                )
            if "system" in row:
                raise ValueError(
                    f"messages JSONL line {line_number} has unsupported top-level "
                    "system; include it as a system message"
                )
            tools = row.get("tools")
            if tools is not None and tools != []:
                raise ValueError(
                    f"messages JSONL line {line_number} has unsupported non-empty "
                    "top-level tools; text-only benchmark does not accept tool schemas"
                )
            has_messages = "messages" in row
            has_conversations = "conversations" in row
            if has_messages and has_conversations:
                raise ValueError(
                    f"messages JSONL line {line_number} must contain only one of "
                    "'messages' or 'conversations'"
                )
            messages = (
                row.get("messages") if has_messages else row.get("conversations")
            )
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"messages JSONL line {line_number} must contain a nonempty "
                    "messages array"
                )
            prompt = []
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"messages JSONL line {line_number} message {message_index} "
                        "must be an object"
                    )
                role = message.get("role")
                content = message.get("content")
                if role not in _MESSAGE_ROLES:
                    raise ValueError(
                        f"messages JSONL line {line_number} message {message_index} "
                        "must use an OpenAI role"
                    )
                unsupported_tool_fields = sorted(
                    _UNSUPPORTED_TOOL_FIELDS.intersection(message)
                )
                if unsupported_tool_fields:
                    fields = ", ".join(unsupported_tool_fields)
                    raise ValueError(
                        f"messages JSONL line {line_number} message {message_index} "
                        f"contains unsupported {fields}; text-only benchmark does not "
                        "accept tool calls"
                    )
                if not isinstance(content, str):
                    raise ValueError(
                        f"messages JSONL line {line_number} message {message_index} "
                        "content must be a string"
                    )
                prompt.append(dict(message))
            if prompt[-1].get("role") != "assistant":
                raise ValueError(
                    f"messages JSONL line {line_number} must end with an assistant answer"
                )
            prompt = prompt[:-1]
            if not prompt:
                raise ValueError(
                    f"messages JSONL line {line_number} contains only an assistant answer"
                )
            if not any(message["role"] == "user" for message in prompt):
                raise ValueError(
                    f"messages JSONL line {line_number} must contain a user message"
                )
            prompts.append(prompt)

    if not prompts:
        raise ValueError(f"messages JSONL {path} did not contain any prompts")
    return _limit_samples(prompts, max_samples)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_messages(sample: list[Any]) -> list[dict[str, Any]]:
    if sample and isinstance(sample[0], dict):
        return sample
    # ponytail: builtin mt-bench remains first-turn; true replay needs
    # sequential server responses, while local JSONL retains assistant history.
    return [{"role": "user", "content": sample[0]}]


def _apply_chat_template(tokenizer, messages, enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def _send_sglang(args, prompt: str) -> dict[str, Any]:
    import requests

    response = requests.post(
        args.base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return payload[0] if isinstance(payload, list) else payload


def _run_sglang(args) -> BenchmarkResult:
    import requests
    from transformers import AutoTokenizer

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    messages_jsonl = getattr(args, "messages_jsonl", None)
    if messages_jsonl:
        dataset = _load_messages_jsonl(messages_jsonl, args.max_samples)
        dataset_name = "messages-jsonl"
        source_metadata = {
            "kind": "messages_jsonl",
            "path": str(Path(messages_jsonl).resolve()),
            "sha256": _sha256_file(messages_jsonl),
            "sample_count": len(dataset),
        }
    else:
        dataset = _load_prompts(args.dataset, args.max_samples)
        dataset_name = args.dataset
        source_metadata = {
            "kind": "builtin",
            "name": args.dataset,
            "sample_count": len(dataset),
        }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_count = args.num_prompts
    warmup_count = args.concurrency
    prompts = [
        _apply_chat_template(
            tokenizer,
            _prompt_messages(dataset[index % len(dataset)]),
            args.enable_thinking,
        )
        for index in range(prompt_count + warmup_count)
    ]

    try:
        requests.get(
            args.base_url.rstrip("/") + "/flush_cache",
            timeout=min(args.timeout_seconds, 60),
        ).raise_for_status()
    except requests.RequestException:
        print("Warning: /flush_cache failed. Continuing.")

    with ThreadPoolExecutor(max_workers=warmup_count) as executor:
        list(
            executor.map(
                lambda prompt: _send_sglang(args, prompt), prompts[:warmup_count]
            )
        )
    prompts = prompts[warmup_count:]

    total_tokens = 0
    verify_count = 0
    acceptance_lengths: list[float] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_send_sglang, args, prompt) for prompt in prompts]
        for future in as_completed(futures):
            output = future.result()
            metadata = output.get("meta_info", {}) or {}
            total_tokens += int(metadata.get("completion_tokens", 0))
            verify_count += int(metadata.get("spec_verify_ct", 0))
            if metadata.get("spec_accept_length") is not None:
                try:
                    acceptance_lengths.append(float(metadata["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        backend="sglang",
        dataset=dataset_name,
        samples=prompt_count,
        output_tokens=total_tokens,
        latency_seconds=elapsed,
        throughput_tokens_per_second=total_tokens / max(elapsed, 1e-12),
        average_acceptance_length=(
            statistics.fmean(acceptance_lengths) if acceptance_lengths else None
        ),
        spec_verify_count=verify_count or None,
        metadata={
            "seed": 42,
            "model": args.model,
            "tokenizer": args.model,
            "source": source_metadata,
            "num_prompts_requested": prompt_count,
            "warmup_prompts": warmup_count,
            "concurrency": args.concurrency,
            "base_url": args.base_url,
            "timeout_seconds": args.timeout_seconds,
            "trust_remote_code": args.trust_remote_code,
            "max_samples": args.max_samples,
            "generation": {
                "max_new_tokens": getattr(args, "max_new_tokens", 2048),
                "temperature": getattr(args, "temperature", 0.0),
                "top_p": getattr(args, "top_p", 1.0),
                "top_k": getattr(args, "top_k", 1),
                "enable_thinking": getattr(args, "enable_thinking", False),
            },
        },
    )


def _print_result(result: BenchmarkResult) -> None:
    print(f"Backend: {result.backend}")
    print(f"Dataset: {result.dataset} ({result.samples} completed prompts)")
    print(f"Output throughput: {result.throughput_tokens_per_second:.2f} tok/s")
    if result.average_acceptance_length is not None:
        print(f"Average acceptance length: {result.average_acceptance_length:.3f}")
    if result.spec_verify_count is not None:
        print(f"Speculative verify count: {result.spec_verify_count}")


def run(args) -> int:
    random.seed(42)
    path = Path(args.output_json).expanduser() if args.output_json else None
    # Reserve before tokenizer/server work; exclusive creation protects baselines.
    output_file = path.open("x", encoding="utf-8") if path else None
    try:
        result = _run_sglang(args)
        if output_file is not None:
            json.dump(asdict(result), output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        if output_file is not None:
            try:
                ours, current = os.fstat(output_file.fileno()), path.stat()
                if (ours.st_dev, ours.st_ino) == (current.st_dev, current.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if output_file is not None:
            output_file.close()
    _print_result(result)
    return 0


__all__ = ["BenchmarkResult", "DATASETS", "run"]
