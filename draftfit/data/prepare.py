"""Prepare local user data for the canonical conversation training flow.

The public CLI uses this module so installed wheels can onboard local JSON and
JSONL files without importing the source-only ``scripts`` package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

LOCAL_DATA_FORMATS = ("auto", "openai", "sharegpt", "canonical")
SUPPORTED_INPUT_SUFFIXES = {".json", ".jsonl"}
DEFAULT_EVAL_RATIO = 0.05
MAX_JSONL_LINE_CHARS = 2 * 1024 * 1024
MAX_JSON_FILE_BYTES = 64 * 1024 * 1024

_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    # The current parser renders tool responses under this role.
    "function": "tool",
}
_SUPPORTED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_MESSAGE_FIELDS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
)
_ROW_FIELDS = frozenset({"id", "messages", "conversations", "system", "tools"})


@dataclass(frozen=True)
class PrepareSummary:
    input_rows: int
    train_rows: int
    eval_rows: int
    train_path: str
    eval_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _input_rows(path: Path) -> Iterator[tuple[int, Any]]:
    path = path.expanduser()
    if not path.is_file():
        raise ValueError(f"input data file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError("input data must be a .json or .jsonl file")

    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            line_number = 0
            while line := handle.readline(MAX_JSONL_LINE_CHARS + 1):
                line_number += 1
                if not line.strip():
                    continue
                if len(line) > MAX_JSONL_LINE_CHARS:
                    raise ValueError(
                        f"input row {line_number}: exceeds 2 MiB character limit"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"input row {line_number}: invalid JSON") from exc
                yield line_number, row
        return

    if path.stat().st_size > MAX_JSON_FILE_BYTES:
        raise ValueError("input JSON exceeds 64 MiB; use JSONL for larger datasets")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"input JSON is invalid: {path}") from exc
    if isinstance(payload, list):
        for index, row in enumerate(payload, start=1):
            yield index, row
    elif isinstance(payload, dict):
        yield 1, payload
    else:
        raise ValueError("input JSON must contain an object or an array of objects")


def _format_for_row(row: Mapping[str, Any], requested: str, row_number: int) -> str:
    if requested not in LOCAL_DATA_FORMATS:
        raise ValueError(f"unsupported data format: {requested}")
    if requested != "auto":
        return requested
    has_messages = "messages" in row
    has_conversations = "conversations" in row
    if has_messages and has_conversations:
        raise ValueError(
            f"input row {row_number}: use either messages or conversations, not both"
        )
    if has_messages:
        return "openai"
    if has_conversations:
        conversations = row["conversations"]
        first = (
            conversations[0]
            if isinstance(conversations, list) and conversations
            else None
        )
        if isinstance(first, Mapping) and ("from" in first or "value" in first):
            return "sharegpt"
        return "canonical"
    raise ValueError(
        f"input row {row_number}: expected OpenAI messages or ShareGPT conversations"
    )


def _normalize_messages(
    raw_messages: Any,
    *,
    source_format: str,
    row_number: int,
) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(
            f"input row {row_number}: messages/conversations must be non-empty"
        )

    normalized: list[dict[str, Any]] = []
    previous_role: str | None = None
    saw_user = False
    saw_assistant = False
    for message_number, message in enumerate(raw_messages, start=1):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"input row {row_number}, message {message_number}: expected an object"
            )
        source_fields = {"from", "value"} if source_format == "sharegpt" else set()
        unsupported_fields = sorted(
            set(message).difference(_MESSAGE_FIELDS | source_fields)
        )
        if unsupported_fields:
            fields = ", ".join(repr(field) for field in unsupported_fields)
            raise ValueError(
                f"input row {row_number}, message {message_number}: "
                f"unsupported message field(s): {fields}"
            )
        if source_format == "sharegpt":
            raw_role = message.get("from", message.get("role"))
            content = message.get("value", message.get("content"))
        else:
            raw_role = message.get("role")
            content = message.get("content")
        if not isinstance(raw_role, str):
            raise ValueError(
                f"input row {row_number}, message {message_number}: missing message role"
            )
        role = _ROLE_ALIASES.get(raw_role, raw_role)
        if role not in _SUPPORTED_ROLES:
            raise ValueError(
                f"input row {row_number}, message {message_number}: unsupported message role"
            )

        tool_calls = message.get("tool_calls")
        if content is None and role == "assistant" and tool_calls:
            content = ""
        if not isinstance(content, str) or (not content.strip() and not tool_calls):
            raise ValueError(
                f"input row {row_number}, message {message_number}: content must be a non-empty string"
            )

        if role in {"system", "developer"}:
            if saw_user:
                raise ValueError(
                    f"input row {row_number}, message {message_number}: system/developer must precede user turns"
                )
        elif role == "user":
            if previous_role not in {None, "system", "developer", "assistant", "tool"}:
                raise ValueError(
                    f"input row {row_number}, message {message_number}: invalid role order"
                )
            saw_user = True
        elif role == "assistant":
            if previous_role not in {"user", "tool"}:
                raise ValueError(
                    f"input row {row_number}, message {message_number}: invalid role order"
                )
            saw_assistant = True
        elif role == "tool":
            if previous_role not in {"assistant", "tool"}:
                raise ValueError(
                    f"input row {row_number}, message {message_number}: tool must follow assistant/tool"
                )

        normalized_message = {
            key: value
            for key, value in message.items()
            if key in _MESSAGE_FIELDS and key not in {"from", "value"}
        }
        normalized_message["role"] = role
        normalized_message["content"] = content
        normalized.append(normalized_message)
        previous_role = role

    if not saw_user or not saw_assistant:
        raise ValueError(
            f"input row {row_number}: requires at least one user and assistant message"
        )
    if normalized[-1]["role"] != "assistant":
        raise ValueError(
            f"input row {row_number}: training conversations must end with assistant"
        )
    return normalized


def normalize_row(
    row: Any,
    *,
    data_format: str = "auto",
    row_number: int = 1,
) -> dict[str, Any]:
    """Convert one OpenAI, ShareGPT, or canonical row to training JSON."""
    if not isinstance(row, Mapping):
        raise ValueError(f"input row {row_number}: expected a JSON object")
    unsupported_fields = sorted(set(row).difference(_ROW_FIELDS))
    if unsupported_fields:
        fields = ", ".join(repr(field) for field in unsupported_fields)
        raise ValueError(f"input row {row_number}: unsupported row field(s): {fields}")
    source_format = _format_for_row(row, data_format, row_number)
    if source_format == "openai":
        raw_messages = row.get("messages")
    else:
        raw_messages = row.get("conversations")
    messages = _normalize_messages(
        raw_messages,
        source_format=source_format,
        row_number=row_number,
    )
    system = row.get("system")
    if system not in (None, ""):
        if not isinstance(system, str):
            raise ValueError(f"input row {row_number}: system must be a string")
        if messages[0]["role"] == "system":
            if messages[0]["content"] != system:
                raise ValueError(
                    f"input row {row_number}: top-level system conflicts with the first system message"
                )
        else:
            messages.insert(0, {"role": "system", "content": system})
    row_id = row.get("id")
    if row_id is None or not str(row_id).strip():
        row_id = hashlib.sha256(
            json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()[:16]
    result = {"id": str(row_id), "conversations": messages}
    if "tools" in row:
        if not isinstance(row["tools"], list):
            raise ValueError(f"input row {row_number}: tools must be a list")
        if not all(isinstance(tool, Mapping) for tool in row["tools"]):
            raise ValueError(f"input row {row_number}: tools must contain objects")
        result["tools"] = row["tools"]
    return result


def load_local_rows(
    input_path: str | Path,
    *,
    data_format: str = "auto",
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Read and validate local rows before any output file is created."""
    if max_rows is not None and max_rows <= 0:
        raise ValueError("--max-rows must be greater than zero")
    rows: list[dict[str, Any]] = []
    source_rows = _input_rows(Path(input_path))
    selected_rows = source_rows if max_rows is None else islice(source_rows, max_rows)
    for row_number, row in selected_rows:
        rows.append(normalize_row(row, data_format=data_format, row_number=row_number))
    if not rows:
        raise ValueError("input data contains no rows")
    return rows


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if not 0 < eval_ratio < 1:
        raise ValueError("eval ratio must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("an evaluation split requires at least two rows")
    fingerprints = [
        hashlib.sha256(
            json.dumps(
                row["conversations"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for row in rows
    ]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(
            "evaluation split requires unique conversations; duplicate rows found"
        )
    eval_count = min(max(1, math.ceil(len(rows) * eval_ratio)), len(rows) - 1)
    # Keep all answers for the same rendered prompt in one split. A full-row
    # fingerprint alone would let the same prompt with different answers leak
    # between train and holdout.
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        context = row["conversations"][:-1]
        context_fingerprint = hashlib.sha256(
            json.dumps(
                context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        groups.setdefault(context_fingerprint, []).append(index)
    grouped_indices = list(groups.values())
    # Keep whole prompt groups together. This is an approximate 5% split when
    # group sizes do not fit exactly; seeded tie order plus ascending sizes
    # avoids a large prompt group consuming nearly the whole evaluation set.
    random.Random(seed).shuffle(grouped_indices)
    grouped_indices.sort(key=len)
    selected_groups: list[int] = []
    selected_count = 0
    for group_index, group in enumerate(grouped_indices):
        if selected_count + len(group) <= eval_count:
            selected_groups.append(group_index)
            selected_count += len(group)
    if not selected_groups:
        candidates = [
            (len(group), group_index)
            for group_index, group in enumerate(grouped_indices)
            if len(rows) - len(group) >= 1
        ]
        if not candidates:
            raise ValueError("evaluation split requires at least two unique prompts")
        _, group_index = min(candidates)
        selected_groups = [group_index]
    eval_indices = {
        index
        for group_index in selected_groups
        for index in grouped_indices[group_index]
    }
    train = [row for index, row in enumerate(rows) if index not in eval_indices]
    evaluation = [row for index, row in enumerate(rows) if index in eval_indices]
    return train, evaluation


def _reserve_outputs(paths: Sequence[Path]) -> list[tuple[Path, Any]]:
    reserved: list[tuple[Path, Any]] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            reserved.append((path, path.open("x", encoding="utf-8")))
    except BaseException:
        _cleanup_reserved_outputs(reserved)
        raise
    return reserved


def _cleanup_reserved_outputs(reserved: Sequence[tuple[Path, Any]]) -> None:
    for path, handle in reversed(reserved):
        try:
            ours = os.fstat(handle.fileno())
            current = path.stat()
            if (ours.st_dev, ours.st_ino) == (current.st_dev, current.st_ino):
                path.unlink()
        except FileNotFoundError:
            pass
        finally:
            try:
                handle.close()
            except OSError:
                pass


def _write_jsonl(handle: Any, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    data_format: str = "auto",
    split_eval: bool = False,
    eval_output_path: str | Path | None = None,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
    seed: int = 42,
    max_rows: int | None = None,
) -> PrepareSummary:
    """Prepare a fresh canonical JSONL file for ``target prepare --train-data``."""
    train_path = Path(output_path).expanduser()
    input_file = Path(input_path).expanduser().absolute()
    train_path = train_path.absolute()
    if input_file == train_path:
        raise ValueError("output path must differ from input data file")
    if train_path.suffix.lower() != ".jsonl":
        raise ValueError("output path must end in .jsonl")
    if train_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {train_path}")

    rows = load_local_rows(input_file, data_format=data_format, max_rows=max_rows)
    evaluation: list[dict[str, Any]] | None = None
    eval_path: Path | None = None
    if split_eval:
        if eval_output_path is None:
            stem = train_path.stem
            if stem == "train":
                eval_name = "test.jsonl"
            elif stem.endswith("_train"):
                eval_name = stem[: -len("_train")] + "_test.jsonl"
            else:
                eval_name = stem + "_holdout.jsonl"
            eval_path = train_path.with_name(eval_name)
        else:
            eval_path = Path(eval_output_path).expanduser().absolute()
        if eval_path == train_path:
            raise ValueError("evaluation output must differ from training output")
        if eval_path == input_file:
            raise ValueError("evaluation output must differ from input data file")
        if eval_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {eval_path}")
        rows, evaluation = split_rows(rows, eval_ratio=eval_ratio, seed=seed)
    elif eval_output_path is not None:
        raise ValueError("evaluation output requires --split-eval")

    output_paths = [train_path] + ([eval_path] if eval_path is not None else [])
    reserved = _reserve_outputs(output_paths)
    try:
        _write_jsonl(reserved[0][1], rows)
        if eval_path is not None and evaluation is not None:
            _write_jsonl(reserved[1][1], evaluation)
        for _, handle in reserved:
            handle.flush()
    except BaseException:
        _cleanup_reserved_outputs(reserved)
        raise
    for _, handle in reserved:
        handle.close()
    return PrepareSummary(
        input_rows=len(rows) + (len(evaluation) if evaluation is not None else 0),
        train_rows=len(rows),
        eval_rows=len(evaluation) if evaluation is not None else 0,
        train_path=str(train_path),
        eval_path=str(eval_path) if eval_path is not None else None,
    )


prepare_local_dataset = prepare_dataset
