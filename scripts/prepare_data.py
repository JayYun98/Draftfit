"""Convert supported text training datasets to Draftfit conversation JSONL.

All presets produce rows with a stable id and a conversations list. Heavy
dataset dependencies stay behind loader functions so row conversion helpers and
their tests remain usable in lightweight environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = (
    "ultrachat",
    "sharegpt",
    "eaglechat",
    "perfectblend",
    "perfectblend-dedup",
    "perfectblend-llama3.1-8b-instruct",
    "perfectblend-llama3.3-70b-instruct",
    "perfectblend-llama4-scout-instruct",
    "perfectblend-llama4-maverick-instruct",
    "magpie-qwen2.5-pro-1m-v0.1",
    "opc",
    "gsm8k",
    "hendrycks_math",
    "math_qa",
    "codealpaca-20k",
    "opencodeinstruct",
    "apigen-mt",
    "nemotron-agentic",
    "baseline",
    "agent-aware",
    "mix",
    "magicoder-evol-instruct",
    "sciq",
    "camel",
    "nebius-llama31-8b-infinity-instruct",
)
UNSUPPORTED_VLM_DATASETS = frozenset({"sharegpt4v", "allava4v"})
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "cache" / "dataset"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}
OPC_SUBSETS = (
    "largescale_diverse_instruct",
    "filtered_infinity_instruct",
    "realuser_instruct",
)
MIX_SOURCES = ("perfectblend-dedup", "opencodeinstruct", "apigen-mt", "nemotron-agentic")
MIX_PRESETS = {
    "baseline": (80, 20, 0, 0),
    "agent-aware": (70, 10, 10, 10),
}

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}

ProcessedRow = tuple[dict[str, Any] | None, int]
RowProcessor = Callable[[Mapping[str, Any], str | None], ProcessedRow]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a supported Draftfit training dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset preset to prepare.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for <dataset>_train.jsonl (default: cache/dataset).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Custom ShareGPT JSON or JSONL file instead of the hosted preset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of output rows for mixtures; source row cap for single presets.",
    )
    parser.add_argument(
        "--ratios",
        help="Override mixture weights, e.g. perfectblend-dedup=70,opencodeinstruct=10,apigen-mt=10,nemotron-agentic=10 (sum 100).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Mixture sampling seed (default: 42).")
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Write a deterministic 5%% evaluation split.",
    )
    parser.add_argument(
        "--opc-subset",
        choices=(*OPC_SUBSETS, "all"),
        default=OPC_SUBSETS[0],
        help="OpenCoder opc-sft-stage1 subset, or all supported subsets.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.data_path is not None:
        if args.dataset != "sharegpt":
            parser.error("--data-path is only supported with --dataset sharegpt")
        if args.data_path.suffix.lower() not in SUPPORTED_DATA_PATH_SUFFIXES:
            parser.error("--data-path must point to a .json or .jsonl file")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero")
    if args.dataset in (*MIX_PRESETS, "mix"):
        if args.sample_size is None:
            parser.error("mixtures require --sample-size")
        if args.dataset == "mix" and args.ratios is None:
            parser.error("--dataset mix requires --ratios")
        try:
            args.ratios = parse_ratios(args.ratios) if args.ratios else MIX_PRESETS[args.dataset]
        except ValueError as exc:
            parser.error(str(exc))
        if args.data_path is not None or args.opc_subset != OPC_SUBSETS[0]:
            parser.error("--data-path and --opc-subset are unavailable for mixtures")
    elif args.ratios is not None:
        parser.error("--ratios requires --dataset baseline, agent-aware, or mix")
    if args.dataset in {"apigen-mt", "nemotron-agentic"} and args.sample_size is None:
        parser.error(f"--dataset {args.dataset} requires --sample-size")

    return args


def parse_ratios(value: str) -> tuple[Decimal, ...]:
    """Validate explicit percentages in any order; omitted sources receive zero."""
    ratios: dict[str, Decimal] = {}
    for part in value.split(","):
        name, sep, weight = part.partition("=")
        name = name.strip()
        if not sep or name not in MIX_SOURCES or name in ratios:
            raise ValueError(f"invalid or repeated mixture source: {part!r}")
        try:
            ratios[name] = Decimal(weight.strip())
        except InvalidOperation as exc:
            raise ValueError(f"invalid mixture weight: {part!r}") from exc
        if not ratios[name].is_finite() or ratios[name] < 0:
            raise ValueError("mixture weights must be finite and nonnegative")
    if sum(ratios.values()) != 100:
        raise ValueError("mixture weights must sum to 100")
    return tuple(ratios.get(name, Decimal(0)) for name in MIX_SOURCES)


def mixture_counts(size: int, ratios: Sequence[Decimal | int]) -> tuple[int, ...]:
    """Largest-remainder allocation with stable source-order tie breaks."""
    exact = [Decimal(size) * Decimal(weight) / 100 for weight in ratios]
    counts = [int(number) for number in exact]
    remainder = size - sum(counts)
    for index in sorted(range(len(counts)), key=lambda i: (-(exact[i] - counts[i]), i))[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _concatenate_hf_datasets(datasets: Sequence[Any]) -> Any:
    from datasets import concatenate_datasets

    return concatenate_datasets(list(datasets))


def _stable_id(*parts: str) -> str:
    return hashlib.md5("".join(parts).encode()).hexdigest()


def _conversation_row(
    row_id: Any,
    user_content: str,
    assistant_content: str,
) -> dict[str, Any]:
    return {
        "id": str(row_id),
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def process_ultrachat_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversations = []
    for message in row["messages"]:
        role = message["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported UltraChat role: {role!r}")
        conversations.append({"role": role, "content": message["content"]})

    return {
        "id": str(row["prompt_id"]),
        "conversations": conversations,
    }, 0


def process_sharegpt_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversations = []
    skipped_count = 0
    for message in row["conversations"]:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            skipped_count += 1
            continue
        conversations.append({"role": role, "content": message["value"]})

    return {
        "id": str(row["id"]),
        "conversations": conversations,
    }, skipped_count


def process_nebius_infinity_instruct(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversation = row["conversation"][0]
    generated_message = row["generated_message"]
    return (
        _conversation_row(
            row["id"],
            conversation["content"],
            generated_message["content"],
        ),
        0,
    )


def process_opc_sft_stage1(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    output = row["output"]
    return (
        _conversation_row(
            _stable_id(instruction, output),
            instruction,
            output,
        ),
        0,
    )


def process_codealpaca_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    prompt_input = row["input"]
    output = row["output"]
    user_content = f"{instruction}\n\n{prompt_input}" if prompt_input else instruction
    return (
        _conversation_row(
            _stable_id(user_content, output),
            user_content,
            output,
        ),
        0,
    )


def process_opencodeinstruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    row_id = row.get("id") or _stable_id(row["input"], row["output"])
    return _conversation_row(row_id, row["input"], row["output"]), 0


def process_apigen_mt_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    messages = []
    call_number = 0
    for message in row["conversations"]:
        role, content = message["from"], message["value"]
        if role == "function_call":
            call_number += 1
            call = json.loads(content)
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"call_{call_number}", "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }],
            })
        elif role == "observation":
            if not call_number:
                raise ValueError("APIGen observation has no preceding tool call")
            messages.append({"role": "tool", "content": content, "tool_call_id": f"call_{call_number}"})
        else:
            mapped_role = ROLE_MAPPING.get(role)
            if mapped_role is None:
                raise ValueError(f"unsupported APIGen role: {role!r}")
            messages.append({"role": mapped_role, "content": content})
    if row.get("system"):
        messages.insert(0, {"role": "system", "content": row["system"]})
    tools = row.get("tools") or "[]"
    return {
        "id": _stable_id(json.dumps(messages, sort_keys=True)),
        "conversations": messages,
        "tools": json.loads(tools) if isinstance(tools, str) else tools,
    }, 0


def process_nemotron_agentic_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    messages = []
    raw_messages = row["messages"]
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    for message in raw_messages:
        normalized = {"role": message["role"], "content": message.get("content") or ""}
        for field in ("tool_calls", "tool_call_id", "reasoning_content"):
            if message.get(field):
                normalized[field] = message[field]
        messages.append(normalized)
    return {
        "id": str(row["uuid"]),
        "conversations": messages,
        "tools": row.get("tools") or [],
    }, 0


def process_magicoder_evol_instruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    response = row["response"]
    return (
        _conversation_row(
            _stable_id(instruction, response),
            instruction,
            response,
        ),
        0,
    )


def process_gsm8k_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    question = row["question"]
    answer = row["answer"]
    return _conversation_row(_stable_id(question, answer), question, answer), 0


def process_hendrycks_math_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    problem = row["problem"]
    solution = row["solution"]
    return _conversation_row(_stable_id(problem, solution), problem, solution), 0


def process_math_qa_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = f"{row['Problem']}\n{row['options']}"
    rationale = row["Rationale"]
    return (
        _conversation_row(
            _stable_id(user_content, rationale),
            user_content,
            rationale,
        ),
        0,
    )


def process_sciq_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    answers = [
        row["distractor3"],
        row["distractor1"],
        row["distractor2"],
        row["correct_answer"],
    ]
    random.shuffle(answers)
    labels = ("a", "b", "c", "d")
    options = list(zip(labels, answers))
    correct_label = next(
        label for label, answer in options if answer == row["correct_answer"]
    )
    options_text = "\n".join(f"{label}) {answer}" for label, answer in options)
    user_content = f"{row['question']}\n{options_text}"
    assistant_content = (
        f"{row['support']}\nanswer: {correct_label}) {row['correct_answer']}"
    )
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def process_camel_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = row["message_1"]
    assistant_content = row["message_2"]
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def _identity_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    return dict(row), 0


def add_index(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    indexed = dict(row)
    indexed["id"] = index
    return indexed


def load_dataset_from_path(data_path: Path) -> Any:
    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_PATH_SUFFIXES:
        raise ValueError(
            f"Unsupported ShareGPT data file {data_path}; expected .json or .jsonl"
        )
    return _load_hf_dataset("json", data_files=str(data_path), split="train")


def _train_split(*args: Any, **kwargs: Any) -> Any:
    return _load_hf_dataset(*args, **kwargs)["train"]


def _indexed(dataset: Any) -> Any:
    return dataset.map(add_index, with_indices=True)


def _iter_nemotron_rows() -> Iterable[dict[str, Any]]:
    # The hosted builder currently fails on inconsistent nested tool schemas.
    url = (
        "https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/"
        "resolve/main/data/interactive_agent.jsonl"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        for line in response:
            yield json.loads(line)


def load_dataset_preset(
    dataset_name: str,
    *,
    data_path: Path | None = None,
    opc_subset: str = OPC_SUBSETS[0],
    streaming: bool = False,
) -> tuple[Any, RowProcessor | None]:
    """Load one named preset and return its row processor."""
    if dataset_name in UNSUPPORTED_VLM_DATASETS:
        raise ValueError(
            f"Dataset preset {dataset_name!r} is not supported; VLM data "
            "preparation and training are not supported"
        )
    if dataset_name == "ultrachat":
        return (
            _load_hf_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
            ),
            process_ultrachat_row,
        )
    if dataset_name == "sharegpt":
        dataset = (
            load_dataset_from_path(data_path)
            if data_path is not None
            else _load_hf_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                split="train",
            )
        )
        return dataset, process_sharegpt_row
    if dataset_name == "eaglechat":
        return _train_split("zhaode/EagleChat"), _identity_row
    if dataset_name == "perfectblend":
        return (
            _indexed(_train_split("mlabonne/open-perfectblend")),
            process_sharegpt_row,
        )
    if dataset_name == "perfectblend-dedup":
        return _indexed(_load_hf_dataset(
            "tepirale/open-perfectblend-deduplicate-mlabonne", split="train",
            streaming=streaming,
        )), process_sharegpt_row
    if dataset_name == "apigen-mt":
        return _load_hf_dataset(
            "Salesforce/APIGen-MT-5k", split="train", streaming=streaming
        ), process_apigen_mt_row
    if dataset_name == "nemotron-agentic":
        from datasets import IterableDataset

        return IterableDataset.from_generator(_iter_nemotron_rows), process_nemotron_agentic_row

    regenerated_presets = {
        "perfectblend-llama3.1-8b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct"
        ),
        "perfectblend-llama3.3-70b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.3-70B-Instruct"
        ),
        "perfectblend-llama4-scout-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Scout-17B-16E-Instruct"
        ),
        "perfectblend-llama4-maverick-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Maverick-17B-128E-Instruct"
        ),
    }
    if dataset_name in regenerated_presets:
        return _indexed(_train_split(regenerated_presets[dataset_name])), _identity_row

    if dataset_name == "magpie-qwen2.5-pro-1m-v0.1":
        dataset = _train_split("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1")
        return dataset.rename_column("uuid", "id"), process_sharegpt_row
    if dataset_name == "nebius-llama31-8b-infinity-instruct":
        dataset = _load_hf_dataset(
            "nebius/Llama-3.1-8B-Instruct-Infinity-Instruct-0625",
            split="train",
        )
        return _indexed(dataset), process_nebius_infinity_instruct
    if dataset_name == "opc":
        if opc_subset == "all":
            dataset = _concatenate_hf_datasets(
                [
                    _train_split("OpenCoder-LLM/opc-sft-stage1", subset)
                    for subset in OPC_SUBSETS
                ]
            )
        else:
            dataset = _train_split("OpenCoder-LLM/opc-sft-stage1", opc_subset)
        return dataset, process_opc_sft_stage1
    if dataset_name == "gsm8k":
        return _train_split("openai/gsm8k", "main"), process_gsm8k_row
    if dataset_name == "hendrycks_math":
        subjects = (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        )
        return (
            _concatenate_hf_datasets(
                [
                    _train_split("EleutherAI/hendrycks_math", subject)
                    for subject in subjects
                ]
            ),
            process_hendrycks_math_row,
        )
    if dataset_name == "math_qa":
        return (
            _train_split("allenai/math_qa", trust_remote_code=True),
            process_math_qa_row,
        )
    if dataset_name == "codealpaca-20k":
        return (
            _train_split("sahil2801/CodeAlpaca-20k", trust_remote_code=True),
            process_codealpaca_row,
        )
    if dataset_name == "opencodeinstruct":
        return (
            _train_split("nvidia/OpenCodeInstruct", streaming=streaming),
            process_opencodeinstruct_row,
        )
    if dataset_name == "magicoder-evol-instruct":
        return (
            _train_split(
                "ise-uiuc/Magicoder-Evol-Instruct-110K",
                trust_remote_code=True,
            ),
            process_magicoder_evol_instruct_row,
        )
    if dataset_name == "sciq":
        return (
            _train_split("allenai/sciq", trust_remote_code=True),
            process_sciq_row,
        )
    if dataset_name == "camel":
        return (
            _concatenate_hf_datasets(
                [
                    _load_hf_dataset(f"camel-ai/{subject}", split="train")
                    for subject in ("biology", "chemistry", "physics")
                ]
            ),
            process_camel_row,
        )
    raise ValueError(
        f"Unsupported dataset preset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )


def load_canonical_dataset(dataset_name: str, data_path: Path | None = None) -> Any:
    """Compatibility wrapper returning only the dataset for a named preset."""
    dataset, _ = load_dataset_preset(dataset_name, data_path=data_path)
    return dataset


def _write_split(
    dataset: Iterable[Mapping[str, Any]],
    output_path: Path,
    processor: RowProcessor | None,
    dataset_name: str,
) -> int:
    skipped_messages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            if processor is None:
                row, skipped_count = dict(item), 0
            else:
                row, skipped_count = processor(item, dataset_name)
            if row is None:
                continue
            skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_messages


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor | None,
    dataset_name: str,
    *,
    eval_dataset: Iterable[Mapping[str, Any]] | None = None,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{dataset_name}_train.jsonl"
    if train_output_path.exists():
        print(f"Dataset already exists at {train_output_path}; skipping conversion.")
        return train_output_path

    skipped_messages = _write_split(
        dataset,
        train_output_path,
        processor,
        dataset_name,
    )
    if eval_dataset is not None:
        eval_output_path = output_directory / f"{dataset_name}_test.jsonl"
        skipped_messages += _write_split(
            eval_dataset,
            eval_output_path,
            processor,
            dataset_name,
        )

    if skipped_messages:
        print(
            f"Skipped {skipped_messages} unsupported messages while "
            f"processing {dataset_name}."
        )
    print(f"Saved {dataset_name} training data to {train_output_path}.")
    return train_output_path


def process_and_save_ds(
    train_ds: Iterable[Mapping[str, Any]],
    test_ds: Iterable[Mapping[str, Any]] | None,
    output_path: Path,
    proc_fn: RowProcessor | None,
    dataset_name: str,
) -> Path:
    """Backward-compatible wrapper for external data preparation callers."""
    return process_and_save_dataset(
        train_ds,
        output_path,
        proc_fn,
        dataset_name,
        eval_dataset=test_ds,
    )


def prepare_mixture(
    dataset_name: str,
    sample_size: int,
    ratios: Sequence[Decimal | int],
    output_directory: Path,
    *,
    seed: int = 42,
    split_eval: bool = False,
) -> Path:
    counts = mixture_counts(sample_size, ratios)
    rows: list[dict[str, Any]] = []
    # Direct script execution puts scripts/, not the checkout root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from draftfit.data.prepare import normalize_row

    for source, count in zip(MIX_SOURCES, counts):
        if not count:
            continue
        dataset, processor = load_dataset_preset(source, streaming=True)
        if hasattr(dataset, "__len__") and len(dataset) < count:
            raise ValueError(f"{source} has {len(dataset)} rows; requested {count}")
        selected = 0
        excluded = 0
        seen_ids: set[str] = set()
        shuffled = (
            dataset.shuffle(seed=seed)
            if hasattr(dataset, "__len__")
            else dataset.shuffle(seed=seed, buffer_size=1_000)
        )
        for item in shuffled:
            try:
                row, _ = processor(item, source) if processor else (dict(item), 0)
                if row is None:
                    continue
                row = normalize_row(row)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                excluded += 1
                continue
            row["id"] = f"{source}:{row['id']}"
            if row["id"] in seen_ids:
                excluded += 1
                continue
            seen_ids.add(row["id"])
            rows.append(row)
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(f"{source} has only {selected} usable unique rows; requested {count}")
        print(f"Selected {count} rows from {source}; excluded {excluded} incomplete or duplicate rows.")
    random.Random(seed).shuffle(rows)
    if len(rows) != sample_size:
        raise ValueError(f"expected {sample_size} output rows, got {len(rows)}")
    output_directory.mkdir(parents=True, exist_ok=True)
    train_path = output_directory / f"{dataset_name}_train.jsonl"
    eval_path = output_directory / f"{dataset_name}_test.jsonl"
    if train_path.exists() or (split_eval and eval_path.exists()):
        raise FileExistsError("mixture output already exists; choose a fresh --output-path")
    if split_eval:
        if sample_size < 2:
            raise ValueError("--split-eval requires at least two rows")
        eval_count = max(1, round(sample_size * 0.05))
        train_rows, eval_rows = rows[eval_count:], rows[:eval_count]
    else:
        train_rows, eval_rows = rows, []
    written = []
    try:
        for path, selected in ((train_path, train_rows), (eval_path, eval_rows)):
            if path == eval_path and not split_eval:
                continue
            with path.open("x", encoding="utf-8") as handle:
                written.append(path)
                for row in selected:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        for path in written:
            path.unlink()
        raise
    print(f"Saved {len(train_rows)} train and {len(eval_rows)} eval rows to {output_directory}.")
    return train_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dataset in {"apigen-mt", "nemotron-agentic"}:
        ratios = tuple(100 if source == args.dataset else 0 for source in MIX_SOURCES)
        prepare_mixture(
            args.dataset, args.sample_size, ratios, args.output_path,
            seed=args.seed, split_eval=args.split_eval,
        )
        return
    if args.dataset in (*MIX_PRESETS, "mix"):
        prepare_mixture(
            args.dataset, args.sample_size, args.ratios, args.output_path,
            seed=args.seed, split_eval=args.split_eval,
        )
        return
    dataset, processor = load_dataset_preset(
        args.dataset,
        data_path=args.data_path,
        opc_subset=args.opc_subset,
    )

    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))
        print(f"Processing {args.sample_size} samples from {args.dataset}.")

    eval_dataset = None
    if args.split_eval:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    process_and_save_dataset(
        dataset,
        args.output_path,
        processor,
        args.dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    main()
