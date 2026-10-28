"""Bounded non-oracle effect probes in an existing <=1B CUDA environment.

Memory is populated only by an earlier real request, never by an evaluation
answer or schema oracle. These synthetic reference probes are not a workload
benchmark, a semantic-retrieval reproduction or a production speedup gate.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_latest_pretrained_gpu import gpu_snapshot, save

COUNT = "Count from 1 to 5, separated by commas. Output only the list."
CASES = [
    {"id": "cold", "text": COUNT, "warm": False, "user": "alice"},
    {"id": "repeat", "text": COUNT, "warm": True, "user": "alice"},
    {
        "id": "paraphrase",
        "text": "Write the numbers 1 through 5, separated by commas. Output only the list.",
        "warm": True,
        "user": "alice",
    },
    {
        "id": "changed_numbers",
        "text": "Count from 6 to 10, separated by commas. Output only the list.",
        "warm": True,
        "user": "alice",
    },
    {
        "id": "new_text",
        "text": "Return exactly: red, green, blue.",
        "warm": True,
        "user": "alice",
    },
    {"id": "new_user", "text": COUNT, "warm": True, "user": "bob"},
]
MATH = [("2 + 3", "5"), ("7 - 4", "3"), ("3 * 4", "12"), ("8 / 2", "4")]


def observation():
    snapshot = gpu_snapshot()
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    snapshot["external_processes"] = [
        row
        for row in result.stdout.splitlines()
        if row.split(",", 1)[0] != str(os.getpid())
    ]
    return snapshot


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def embedding(text):
    """Fixed lexical control: 128 hashed word-count bins, no learned encoder."""
    vector = [0.0] * 128
    for word in re.findall(r"\w+", text.lower()):
        index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
        vector[index % len(vector)] += 1.0
    return vector


def chat(tokenizer, text):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, add_special_tokens=False).input_ids


def models(args, count=1):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    free = int(gpu_snapshot()["memory.free"]) / 1024
    if free < args.min_free_gib:
        raise RuntimeError("insufficient free VRAM; no model was loaded")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.max_gib * 1024**3 / total)
    torch.set_num_threads(2)
    result = []
    for _ in range(count):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        if (
            model.config.model_type != "qwen3"
            or sum(p.numel() for p in model.parameters()) > 1_000_000_000
        ):
            raise ValueError("this probe requires a Qwen3 target no larger than 1B")
        result.append(model.to("cuda:0").eval().requires_grad_(False))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    eos = result[0].generation_config.eos_token_id
    args.eos = [eos] if isinstance(eos, int) else list(eos)
    return result, tokenizer


def decode(args, target, ids, proposer):
    import torch

    from draftfit.research.reference_decode import generate

    output, stats = generate(
        target,
        torch.tensor([ids], device=target.device),
        proposer,
        max_new_tokens=args.max_tokens,
        stop_token_ids=args.eos,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
    return output[0, len(ids) :].tolist(), stats


def measured(callback):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokens, stats, extra = callback()
    torch.cuda.synchronize()
    return {
        "token_ids": tokens,
        "seconds": time.perf_counter() - started,
        "target_forwards": stats["target_forwards"],
        "accepted_tokens": stats["accepted_tokens"],
        "processed_target_positions": stats["processed_target_positions"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        **extra,
    }


def pairs(args, baseline, candidate):
    rows = []
    for repeat in range(-1, args.repetitions):
        order = ["ar", "candidate"] if repeat % 2 == 0 else ["candidate", "ar"]
        observed = {"before": observation()}
        values = {}
        for arm in order:
            # Restore the same causal history outside the measured request.
            callback = baseline() if arm == "ar" else candidate()
            values[arm] = measured(callback)
        observed["after"] = observation()
        if values["ar"]["token_ids"] != values["candidate"]["token_ids"]:
            raise ValueError("candidate changed exact target greedy output")
        if repeat >= 0:
            rows.append({"repeat": repeat, "order": order, **values, "gpu": observed})
    return rows


def memory_probe(args):
    from draftfit.research.memory import NgramMemory, Proposal, SpecMemMemory

    (target,), tokenizer = models(args)
    prime_ids = chat(tokenizer, COUNT)
    prior_tokens, prior_stats = decode(args, target, prime_ids, lambda *_: Proposal())
    ngram, specmem = NgramMemory(), SpecMemMemory()
    key = digest(prime_ids)
    ngram.add_verified(key, prime_ids + prior_tokens, step=0)
    complete = bool(prior_tokens and prior_tokens[-1] in args.eos)
    if complete:
        specmem.add_verified("alice", key, embedding(COUNT), prior_tokens)
    result = {
        "scope": "non_oracle_lexical_memory_reference",
        "embedding": "sha256_128_bin_lexical_word_counts",
        "prime": {
            "token_ids": prior_tokens,
            "eos_complete": complete,
            "stats": prior_stats,
        },
        "schema_oracle_used": False,
        "cases": [],
    }
    for method in ["ngramrl", "specmem"]:
        for case in CASES:
            if method == "ngramrl" and case["id"] == "new_user":
                continue
            ids = chat(tokenizer, case["text"])

            def baseline():
                def execute():
                    tokens, stats = decode(args, target, ids, lambda *_: Proposal())
                    return tokens, stats, {}

                return execute

            def candidate():
                memory = NgramMemory() if method == "ngramrl" else SpecMemMemory()
                if case["warm"]:
                    source = ngram if method == "ngramrl" else specmem
                    memory.load_state_dict(copy.deepcopy(source.state_dict()))

                def execute():
                    if method == "ngramrl":
                        snapshot = memory.snapshot(digest(ids), step=1)
                        source = {"source": "rollout_snapshot"}

                        def propose(context, remaining, rng):
                            return snapshot.propose(context, min(8, remaining))

                    else:
                        vector = embedding(case["text"])
                        cached, source = memory.retrieve(case["user"], vector)

                        def propose(context, remaining, rng):
                            committed = context[len(ids) :]
                            if tuple(committed) != cached[: len(committed)]:
                                return Proposal()
                            tokens = cached[
                                len(committed) : len(committed) + min(32, remaining)
                            ]
                            return Proposal(tokens, tuple({t: 1.0} for t in tokens))

                    tokens, stats = decode(args, target, ids, propose)
                    if method == "ngramrl":
                        memory.add_verified(digest(ids), ids + tokens, step=1)
                    elif tokens and tokens[-1] in args.eos:
                        memory.add_verified(case["user"], digest(ids), vector, tokens)
                    return tokens, stats, {"retrieval": source}

                return execute

            rows = pairs(args, baseline, candidate)
            result["cases"].append({"method": method, **case, "rows": rows})
            save(args.output / "memory.json", result)
            print(method + "/" + case["id"] + ": passed parity", flush=True)
    return result


def rift_probe(args):
    from draftfit.research.config import RIFTOptions
    from draftfit.research.memory import Proposal
    from draftfit.research.rift import CompressedARProposer

    (target, draft), tokenizer = models(args, 2)
    result = {"scope": "same_model_rift_cache_ablation", "cases": []}
    for text in [COUNT, CASES[3]["text"], CASES[4]["text"]]:
        ids = chat(tokenizer, text)
        for budget in [16, target.config.max_position_embeddings]:

            def baseline():
                def execute():
                    tokens, stats = decode(args, target, ids, lambda *_: Proposal())
                    return tokens, stats, {}

                return execute

            def candidate():
                proposer = CompressedARProposer(
                    draft, RIFTOptions(context_budget=budget, draft_tokens=4)
                )

                def execute():
                    tokens, stats = decode(args, target, ids, proposer)
                    cache = [
                        {
                            k: row[k]
                            for k in [
                                "dense_history_kv_bytes_before_gather",
                                "active_context_kv_bytes_after_gather",
                                "draft_forwards",
                            ]
                        }
                        for row in proposer.rounds
                    ]
                    return tokens, stats, {"draft_cache_rounds": cache}

                return execute

            rows = pairs(args, baseline, candidate)
            result["cases"].append(
                {"text": text, "context_budget": budget, "rows": rows}
            )
            save(args.output / "rift.json", result)
            print("rift/" + str(budget) + ": passed parity", flush=True)
    return result


def arr_probe(args):
    import gc

    import torch

    from draftfit.research.arr import run_arr
    from draftfit.research.config import ARRConfig
    from draftfit.research.memory import Proposal

    (target,), tokenizer = models(args)
    rows, baselines = [], []
    for index, (question, expected) in enumerate(MATH):
        text = "What is " + question + "? Reply with only the integer."
        ids = chat(tokenizer, text)
        tokens, stats = decode(args, target, ids, lambda *_: Proposal())
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        rows.append({"id": index, "input_ids": ids})
        baselines.append(
            {
                "text": text,
                "expected": expected,
                "answer": answer,
                "correct": answer == expected,
                "stats": stats,
            }
        )
    del target
    gc.collect()
    torch.cuda.empty_cache()
    path = args.output / "math-input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = ARRConfig(
        method="arr",
        model={"target_model_path": args.model_path, "torch_dtype": "float32"},
        data={"input_path": str(path), "max_samples": len(rows)},
        refinement={"active_roots": 2, "max_roots": 3, "max_processed_positions": 6000},
        generation={"max_new_tokens": args.max_tokens, "stop_token_ids": args.eos},
        runtime={"device": "cuda:0", "seed": 42},
        output_dir=str(args.output / "arr-native"),
    )
    native = run_arr(config)
    for baseline, answer in zip(baselines, native["answers"]):
        baseline["arr_winner"] = answer["winner"]
        baseline["arr_correct"] = answer["winner"] == baseline["expected"]
        baseline["arr_processed_positions"] = answer["processed_target_positions"]
    result = {
        "scope": "four_toy_quality_questions_not_a_benchmark",
        "comparisons": baselines,
        "native": native,
        "config": config.model_dump(),
    }
    save(args.output / "arr.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gib", type=float, default=5.5)
    parser.add_argument("--min-free-gib", type=float, default=7.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", choices=["memory", "rift", "arr"])
    args = parser.parse_args()
    if not (
        0 < args.max_gib < args.min_free_gib
        and 3 <= args.repetitions <= 10
        and 1 <= args.max_tokens <= 32
    ):
        parser.error(
            "require bounded memory, at least three repeats and <=32 output tokens"
        )
    args.model_path = str(Path(args.model_path).resolve())
    args.output = args.output.resolve()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    if args.worker:
        {"memory": memory_probe, "rift": rift_probe, "arr": arr_probe}[args.worker](
            args
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "synthetic_shared_gpu_reference_effects",
        "production_speed_claim": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("draftfit/research").rglob("*.py")]
        },
        "workload_sha256": digest([CASES, MATH]),
        "packages": {
            k: importlib.metadata.version(k)
            for k in ["torch", "transformers", "pydantic"]
        },
        "initial_gpu": observation(),
        "cases": {},
    }
    for worker in ["memory", "rift", "arr"]:
        command = [
            sys.executable,
            __file__,
            "--model-path",
            args.model_path,
            "--output",
            str(args.output),
            "--max-gib",
            str(args.max_gib),
            "--min-free-gib",
            str(args.min_free_gib),
            "--max-tokens",
            str(args.max_tokens),
            "--repetitions",
            str(args.repetitions),
            "--worker",
            worker,
        ]
        with (
            (args.output / (worker + ".stdout.log")).open("w") as stdout,
            (args.output / (worker + ".stderr.log")).open("w") as stderr,
        ):
            try:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=240
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        summary["cases"][worker] = {"exit_code": code}
        save(args.output / "summary.json", summary)
        print(worker + ": exit " + str(code), flush=True)
        if code:
            break
    summary["final_gpu"] = observation()
    save(args.output / "summary.json", summary)
    return int(any(row["exit_code"] for row in summary["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
