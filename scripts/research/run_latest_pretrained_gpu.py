"""Bounded pretrained GPU checks for the latest native research references.

Run from a source checkout in an already provisioned CUDA environment. Supply a
local, pinned Qwen3 snapshot (<=1B parameters); no dependencies or weights are
downloaded by this script. Every case runs in its own process and result folder.
The same-size RIFT draft and untrained DFlash are correctness controls, not
evidence of draft quality, paper reproduction or production speedup.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

CASES = (
    "ngramrl",
    "ngramrl_resume",
    "specmem",
    "specmem_resume",
    "rift",
    "valuspec",
    "arr",
    "treewy",
    "query_layout",
    "moe_spac",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_snapshot():
    fields = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    rows = [line.split(", ") for line in result.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise ValueError(
            "this shared-capacity gate requires exactly one visible physical GPU"
        )
    row = rows[0]
    return dict(zip(fields.split(","), row))


def prepare(root, model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    from draftfit.modeling.draft.dflash import DFlashDraftModel
    from draftfit.research.memory import Proposal
    from draftfit.research.reference_decode import generate

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    target = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to("cuda:0")
        .eval()
    )
    count = sum(p.numel() for p in target.parameters())
    if target.config.model_type != "qwen3" or count > 1_000_000_000:
        raise ValueError("this gate requires a real Qwen3 target no larger than 1B")
    eos = target.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)

    def chat(text):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer(text, add_special_tokens=False).input_ids

    prompt = chat("Count from 1 to 5, separated by commas. Output only the list.")
    calibration = chat("What is the capital of France? Answer with one word.")
    question = chat("What is 1 + 1? Reply with only the single digit.")
    with torch.inference_mode():
        output, stats = generate(
            target,
            torch.tensor([prompt], device="cuda:0"),
            lambda *_: Proposal(),
            max_new_tokens=32,
            stop_token_ids=eos,
            temperature=0,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
    gold = output[0, len(prompt) :].tolist()
    save(
        root / "gold.json",
        {
            "input_ids": prompt,
            "token_ids": gold,
            "decoded": tokenizer.decode(gold),
            "eos_terminated": bool(gold and gold[-1] in eos),
            "target_parameter_count": count,
            "model_path": model_path,
            "stats": stats,
        },
    )
    cfg = target.config.to_dict()
    cfg.update(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_target_layers=target.config.num_hidden_layers,
        block_size=4,
        dflash_config={
            "target_layer_ids": [1, target.config.num_hidden_layers - 2],
            "mask_token_id": tokenizer.pad_token_id,
        },
    )
    draft_cfg = Qwen3Config(**cfg)
    draft_cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    draft = DFlashDraftModel(draft_cfg).eval()
    draft_count = sum(p.numel() for p in draft.parameters())
    if draft_count > 1_000_000_000:
        raise ValueError("draft exceeds the user model-size limit")
    draft.save_pretrained(root / "untrained-dflash")
    save(
        root / "draft-identity.json",
        {
            "parameter_count": draft_count,
            "trained": False,
            "seed": 7,
            "purpose": "verifier_and_filter_correctness_only",
        },
    )

    def rows(name, values):
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(v) + "\n" for v in values))
        return {"input_path": str(path), "max_samples": len(values)}

    model = {"target_model_path": model_path, "torch_dtype": "float32"}
    shared = {
        "model": model,
        "runtime": {"device": "cuda:0", "seed": 42},
        "generation": {"max_new_tokens": 32, "stop_token_ids": eos},
        "diagnostics": {
            "compare_ar": True,
            "record_acceptance": True,
            "warmup": 1,
            "repetitions": 2,
            "position_bin_size": 8,
        },
    }
    ng_memory = {"draft_tokens": 8}
    sm_memory = {"draft_tokens": 32}
    recipes = {
        "ngramrl": {
            "method": "ngramrl",
            **shared,
            "memory": ng_memory,
            "data": rows(
                "rollouts",
                [
                    {"input_ids": prompt, "prompt_id": "count", "step": t}
                    for t in range(3)
                ],
            ),
        },
        "ngramrl_resume": {
            "method": "ngramrl",
            **shared,
            "memory": {
                **ng_memory,
                "state_path": str(root / "ngramrl-result/memory.json"),
            },
            "data": rows(
                "rollouts-resumed",
                [{"input_ids": prompt, "prompt_id": "count", "step": 3}],
            ),
        },
        "specmem": {
            "method": "specmem",
            **shared,
            "memory": sm_memory,
            "data": rows(
                "users",
                [
                    {
                        "input_ids": prompt,
                        "user_id": "alice",
                        "query_embedding": [1.0, 0.0],
                        "schema_draft_tokens": gold,
                    },
                    {
                        "input_ids": prompt,
                        "user_id": "bob",
                        "query_embedding": [1.0, 0.0],
                    },
                    {
                        "input_ids": prompt,
                        "user_id": "alice",
                        "query_embedding": [1.0, 0.0],
                    },
                ],
            ),
        },
        "specmem_resume": {
            "method": "specmem",
            **shared,
            "memory": {
                **sm_memory,
                "state_path": str(root / "specmem-result/memory.json"),
            },
            "data": rows(
                "users-resumed",
                [
                    {
                        "input_ids": prompt,
                        "user_id": "alice",
                        "query_embedding": [1.0, 0.0],
                    }
                ],
            ),
        },
        "rift": {
            "method": "rift",
            **shared,
            "model": {**model, "draft_model_path": model_path},
            "cache": {
                "context_budget": 16,
                "draft_tokens": 4,
                "observation_window": 16,
            },
            "data": rows("rift", [{"input_ids": prompt}]),
        },
        "valuspec": {
            "method": "cast",
            "model": {**model, "draft_checkpoint_path": str(root / "untrained-dflash")},
            "runtime": shared["runtime"],
            "generation": shared["generation"],
            "data": rows("cast", [{"input_ids": prompt}]),
            "speculation": {"budget": 7, "branch_k": 2},
            "valuation": {
                "scorer": model,
                "calibration": rows("calibration", [{"input_ids": calibration}]),
            },
        },
        "arr": {
            "method": "arr",
            "model": model,
            "runtime": shared["runtime"],
            "data": rows("questions", [{"input_ids": question}]),
            "generation": {"max_new_tokens": 16, "stop_token_ids": eos},
            "refinement": {
                "active_roots": 2,
                "max_roots": 2,
                "root_temperature": 0.7,
                "max_processed_positions": 12000,
                "replay_width": 8,
                "revision_template": "<|im_start|>user\nReview this original question and answer. Return only the corrected single-digit answer.\nOriginal conversation:\n{question}\nPrevious answer:\n{solution}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            },
        },
    }
    from draftfit.research.config import load_recipe

    for name in ("treewy", "query_layout", "moe_spac"):
        filename = {
            "query_layout": "query-layout.yaml",
            "moe_spac": "moe-spac.yaml",
        }.get(name, name + ".yaml")
        recipe = load_recipe("examples/configs/research/" + filename).model_dump()
        recipe["data"]["input_path"] = str(Path(recipe["data"]["input_path"]).resolve())
        if "runtime" in recipe:
            recipe["runtime"]["device"] = "cuda:0"
        recipes[name] = recipe
    for name, recipe in recipes.items():
        recipe["output_dir"] = str(root / (name + "-result"))
        save(root / (name + ".json"), recipe)


def worker(args):
    before = gpu_snapshot()
    if float(before["memory.free"]) < args.min_free_gib * 1024:
        save(
            args.output / (args.worker + "-resource.json"),
            {"before": before, "execution": "skipped_insufficient_headroom"},
        )
        return 75
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.max_gib * 1024**3 / total, device=0)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        if args.worker == "prepare":
            prepare(args.output, args.model_path)
        else:
            sys.argv = [
                "draftfit.research",
                "-c",
                str(args.output / (args.worker + ".json")),
            ]
            runpy.run_module("draftfit.research", run_name="__main__")
    finally:
        torch.cuda.synchronize()
        save(
            args.output / (args.worker + "-resource.json"),
            {
                "before": before,
                "after": gpu_snapshot(),
                "elapsed_process_seconds": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocator_limit_gib": args.max_gib,
                "includes_cuda_context_or_other_processes": False,
            },
        )
    return 0


def inspect_result(name, result, gold):
    evidence = {}
    if name in {
        "ngramrl",
        "ngramrl_resume",
        "specmem",
        "specmem_resume",
        "rift",
        "valuspec",
    }:
        equal = all(a["token_ids"] == gold["token_ids"] for a in result["answers"])
        evidence["target_greedy_tokens_equal"] = equal
        if not equal:
            raise ValueError(
                "generated tokens differ from the identical pretrained target AR reference"
            )
    if name == "arr":
        for answer in result["answers"]:
            if answer["no_vote"] or not answer["roots"]:
                raise ValueError("ARR produced no complete voting root")
            if answer["processed_target_positions"] > answer["budget_limit"]:
                raise ValueError("ARR exceeded its actual processed-position budget")
        evidence["complete_root_vote_and_budget"] = True
    if name == "query_layout":
        if not all(a["admitted"] for a in result["answers"]):
            raise ValueError("the intended CUDA layout comparison was not admitted")
        evidence["max_abs_error"] = max(a["max_abs_error"] for a in result["answers"])
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-gib", type=float, default=5.5)
    parser.add_argument("--min-free-gib", type=float, default=7.0)
    parser.add_argument("--worker", choices=("prepare", *CASES))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.model_path = str(Path(args.model_path).resolve())
    if not (0 < args.max_gib < args.min_free_gib):
        parser.error("require 0 < allocator limit < minimum free VRAM")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
    )
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "bounded_pretrained_shared_gpu_reference_checks",
        "paper_reproduction": False,
        "production_speed_claim": False,
        "python": platform.python_version(),
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("torch", "transformers", "pydantic", "safetensors")
        },
        "initial_gpu": gpu_snapshot(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("draftfit").rglob("*.py"))
        },
        "cases": {},
    }
    for name in ("prepare", *CASES):
        started = time.monotonic()
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
            "--worker",
            name,
        ]
        try:
            with (
                (args.output / (name + ".stdout.log")).open("w") as stdout,
                (args.output / (name + ".stderr.log")).open("w") as stderr,
            ):
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=180
                )
            record = {
                "exit_code": process.returncode,
                "elapsed_seconds": time.monotonic() - started,
            }
            if process.returncode == 0 and name != "prepare":
                record.update(
                    inspect_result(
                        name,
                        json.loads(
                            (args.output / (name + "-result/result.json")).read_text()
                        ),
                        json.loads((args.output / "gold.json").read_text()),
                    )
                )
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            record = {
                "exit_code": -1,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            }
        summary["cases"][name] = record
        save(args.output / "summary.json", summary)
        print(name + ": " + json.dumps(record), flush=True)
        if name == "prepare" and record["exit_code"]:
            break
        if record["exit_code"] == 75:
            break
    summary["final_gpu"] = gpu_snapshot()
    save(args.output / "summary.json", summary)
    return int(any(v["exit_code"] for v in summary["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
