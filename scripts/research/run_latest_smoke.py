"""Execute latest-method recipes through the real CLI using tiny local models.

Run from the repository root with the documented CPU environment:
    python scripts/research/run_latest_smoke.py --output outputs/latest-methods-smoke
No remote weights, tools, engine installation or paid hardware are used.
This functional fixture does not measure paper-model quality or serving speed.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled"
    )

    import torch
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from draftfit.modeling.draft.dflash import DFlashDraftModel

    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    target = Qwen3ForCausalLM(cfg).eval()
    target.save_pretrained(root / "target")
    backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="0", pad_token="0"
    )
    tokenizer.save_pretrained(root / "target")
    draft_cfg = Qwen3Config(
        architectures=["DFlashDraftModel"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_target_layers=3,
        block_size=4,
        dflash_config={"target_layer_ids": [0, 2], "mask_token_id": 31},
    )
    draft_cfg._attn_implementation = "sdpa"
    DFlashDraftModel(draft_cfg).save_pretrained(root / "draft")
    prompt = [1, 2, 3, 4, 5, 6]
    with torch.inference_mode():
        gold = target.generate(
            torch.tensor([prompt]),
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(prompt) :].tolist()
    # A separate target fixture guarantees EOS-inclusive ARR fixed points.
    with torch.no_grad():
        target.lm_head.weight.zero_()
    target.save_pretrained(root / "arr-target")
    tokenizer.save_pretrained(root / "arr-target")

    def rows(name, values):
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(v) + "\n" for v in values))
        return {"input_path": str(path), "max_samples": len(values)}

    model = {"target_model_path": str(root / "target")}
    diagnostic = {
        "compare_ar": True,
        "record_acceptance": True,
        "warmup": 1,
        "repetitions": 3,
        "position_bin_size": 4,
    }
    shared = {
        "model": model,
        "generation": {"max_new_tokens": 8},
        "diagnostics": diagnostic,
    }
    recipes = {
        "ngramrl": {
            "method": "ngramrl",
            **shared,
            "data": rows(
                "rollouts",
                [
                    {"input_ids": prompt, "prompt_id": "fixture", "step": t}
                    for t in range(3)
                ],
            ),
        },
        "specmem": {
            "method": "specmem",
            **shared,
            "generation": {"max_new_tokens": 8, "stop_token_ids": [gold[0]]},
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
        "rift": {
            "method": "rift",
            **shared,
            "model": {**model, "draft_model_path": str(root / "target")},
            "cache": {"context_budget": 2, "draft_tokens": 4},
            "data": rows("rift", [{"input_ids": prompt}]),
        },
        "valuspec": {
            "method": "cast",
            "model": {**model, "draft_checkpoint_path": str(root / "draft")},
            "data": rows("cast", [{"input_ids": prompt}]),
            "generation": {"max_new_tokens": 8},
            "speculation": {"budget": 7, "branch_k": 2},
            "valuation": {
                "scorer": model,
                "calibration": rows("calibration", [{"input_ids": [2, 4, 3]}]),
            },
        },
        "arr": {
            "method": "arr",
            "model": {"target_model_path": str(root / "arr-target")},
            "data": rows("questions", [{"input_ids": [1, 2]}]),
            "generation": {"max_new_tokens": 3, "stop_token_ids": [0]},
            "refinement": {
                "active_roots": 2,
                "max_roots": 3,
                "root_temperature": 0,
                "max_processed_positions": 1000,
                "revision_template": "{question} {solution}",
                "answer_extractor": "token_sequence",
            },
        },
        "treewy": {
            "method": "treewy",
            "data": rows(
                "tree",
                [
                    {
                        "q": [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]],
                        "k": [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, -1.0]]],
                        "v": [[[2.0]], [[3.0]], [[4.0]]],
                        "g": [[-0.1], [-0.2], [-0.3]],
                        "beta": [[0.5], [0.5], [0.5]],
                        "parent": [-1, 0, 0],
                        "initial_state": [[[0.0, 0.0]]],
                        "accepted_node": 1,
                    }
                ],
            ),
        },
        "moe_spac": {
            "method": "moe_spac",
            "policy": dict(
                gamma=4,
                max_utility=2,
                target_layers=1,
                expert_bytes=64,
                cpu_unit_ms=1.0,
                gpu_unit_ms=1.0,
                io_unit_ms=0.0,
                draft_unit_ms=0.0,
            ),
            "data": rows(
                "routes",
                [
                    {
                        "layer": 0,
                        "frequencies": [4, 0],
                        "resident_expert_ids": [],
                        "ratios": [{"cpu_ratio": 0.5, "gpu_ratio": 0.5}] * 2,
                        "free_bytes": 128,
                    }
                ],
            ),
        },
        "query_layout": {
            "method": "query_layout",
            "data": rows(
                "layout",
                [
                    {
                        "q": [[[1.0, 0.0]]],
                        "k": [[[1.0, 0.0]], [[0.0, 1.0]]],
                        "v": [[[2.0]], [[3.0]]],
                        "mask": [[0.0, 0.0]],
                    }
                ],
            ),
        },
    }
    summary = {
        "scope": "tiny_local_cpu_functional_fixtures",
        "paper_reproduction": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hardware": platform.machine(),
        "results": {},
    }
    for name, recipe in recipes.items():
        recipe["output_dir"] = str(root / (name + "-result"))
        path = root / (name + ".json")
        path.write_text(json.dumps(recipe, indent=2) + "\n")
        result = subprocess.run(
            [sys.executable, "-m", "draftfit.research", "-c", str(path)],
            text=True,
            capture_output=True,
        )
        (root / (name + ".log")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")
        payload = json.loads(result.stdout)
        if name in {"ngramrl", "specmem", "rift", "valuspec"}:
            expected = gold[:1] if name == "specmem" else gold
            if not all(
                answer["token_ids"] == expected for answer in payload["answers"]
            ):
                raise RuntimeError(f"{name} changed target greedy output")
        summary["results"][name] = {
            "exit_code": result.returncode,
            "recipe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "result_path": str(Path(recipe["output_dir"]) / "result.json"),
        }
        print(f"{name}: passed", flush=True)
    summary["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("draftfit/research").rglob("*"))
        if path.is_file() and "__pycache__" not in str(path)
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
