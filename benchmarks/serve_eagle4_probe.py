"""Native vLLM greedy correctness probe; run each arm in a separate process.

Requires the upstream patched vLLM/eagle-harness environment. This small static
tree checks real export loading and requests, not calibrated speed claims.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--arm", choices=("target", "eagle4"), required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    free, _ = torch.cuda.mem_get_info()
    if free < (5600 + 1536) * 2**20:
        raise RuntimeError("Need 5,600 MiB budget plus 1,536 MiB reserve")
    prompts = [
        TokensPrompt(
            prompt_token_ids=torch.load(p, weights_only=True)["prompt_ids"][0].tolist()
        )
        for p in sorted((args.comparison / "test_sft").glob("*.pt"))[: args.samples]
    ]
    if len(prompts) != args.samples:
        raise ValueError("Not enough captured held-out prompts")
    options = dict(
        model=args.target,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        gpu_memory_utilization=0.06,
        kv_cache_memory_bytes=128 * 2**20,
        seed=29,
        attention_backend="CUSTOM",
        enable_prefix_caching=False,
        trust_remote_code=False,
    )
    if args.arm == "eagle4":
        options.update(
            speculative_config=dict(
                method="eagle_harness",
                model=str(args.comparison / "eagle4" / "export"),
                method_config=dict(
                    mode="static", node_top_k=2, tree_depth=3, max_verify_width=8
                ),
            ),
        )
    model = LLM(**options)
    outputs = model.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=args.max_new_tokens),
        use_tqdm=False,
    )
    records = [
        dict(
            prompt_token_ids=r.prompt_token_ids,
            token_ids=list(r.outputs[0].token_ids),
            finish_reason=r.outputs[0].finish_reason,
        )
        for r in outputs
    ]
    destination = args.comparison / "native-serving"
    destination.mkdir(exist_ok=True)
    (destination / f"{args.arm}.json").write_text(
        json.dumps(
            dict(
                arm=args.arm,
                settings=options,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                arm=args.arm,
                requests=len(records),
                generated_tokens=sum(len(r["token_ids"]) for r in records),
            )
        ),
        flush=True,
    )
    if args.arm == "eagle4":
        baseline = json.loads((destination / "target.json").read_text())["records"]
        equal = len(baseline) == len(records) and all(
            a["prompt_token_ids"] == b["prompt_token_ids"]
            and a["token_ids"] == b["token_ids"]
            for a, b in zip(baseline, records)
        )
        (destination / "parity.json").write_text(
            json.dumps(
                dict(
                    greedy_equal=equal,
                    samples=len(records),
                    tree_depth=3,
                    node_top_k=2,
                    max_verify_width=8,
                    calibrated=False,
                ),
                indent=2,
            )
            + "\n"
        )
        if not equal:
            raise RuntimeError(
                "Native EAGLE4 greedy tokens differ from native target baseline"
            )


if __name__ == "__main__":
    main()
