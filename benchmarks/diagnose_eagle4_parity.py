"""Replay one retained greedy mismatch without loading a draft model.

Compare the target on the same causal prefix at two sequence lengths. CPU
replays are controls; they cannot certify the original CUDA BF16 execution.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM

from benchmarks.train_eagle4_comparison import LAYERS, digest, memory_guard, write_json
from draftfit.offline_capture.transformers import OfflineTransformersCapture


@torch.inference_mode()
def replay(args):
    root = args.comparison
    baseline_path = root / args.result_dir / "target.json"
    actual_path = root / args.result_dir / f"{args.algorithm}.json"
    reference = json.loads(baseline_path.read_text())[args.sample]
    actual = json.loads(actual_path.read_text())[args.sample]
    j = next(
        (
            i
            for i, (a, b) in enumerate(zip(reference["token_ids"], actual["token_ids"]))
            if a != b
        ),
        None,
    )
    if j is None:
        raise ValueError("This sample has no differing token to replay")
    prompt = torch.load(
        sorted((root / "test_sft").glob("*.pt"))[args.sample], weights_only=True
    )["prompt_ids"][0].tolist()
    start = 0
    for trace in actual["traces"]:
        if j < start + trace["accepted_tokens"] + 2:
            break
        start += trace["accepted_tokens"] + 2
    else:
        raise ValueError("The mismatch is outside retained verification traces")
    if j <= start:
        raise ValueError("The mismatch is an anchor, not a block verification token")
    prefix = prompt + actual["token_ids"][:start]
    verify = prefix + [actual["token_ids"][start]] + trace["proposed_token_ids"]
    short = prompt + reference["token_ids"][:j]
    if verify[: len(short)] != short:
        raise ValueError("The retained block does not share the first-mismatch prefix")
    if args.device == "cuda":
        memory_guard(args)
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target,
            dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(args.device)
        .eval()
        .requires_grad_(False)
    )
    teacher = OfflineTransformersCapture(target)
    teacher.set_capture_layers(LAYERS, capture_method="dspark")

    def capture(ids):
        ids = torch.tensor([ids], device=args.device)
        return teacher.capture(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            loss_mask=torch.zeros_like(ids),
        ).last_hidden_states

    a, b = capture(short), capture(verify)
    position = len(short) - 1
    usual = target.lm_head(a[:, -1:]).float()[0, 0]
    single = target.lm_head(b[:, position : position + 1]).float()[0, 0]
    multiple = target.lm_head(b[:, len(prefix) :]).float()[0, j - start - 1]

    def scores(logits):
        values, ids = logits.topk(2)
        return {
            "argmax": int(logits.argmax()),
            "top_two_ids": ids.tolist(),
            "top_two_logits": values.tolist(),
            "top_two_gap": float(values[0] - values[1]),
            "reference_token_logprob": float(
                logits.log_softmax(-1)[reference["token_ids"][j]]
            ),
            "observed_token_logprob": float(
                logits.log_softmax(-1)[actual["token_ids"][j]]
            ),
        }

    result = {
        "algorithm": args.algorithm,
        "sample": args.sample,
        "first_divergence": j,
        "device": args.device,
        "dtype": args.dtype,
        "baseline_sha256": digest(baseline_path),
        "actual_sha256": digest(actual_path),
        "reference_token": reference["token_ids"][j],
        "observed_token": actual["token_ids"][j],
        "prefix_tokens": len(short),
        "verification_tokens": len(verify),
        "canonical": scores(usual),
        "block_single_head": scores(single),
        "block_multiple_head": scores(multiple),
        "hidden_max_difference": float((a[:, -1] - b[:, position]).abs().max()),
        "head_shape_max_difference": float((single - multiple).abs().max()),
        "same_argmax_across_shapes": int(usual.argmax()) == int(multiple.argmax()),
    }
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=7)
    parser.add_argument(
        "--algorithm", choices=("dspark", "dflash2", "eagle4"), default="eagle4"
    )
    parser.add_argument("--result-dir", default="evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--memory-mib", type=int, default=6200)
    parser.add_argument("--reserve-mib", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    replay(args)


if __name__ == "__main__":
    main()
