"""Generate with a published DSpark draft in the patched mlx-dspark runtime.

See runtime-setup.md in the model repository for pinned runtime and target setup.
"""

from __future__ import annotations

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="MLX target ID or prepared local directory")
    parser.add_argument("--draft", required=True, help="Draftfit DSpark ID or local directory")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-draft", type=int, default=3)
    parser.add_argument("--check", action="store_true", help="Also compare target-only token IDs")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_draft < 1:
        parser.error("token limits must be positive")

    import mlx.core as mx
    from mlx_dspark.generate import encode_messages, greedy_generate, speculative_generate
    from mlx_dspark.load import load_drafter, load_target

    mx.set_cache_limit(128 * 1024 * 1024)
    target, tokenizer = load_target(args.target, require_tap=True)
    # Preserve packed quantized weights while using the verified floating-state dtype.
    target.model.set_dtype(mx.float32)
    draft, _ = load_drafter(args.draft, bits=4)
    ids = encode_messages(tokenizer, [{"role": "user", "content": args.prompt}])
    baseline = None
    if args.check:
        baseline = greedy_generate(
            target, tokenizer, prompt_ids=ids,
            max_new_tokens=args.max_new_tokens, temperature=0,
        )
    result = speculative_generate(
        target, tokenizer, draft, prompt_ids=ids,
        max_new_tokens=args.max_new_tokens, temperature=0,
        max_draft_tokens=args.max_draft, lookup_drafts=False,
    )
    if len(result.token_ids) > args.max_new_tokens:
        raise RuntimeError("runtime exceeded the output token budget")
    if baseline is not None and baseline.token_ids != result.token_ids:
        raise RuntimeError("speculative output differs from target-only token IDs")
    print(result.text)
    if baseline is not None:
        print(f"Exact target-only token match: {len(result.token_ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
