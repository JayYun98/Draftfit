"""Compare live MLX cache rollback with a committed-prefix replay."""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tap", type=int, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.005)
    args = ap.parse_args()
    sys.path.insert(0, str(args.runtime / "src"))
    import mlx.core as mx
    mx.set_cache_limit(128 * 1024 * 1024)
    from mlx_dspark.generate import encode_prompt
    from mlx_dspark.load import load_target

    target, tok = load_target(str(args.model), require_tap=True)
    if args.float32:
        target.model.set_dtype(mx.float32)
    ids = encode_prompt(tok, "Write a Python factorial function.")
    block, results = [1, 2, 3, 4], []
    for keep in (1, 2, 4):
        baseline, candidate = target.make_cache(), target.make_cache()
        mx.eval(target.plain(mx.array([ids]), baseline),
                target.plain(mx.array([ids]), candidate))
        ref = target.plain(mx.array([block[:keep]]), baseline)
        logits, hidden = target.verify(mx.array([block]), candidate, args.tap)
        mx.eval(ref, logits, hidden)
        target.rollback(candidate, len(block) - keep, block[1:keep])
        deltas, offsets_equal = [], True
        for a, b in zip(baseline, candidate):
            offsets_equal &= getattr(a, "offset", None) == getattr(b, "offset", None)
            if len(a.state) != len(b.state):
                raise ValueError("cache state component count differs")
            for x, y in zip(a.state, b.state):
                if x is None or y is None:
                    if x is not None or y is not None:
                        raise ValueError("cache state presence differs")
                    continue
                if x.shape != y.shape:
                    raise ValueError("cache state shape differs")
                deltas.append(float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item()))
        a = target.plain(mx.array([[88]]), baseline)
        b = target.plain(mx.array([[88]]), candidate)
        mx.eval(a, b)
        delta = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
        same_token = int(mx.argmax(a[0, -1]).item()) == int(mx.argmax(b[0, -1]).item())
        row = {"kept": keep, "rejected": len(block) - keep,
               "max_state_abs": max(deltas), "next_logits_max_abs": delta,
               "offsets_equal": offsets_equal, "next_token_equal": same_token,
               "pass": offsets_equal and same_token and max(deltas) <= args.tolerance
                       and delta <= args.tolerance}
        results.append(row)
        print(row, flush=True)
    with args.output.open("x") as f:
        json.dump({"model": str(args.model), "float32": args.float32,
                   "tolerance": args.tolerance, "cases": results}, f, indent=2)
    if not all(r["pass"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
