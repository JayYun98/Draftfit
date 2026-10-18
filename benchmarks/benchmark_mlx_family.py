"""Compare local drafts on identical held-out token IDs using mlx-dspark."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--cache-mib", type=int, default=128)
    ap.add_argument(
        "--verified-results",
        type=Path,
        help="reuse prior exact-token checks only for identical data, model and runtime",
    )
    args = ap.parse_args()
    if min(args.per_domain, args.trials, args.max_tokens) < 1:
        ap.error("sample, trial and token counts must be positive")
    sys.path.insert(0, str(args.runtime / "src"))
    import mlx.core as mx

    if args.cache_mib < 0:
        ap.error("cache limit must not be negative")
    mx.set_cache_limit(args.cache_mib * 1024 * 1024)
    from mlx_dspark.generate import (
        dflash_generate,
        encode_messages,
        greedy_generate,
        speculative_generate,
    )
    from mlx_dspark.load import load_dflash, load_drafter, load_target

    cfg = json.loads(args.manifest.read_text())
    with args.output.open("x") as output:

        def emit(row):
            output.write(json.dumps(row) + "\n")
            output.flush()

        target, tok = load_target(cfg["target"], require_tap=True)
        if cfg.get("target_float32", False):
            target.model.set_dtype(mx.float32)
        selected, counts = [], {}
        for line in args.data.read_text().splitlines():
            row = json.loads(line)
            domain = row["id"].split(":", 1)[0]
            if counts.get(domain, 0) >= args.per_domain:
                continue
            messages = row.get("messages", row.get("conversations"))
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(
                    "held-out conversation must end in an assistant answer"
                )
            ids = encode_messages(tok, messages[:-1])
            selected.append((row["id"], domain, ids))
            counts[domain] = counts.get(domain, 0) + 1
        if not selected:
            raise ValueError("empty evaluation set")
        emit(
            {
                "kind": "provenance",
                "manifest": cfg,
                "device": mx.device_info(),
                "mlx": mx.__version__,
                "runtime_commit": subprocess.check_output(
                    ["git", "-C", str(args.runtime), "rev-parse", "HEAD"], text=True
                ).strip(),
                "runtime_diff_sha256": hashlib.sha256(
                    subprocess.check_output(["git", "-C", str(args.runtime), "diff"])
                ).hexdigest(),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "selected_ids": [r[0] for r in selected],
                "domain_counts": counts,
                "prompt_lengths": {r[0]: len(r[2]) for r in selected},
                "prompt_ids_sha256": hashlib.sha256(
                    json.dumps(selected).encode()
                ).hexdigest(),
                "trials": args.trials,
                "max_new_tokens": args.max_tokens,
                "mlx_cache_mib": args.cache_mib,
                "lookup_drafts": False,
                "draft_bits": cfg.get("draft_bits", 4),
            }
        )
        models = {}
        for arm in cfg["arms"]:
            if arm["mode"] != "baseline":
                loader = load_drafter if arm["mode"] == "dspark" else load_dflash
                models[arm["name"]], _ = loader(
                    arm["path"], bits=cfg.get("draft_bits", 4)
                )

        def generate(arm, ids, tokens, record=False):
            rounds = []
            kwargs = dict(prompt_ids=ids, max_new_tokens=tokens, temperature=0)
            if record:
                kwargs["on_round"] = lambda **row: rounds.append(row)
            if arm["mode"] == "baseline":
                result = greedy_generate(target, tok, **kwargs)
            else:
                fn = (
                    speculative_generate if arm["mode"] == "dspark" else dflash_generate
                )
                result = fn(
                    target,
                    tok,
                    models[arm["name"]],
                    max_draft_tokens=arm.get("cap", 3),
                    lookup_drafts=False,
                    **kwargs,
                )
            return result, rounds

        for arm in cfg["arms"]:
            generate(arm, selected[0][2], min(32, args.max_tokens))
        baseline = next(a for a in cfg["arms"] if a["mode"] == "baseline")
        expected, failed = {}, set()
        if args.verified_results:
            prior = [
                json.loads(x) for x in args.verified_results.read_text().splitlines()
            ]
            provenance = prior[0]
            diff = hashlib.sha256(
                subprocess.check_output(["git", "-C", str(args.runtime), "diff"])
            ).hexdigest()
            if (
                provenance["manifest"] != cfg
                or provenance["runtime_diff_sha256"] != diff
                or provenance["data_sha256"]
                != hashlib.sha256(args.data.read_bytes()).hexdigest()
                or provenance["selected_ids"] != [r[0] for r in selected]
                or provenance["max_new_tokens"] != args.max_tokens
            ):
                raise ValueError("previous correctness checks do not match this run")
            checks = [r for r in prior if r["kind"] == "correctness"]
            keys = {(r["id"], r["arm"]) for r in checks}
            if keys != {
                (r[0], a["name"]) for r in selected for a in cfg["arms"]
            } or not all(r["parity"] and r["output_bounded"] for r in checks):
                raise ValueError("previous correctness checks are incomplete or failed")
            expected = {
                r["id"]: r["result"]["token_ids"]
                for r in checks
                if r["arm"] == baseline["name"]
            }
            emit(
                {
                    "kind": "reused_correctness",
                    "source": str(args.verified_results),
                    "sha256": hashlib.sha256(
                        args.verified_results.read_bytes()
                    ).hexdigest(),
                    "checks": len(checks),
                }
            )
        for case_id, domain, ids in [] if expected else selected:
            result, _ = generate(baseline, ids, args.max_tokens)
            expected[case_id] = result.token_ids
            for arm in cfg["arms"]:
                result, rounds = generate(arm, ids, args.max_tokens, record=True)
                parity = result.token_ids == expected[case_id]
                bounded = len(result.token_ids) <= args.max_tokens
                if not parity or not bounded:
                    failed.add(arm["name"])
                emit(
                    {
                        "kind": "correctness",
                        "arm": arm["name"],
                        "id": case_id,
                        "domain": domain,
                        "parity": parity,
                        "output_bounded": bounded,
                        "result": dataclasses.asdict(result),
                        "rounds": rounds,
                    }
                )
            print("parity", domain, case_id, "failed arms", sorted(failed), flush=True)
        eligible = [a for a in cfg["arms"] if a["name"] not in failed]
        records = []
        for trial in range(args.trials):
            for index, (case_id, domain, ids) in enumerate(selected):
                order = (
                    eligible if (trial + index) % 2 == 0 else list(reversed(eligible))
                )
                for arm in order:
                    result, rounds = generate(arm, ids, args.max_tokens, record=True)
                    row = {
                        "kind": "measurement",
                        "arm": arm["name"],
                        "id": case_id,
                        "domain": domain,
                        "trial": trial,
                        "parity": result.token_ids == expected[case_id],
                        "tok_s": result.tokens_per_sec,
                        "decode_tok_s": result.decode_tokens_per_sec,
                        "mean_commit_length": sum(x["committed"] for x in rounds)
                        / max(len(rounds), 1),
                        "process_peak_memory_bytes": mx.get_peak_memory(),
                        "active_memory_bytes": mx.get_active_memory(),
                        "cache_memory_bytes": mx.get_cache_memory(),
                        "prompt_tokens": len(ids),
                        "result": dataclasses.asdict(result),
                        "rounds": rounds,
                    }
                    records.append(row)
                    emit(row)
            print("measurement trial", trial + 1, "complete", flush=True)
        summaries = []
        for arm in eligible:
            for domain in counts:
                rows = [
                    r
                    for r in records
                    if r["arm"] == arm["name"] and r["domain"] == domain
                ]
                summaries.append(
                    {
                        "arm": arm["name"],
                        "domain": domain,
                        "requests": len(rows),
                        "parity": all(r["parity"] for r in rows),
                        "median_tok_s": statistics.median(r["tok_s"] for r in rows),
                        "median_decode_tok_s": statistics.median(
                            r["decode_tok_s"] for r in rows
                        ),
                        "median_commit_length": statistics.median(
                            r["mean_commit_length"] for r in rows
                        ),
                    }
                )
        emit({"kind": "summary", "failed_arms": sorted(failed), "domains": summaries})
        print(
            json.dumps({"failed_arms": sorted(failed), "domains": summaries}),
            flush=True,
        )
        if failed or any(not row["parity"] for row in records):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
