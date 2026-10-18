#!/usr/bin/env python3
"""Bounded SGLang 0.5.19 target/current-best short smoke, NOT speed proof.

Run with the provisioned runtime interpreter and its existing activation script.
All GPU/server/JIT work lives in killable process groups. No downloads, HF writes,
algorithm substitutions, compiler environment overrides, or baseline cache reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

FAMILIES = ("dspark", "dflash", "dflash2")
PROMPTS = [
    ("chat", "A neighbor is nervous about their first community gardening day. Write four practical, friendly suggestions."),
    ("chat", "Explain to a curious child why a metal spoon and a wooden spoon feel different at the same room temperature."),
    ("code", "Write a Python function that merges two sorted integer lists without using sorted, and explain its time complexity."),
    ("code", "Write a Python function that counts consecutive runs in a string, with examples for an empty string and repeated letters."),
    ("math", "A library has 84 books. Three sevenths are borrowed, and 15 are returned. Explain how many books are now present."),
    ("math", "A rectangle has perimeter 54 and length 3 greater than twice its width. Find its dimensions and verify your result."),
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def remaining(deadline):
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError("explicit outer deadline reached")
    return seconds


def file_sha(path, deadline):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            remaining(deadline)
            h.update(chunk)
    return h.hexdigest()


def inventory(path, deadline):
    root = Path(path).resolve(strict=True)
    files = {}
    for p in sorted(root.rglob("*")):
        remaining(deadline)
        if p.is_file():
            files[p.relative_to(root).as_posix()] = file_sha(p, deadline)
    if not files or "config.json" not in files:
        raise ValueError(f"model directory lacks config.json: {root}")
    return files


def run_bounded(command, log, deadline, seconds, env=None):
    """Supervisor owns the whole worker/server/compiler group, even on failure."""
    allowance = min(seconds, remaining(deadline) - 3.0)
    if allowance <= 0:
        raise TimeoutError("insufficient deadline for bounded cleanup")
    started = time.monotonic()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        try:
            code = process.wait(timeout=allowance)
            if code:
                raise RuntimeError(f"worker exited {code}; see {Path(log).name}")
        finally:
            # Do not skip descendants merely because the group leader exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1)
    return time.monotonic() - started


def prepare(job):
    """CPU-only independent draft conversion; controller bests remain immutable."""
    from transformers import AutoTokenizer
    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.export.checkpoint_io import apply_legacy_rope_scaling
    from draftfit.export.to_sglang import _serving_state, _REQUIRED_DFLASH2_SERVING_KEYS

    out, deadline = Path(job["output_dir"]), job["deadline_unix"]
    state = read(job["controller_state"])
    if state.get("status") != "complete":
        raise ValueError("controller must finish before serving its immutable best exports")
    target = Path(job["target_model"]).resolve(strict=True)
    target_files = inventory(target, deadline)
    # Bind target bytes to the controller's source receipt, not only its name.
    for source in state["sources"].values():
        if Path(source["target"]).resolve() != target:
            raise ValueError("serving target differs from controller target")
        for name, expected in source["files"].items():
            p = Path(name).resolve()
            if p.is_relative_to(target) and target_files.get(p.relative_to(target).as_posix()) != expected:
                raise ValueError("target bytes changed since controller training")
    arms = {}
    for name, best in state["best"].items():
        strategy = best["strategy"]
        if strategy not in FAMILIES or strategy in arms:
            raise ValueError("expected exactly one controller best per registered family")
        source = Path(best["export"]).resolve(strict=True)
        source_files = inventory(source, deadline)
        if source_files != best["files"]:
            raise ValueError("controller best export file SHA mismatch")
        if file_sha(best["checkpoint"], deadline) != best["checkpoint_sha256"]:
            raise ValueError("controller best checkpoint SHA mismatch")
        config = read(source / "config.json")
        expected_arch = {"dspark": "DSparkDraftModel", "dflash": "DFlashDraftModel", "dflash2": "DFlash2DraftModel"}[strategy]
        if config.get("architectures") != [expected_arch]:
            raise ValueError("controller best architecture differs from its strategy")
        serving = source
        if strategy == "dflash2":
            model, info = AutoDraftModel.from_pretrained(source, local_files_only=True,
                trust_remote_code=False, device_map="cpu", torch_dtype="auto", output_loading_info=True)
            if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("incomplete DFlash2 HF source load")
            serving = out / "dflash2-sglang"
            serving_config = model.config_for_serving()
            serving_weights = _serving_state(model.state_dict_for_serving(), {},
                                             required_keys=_REQUIRED_DFLASH2_SERVING_KEYS)
            model.save_pretrained(serving, state_dict=serving_weights)
            write(serving / "config.json", serving_config)
            apply_legacy_rope_scaling(str(serving))
            del model, serving_weights
        elif strategy == "dspark":
            method, runtime = config.get("dflash_config", {}), config.get("dspark_config", {})
            if method.get("markov_rank", 0) != runtime.get("markov_rank", 0):
                raise ValueError("DSpark export lacks matching serving Markov metadata")
        serving_config = read(serving / "config.json")
        block = serving_config.get("block_size", serving_config.get("dflash_config", {}).get("block_size"))
        if type(block) is not int or block < 2:
            raise ValueError("draft block size must come from its real export config")
        arms[strategy] = dict(arm_name=name, source=str(source), source_files=source_files,
            source_checkpoint=best["checkpoint"], checkpoint_sha256=best["checkpoint_sha256"],
            serving=str(serving), serving_files=inventory(serving, deadline), block_size=block)
    if set(arms) != set(FAMILIES):
        raise ValueError("all three controller best families are required")
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)
    prompts = []
    for i, (domain, text) in enumerate([*PROMPTS, ("warmup", "Name two colors that can be found in a rainbow.")]):
        ids = tokenizer.apply_chat_template([dict(role="user", content=text)],
                                           tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or not ids or any(type(t) is not int for t in ids):
            raise ValueError("tokenizer must return concrete input token IDs")
        if len(ids) + job["max_new_tokens"] > job["context_length"]:
            raise ValueError("smoke workload exceeds server context")
        prompts.append(dict(id=f"{domain}-{i}", domain=domain, text=text, input_ids=ids,
                            input_sha256=digest(ids)))
    versions = {}
    for name in ("sglang", "flashinfer-python", "torch", "transformers", "draftfit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    audit = {}
    try:
        package = importlib.metadata.distribution("flashinfer-python")
        for relative in ("flashinfer/topk.py", "flashinfer/jit/core.py", "flashinfer/jit/cpp_ext.py", "flashinfer/jit/env.py"):
            path = Path(package.locate_file(relative))
            if path.is_file():
                lines = path.read_text().splitlines()
                audit[relative] = dict(sha256=file_sha(path, deadline),
                    compile_environment_source=[dict(line=i + 1, text=line) for i, line in enumerate(lines)
                        if any(s in line for s in ("getenv", "environ", "MAX_JOBS", "CUDA_ARCH", "run_ninja"))])
    except importlib.metadata.PackageNotFoundError:
        pass
    prepared = dict(target=str(target), target_files=target_files, arms=arms, prompts=prompts[:-1], warmup=prompts[-1],
        prompt_origin="independent handwritten smoke; not training/selection holdout",
        versions=versions, flashinfer_source_audit=audit,
        controller_state_sha256=file_sha(job["controller_state"], deadline))
    write(out / "prepared.json", prepared)
    return prepared


def request_json(url, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def healthy(url, timeout):
    # SGLang /health can return an empty/plain-text HTTP 200 body.
    with urllib.request.urlopen(url + "/health", timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("SGLang health status is not 200")


def output_ids(response):
    meta = response.get("meta_info", {})
    ids = meta.get("output_token_ids")
    if ids is None:
        rows = meta.get("output_token_logprobs")
        if not isinstance(rows, list) or any(not isinstance(r, list) or len(r) < 2 for r in rows):
            raise ValueError("missing output token IDs/logprobs")
        ids = [r[1] for r in rows]
    if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("invalid output token IDs")
    return ids


def sample(url, prompt, job, speculative, *, tokens=None):
    count = tokens or job["max_new_tokens"]
    sampling = dict(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=count,
                    sampling_seed=job["seed"], ignore_eos=True, skip_special_tokens=False)
    request = dict(input_ids=prompt["input_ids"], sampling_params=sampling,
                   return_logprob=True, logprob_start_len=-1, return_text_in_logprobs=False)
    start = time.monotonic()
    response = request_json(url + "/generate", request,
                            min(job["request_timeout_seconds"], remaining(job["deadline_unix"])))
    seconds = time.monotonic() - start
    ids = output_ids(response)
    meta = response["meta_info"]
    if len(ids) != count or meta.get("completion_tokens") != count:
        raise ValueError("fixed-length smoke did not produce the full requested token count")
    if speculative and (type(meta.get("spec_verify_ct")) is not int or meta["spec_verify_ct"] <= 0):
        raise ValueError("candidate did not execute speculative verification")
    if speculative:
        proposed, accepted = meta.get("spec_num_proposed_drafts"), meta.get("spec_num_correct_drafts")
        if type(proposed) is not int or type(accepted) is not int or not 0 <= accepted <= proposed or proposed <= 0:
            raise ValueError("missing/invalid actual speculative acceptance counters")
    return dict(id=prompt["id"], domain=prompt["domain"], input_sha256=prompt["input_sha256"],
        request_sha256=digest(request), output_ids=ids, output_sha256=digest(ids),
        seconds=seconds, tokens=count, tokens_per_second=count / seconds, meta_info=meta)


def serve(job, arm):
    out, deadline = Path(job["output_dir"]), job["deadline_unix"]
    prepared = read(out / "prepared.json")
    if importlib.metadata.version("sglang") != job["expected_sglang_version"]:
        raise ValueError("installed SGLang version differs from explicit expected version")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", prepared["target"],
        "--host", "127.0.0.1", "--port", str(port), "--tp-size", "1", "--dtype", "bfloat16",
        "--context-length", str(job["context_length"]), "--mem-fraction-static", "0.72",
        "--max-running-requests", "1", "--max-total-tokens", "4096", "--skip-server-warmup",
        "--disable-radix-cache", "--disable-cuda-graph"]
    if arm != "target":
        item = prepared["arms"][arm]
        algorithm = "DSPARK" if arm == "dspark" else "DFLASH"
        block_flag = "--speculative-dspark-block-size" if arm == "dspark" else "--speculative-dflash-block-size"
        command += ["--speculative-algorithm", algorithm, "--speculative-draft-model-path",
                    item["serving"], block_flag, str(item["block_size"])]
    write(out / f"{arm}-command.json", command)
    base = f"http://127.0.0.1:{port}"
    result = dict(status="running", arm=arm, samples=[], speed_claim=False,
                  timing_scope="short_smoke_after_separate_warmup_including_request_prefill")
    write(out / f"{arm}-result.json", result)
    with (out / f"{arm}-server.log").open("w") as log:
        # Inherit this worker's session; the outer supervisor kills every child.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            end = min(deadline, time.time() + job["startup_timeout_seconds"])
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"SGLang server exited {process.returncode}")
                remaining(end)
                try:
                    healthy(base, min(2, remaining(end)))
                    break
                except (OSError, ValueError):
                    time.sleep(min(.25, remaining(end)))
            # Distinct warmup request, never substituted for a measured request.
            result["warmup"] = sample(base, prepared["warmup"], job, arm != "target", tokens=8)
            for prompt in prepared["prompts"]:
                result["samples"].append(sample(base, prompt, job, arm != "target"))
                write(out / f"{arm}-result.json", result)
            result["status"] = "complete"
            write(out / f"{arm}-result.json", result)
        finally:
            process.terminate() if process.poll() is None else None
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def compare(baseline, candidate):
    bases = {r["id"]: r for r in baseline["samples"]}
    rows = candidate["samples"]
    if len(bases) != len(baseline["samples"]) or len(rows) != len(bases) or {r["id"] for r in rows} != set(bases):
        raise ValueError("duplicate, missing or unmatched request pairs")
    pairs = []
    for row in rows:
        base = bases[row["id"]]
        if row["request_sha256"] != base["request_sha256"] or row["input_sha256"] != base["input_sha256"]:
            raise ValueError("request pairing differs from target-only baseline")
        pairs.append(dict(id=row["id"], domain=row["domain"], request_sha256=row["request_sha256"],
            baseline_output_sha256=base["output_sha256"], candidate_output_sha256=row["output_sha256"],
            exact_token_parity=row["output_ids"] == base["output_ids"]))
    return pairs


def worker(job, stage):
    if stage == "prepare":
        prepare(job)
    elif stage == "prewarm":
        if importlib.metadata.version("sglang") != job["expected_sglang_version"]:
            raise ValueError("unexpected SGLang version before prewarm")
        from flashinfer.topk import get_topk_module
        start = time.monotonic()
        get_topk_module()  # Same native selector top-k JIT path in recovered traceback.
        write(Path(job["output_dir"]) / "topk-prewarm.json", dict(status="complete", seconds=time.monotonic() - start))
    elif stage == "verify-sources":
        p = read(Path(job["output_dir"]) / "prepared.json")
        if file_sha(job["controller_state"], job["deadline_unix"]) != p["controller_state_sha256"]:
            raise ValueError("controller state changed during gate")
        if inventory(p["target"], job["deadline_unix"]) != p["target_files"]:
            raise ValueError("target changed during gate")
        for arm in p["arms"].values():
            if file_sha(arm["source_checkpoint"], job["deadline_unix"]) != arm["checkpoint_sha256"]:
                raise ValueError("source checkpoint changed during gate")
            if inventory(arm["source"], job["deadline_unix"]) != arm["source_files"] or inventory(arm["serving"], job["deadline_unix"]) != arm["serving_files"]:
                raise ValueError("source/serving export changed during gate")
    else:
        serve(job, stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-state", type=Path)
    parser.add_argument("--target-model", default="/workspace/draftfit-runtime-20261017/target-model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--max-new-tokens", type=int, choices=(32, 64), default=64)
    parser.add_argument("--seed", type=int, default=481)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300)
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    parser.add_argument("--compile-timeout-seconds", type=float, default=300)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--expected-sglang-version", default="0.5.19")
    parser.add_argument("--prewarm-topk", action="store_true")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("--_job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        worker(read(args._job), args._worker)
        return 0
    if not args.controller_state or not args.output_dir or args.deadline_unix is None:
        parser.error("--controller-state, --output-dir and --deadline-unix are required")
    for name in ("deadline_unix", "startup_timeout_seconds", "request_timeout_seconds", "compile_timeout_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.context_length < 128:
        parser.error("context length must be >=128")
    job = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")}
    out = Path(job["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    write(out / "job.json", job)
    report = dict(schema="draftfit-controller-best-sglang-smoke-v1", status="running", all_passed=False,
        speed_claim=False, measurement="short_smoke_not_production_speed_proof", stages={},
        deadline_unix=args.deadline_unix, max_new_tokens=args.max_new_tokens, seed=args.seed,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    sequence = ["prepare", *(["prewarm"] if args.prewarm_topk else []), "target", *FAMILIES, "verify-sources"]
    try:
        for stage in sequence:
            report["active_stage"] = stage
            write(out / "receipt.json", report)
            timeout = args.compile_timeout_seconds if stage == "prewarm" else remaining(args.deadline_unix)
            try:
                seconds = run_bounded([sys.executable, str(Path(__file__).resolve()), "--_worker", stage,
                    "--_job", str(out / "job.json")], out / f"{stage}-worker.log", args.deadline_unix, timeout, env)
                report["stages"][stage] = dict(status="complete", seconds=seconds)
                if stage in FAMILIES:
                    baseline = read(out / "target-result.json")
                    candidate = read(out / f"{stage}-result.json")
                    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
                        raise ValueError("completed baseline and family results required")
                    pairs = compare(baseline, candidate)
                    report["stages"][stage]["pairs"] = pairs
                    if not all(p["exact_token_parity"] for p in pairs):
                        # A completed comparison failure is evidence, not a worker
                        # failure: still measure the other families and check sources.
                        report["stages"][stage].update(status="failed", error_type="ValueError",
                                                       error=f"{stage} token parity failed")
            except Exception as exc:
                report["stages"].setdefault(stage, {}).update(status="failed", error_type=type(exc).__name__, error=str(exc))
                raise
        failed = [stage for stage in FAMILIES if report["stages"][stage]["status"] == "failed"]
        report.update(status="failed" if failed else "complete", all_passed=not failed)
        if failed:
            report.update(error_type="ValueError", error="token parity failed: " + ", ".join(failed))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        for stage in sequence:
            report["stages"].setdefault(stage, dict(status="not_run"))
        report["finished_unix"] = time.time()
        write(out / "receipt.json", report)
    print(json.dumps({k: report[k] for k in ("status", "all_passed", "active_stage", "speed_claim")}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
