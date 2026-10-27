"""Trusted SGLang 0.5.19 producer for serving_dispatch's strict JSON contract.

CLI: python -m draftfit.training.sglang_producer --job-dir ATTEMPT
Inspect before preparing independent expectations:
  ... --inspect-runtime --output runtime.json --deadline-unix UNIX

request.json uses serving_dispatch's contract, including source:target and
source:validation. Validation is fixed JSONL of {row_index, domain, input_ids};
IDs are already rendered/tokenized prompts, not training answers. Target weights
and tokenizer bindings both fingerprint the entire target file inventory (a
conservative binding). Runtime build hashes installed SGLang/FlashInfer Python
and native files plus dependency versions. No report claim supplies expectations.

Only sequential TP1 CUDA/bfloat16 greedy serving, disabled radix/cache graphs,
32/64 fixed output tokens, and native DSPARK/DFLASH architectures are supported.
DFlash2 must already be converted with its own config/state_for_serving methods.
Independent guardians retain the job lock and clean stage process groups on
deadline, signal, or abrupt producer death. Children must not daemonize or leave
their stage process group. Dispatcher restore requires cleanup confirmation.
A subprocess loads its own target; no resident-training-target claim is made.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from draftfit.training.joint_policy import fingerprint
from draftfit.training.serving_dispatch import (
    _CLEANUP_FD_ENV,
    _CLEANUP_SECONDS,
    _LOCK_FD_ENV,
    _check_sources,
    _kill_group,
    _native,
    _read,
    _remaining,
    _sha,
    _source,
    _write,
)
from draftfit.training.serving_report import (
    ACCEPTANCE_SCOPE,
    SCHEMA,
    TIMING_SCOPE,
    normalize_serving_report,
)

DEFAULT_CONFIG = dict(
    temperature=0,
    max_new_tokens=64,
    trials=1,
    concurrency=1,
    warmup_requests=1,
    cache_policy="disabled",
    seed=481,
    context_length=512,
    startup_timeout_seconds=300,
    request_timeout_seconds=120,
)
_ARCH = {
    "dspark": "DSparkDraftModel",
    "dflash": "DFlashDraftModel",
    "dflash2": "DFlash2DraftModel",
}


def runtime_identity(deadline):
    """Inspect actual installed code and one visible CUDA device, never a report."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("SGLang producer requires exactly one visible CUDA device")
    versions, files = {}, {}
    for name in (
        "sglang",
        "flashinfer-python",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
        "torch",
        "transformers",
        "sglang-kernel",
    ):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = dist.version
        if name in ("sglang", "flashinfer-python"):
            for relative in dist.files or []:
                path = Path(dist.locate_file(relative))
                if path.is_file() and path.suffix in (
                    ".py",
                    ".so",
                    ".h",
                    ".cuh",
                    ".cu",
                ):
                    files[name + "/" + str(relative)] = _sha(path, deadline)
    if versions["sglang"] != "0.5.19" or not files:
        raise ValueError("actual installed SGLang 0.5.19 source inventory required")
    props = torch.cuda.get_device_properties(0)
    return dict(
        engine="sglang",
        version=versions["sglang"],
        build_sha256=fingerprint(dict(files=files, versions=versions)),
        device=props.name,
        dtype="bfloat16",
        device_uuid=str(getattr(props, "uuid", "unavailable")),
        cuda_version=torch.version.cuda,
        compute_capability=[props.major, props.minor],
        versions=versions,
        tp_size=1,
        cuda_graph=False,
        radix_cache=False,
    )


def load_holdout(path, config):
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index, domain, ids = row["row_index"], row["domain"], row["input_ids"]
        if (
            type(index) is not int
            or index < 0
            or index in seen
            or not isinstance(domain, str)
            or not domain
            or not isinstance(ids, list)
            or not ids
            or any(type(t) is not int or t < 0 for t in ids)
            or len(ids) + config["max_new_tokens"] > config["context_length"]
        ):
            raise ValueError("invalid fixed holdout row/index/domain/token IDs/context")
        seen.add(index)
        rows.append(
            dict(
                row_index=index,
                domain=domain,
                input_ids=ids,
                prompt_sha256=fingerprint(ids),
            )
        )
    if not rows:
        raise ValueError("fixed holdout is empty")
    return rows


def validate_config(config):
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError(
            "producer config must contain exactly documented fixed settings"
        )
    if (
        config["temperature"] != 0
        or config["max_new_tokens"] not in (32, 64)
        or config["concurrency"] != 1
        or config["cache_policy"] != "disabled"
        or type(config["seed"]) is not int
        or type(config["trials"]) is not int
        or config["trials"] < 1
        or type(config["warmup_requests"]) is not int
        or config["warmup_requests"] < 1
        or type(config["context_length"]) is not int
        or config["context_length"] < 128
    ):
        raise ValueError("unsupported SGLang smoke settings")
    for key in ("startup_timeout_seconds", "request_timeout_seconds"):
        if (
            type(config[key]) not in (int, float)
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError("positive bounded server/request timeout required")


def prepare_exports(exported, directory, deadline):
    """Materialize only native DFlash2 serving layout; caller preserves training RNG."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "exports.json"
    source_files = {
        name: _source(row["export"], deadline) for name, row in exported["arms"].items()
    }
    binding = fingerprint(
        dict(checkpoint=exported["checkpoint_sha256"], sources=source_files)
    )
    if receipt_path.exists():
        saved = _read(receipt_path)
        if saved["source_binding"] != binding:
            raise ValueError("serving export source changed")
        for value in saved["exports"].values():
            if _source(value["path"], deadline) != value:
                raise ValueError("serving export changed")
        return saved["exports"]
    outputs = {}
    for name, row in exported["arms"].items():
        if row["strategy"] != "dflash2":
            outputs[name] = source_files[name]
            continue
        from draftfit.export.checkpoint_io import apply_legacy_rope_scaling
        from draftfit.export.to_sglang import (
            _REQUIRED_DFLASH2_SERVING_KEYS,
            _serving_state,
        )
        from draftfit.modeling.auto import AutoDraftModel

        model, info = AutoDraftModel.from_pretrained(
            row["export"],
            device_map="cpu",
            torch_dtype="auto",
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        if any(
            info.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError("incomplete DFlash2 source load")
        destination = root / fingerprint(name)[:16]
        # A prior crash may leave this private uncommitted conversion, never the HF export.
        if destination.exists():
            import shutil

            shutil.rmtree(destination)
        weights = _serving_state(
            model.state_dict_for_serving(),
            {},
            required_keys=_REQUIRED_DFLASH2_SERVING_KEYS,
        )
        cfg = model.config_for_serving()
        model.save_pretrained(destination, state_dict=weights)
        _write(destination / "config.json", cfg)
        apply_legacy_rope_scaling(str(destination))
        outputs[name] = _source(destination, deadline)
        del model, weights
    _write(receipt_path, dict(source_binding=binding, exports=outputs))
    return outputs


def build_expectations(
    *,
    artifacts,
    target_model,
    target_revision,
    validation_path,
    runtime,
    config=None,
    min_requests=32,
    deadline_unix,
):
    """Caller-side builder from actual sources and a prior independent runtime inspection.

    Artifacts already bind the authoritative checkpoint and actual serving exports.
    No worker measurements are used. Use with dispatch or a standalone request.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    validate_config(cfg)
    target = _source(target_model, deadline_unix)
    if "config.json" not in target.get("files", {}):
        raise ValueError("local target required")
    rows = load_holdout(validation_path, cfg)
    bindings = dict(
        target=dict(
            model=str(Path(target_model).resolve()),
            revision=target_revision,
            weights_sha256=fingerprint(target["files"]),
            tokenizer_sha256=fingerprint(target["files"]),
        ),
        runtime=deepcopy(runtime),
        validation_data_sha256=_sha(validation_path, deadline_unix),
        config=cfg,
        prompts=[
            {k: r[k] for k in ("row_index", "domain", "prompt_sha256")} for r in rows
        ],
    )
    return dict(
        artifacts=deepcopy(artifacts),
        bindings=bindings,
        domains=sorted({r["domain"] for r in rows}),
        min_requests=min_requests,
    )


def preflight(request, deadline, *, inspect=runtime_identity):
    expected, sources = request["expected"], request["sources"]
    bindings = expected["bindings"]
    config = bindings["config"]
    validate_config(config)
    if expected.get("allow_speed_decisions", False):
        raise ValueError("smoke cannot enable speed decisions")
    _check_sources(request, deadline)
    if fingerprint(inspect(deadline)) != fingerprint(bindings["runtime"]):
        raise ValueError(
            "actual installed runtime/device differs from independent expectations"
        )
    target = sources["source:target"]
    validation = sources["source:validation"]
    if (
        str(Path(bindings["target"]["model"]).resolve()) != target["path"]
        or bindings["target"]["weights_sha256"] != fingerprint(target["files"])
        or bindings["target"]["tokenizer_sha256"] != fingerprint(target["files"])
        or bindings["validation_data_sha256"] != validation["sha256"]
    ):
        raise ValueError("actual target/tokenizer/holdout bytes do not match bindings")
    rows = load_holdout(validation["path"], config)
    prompts = [
        {k: r[k] for k in ("row_index", "domain", "prompt_sha256")} for r in rows
    ]
    if prompts != bindings["prompts"] or set(expected["domains"]) != {
        r["domain"] for r in rows
    }:
        raise ValueError("fixed holdout prompt/domain binding mismatch")
    if len(rows) * config["trials"] < expected.get("min_requests", 32):
        raise ValueError("insufficient fixed holdout requests")
    checkpoint_sha = sources["checkpoint"]["sha256"]
    _native(request["native_report"], expected, checkpoint_sha)
    arms = []
    for name, artifact in sorted(expected["artifacts"].items()):
        source = sources["export:" + name]
        if (
            artifact["artifact_sha256"] != checkpoint_sha
            or artifact["checkpoint_sha256"] != checkpoint_sha
            or artifact["export_sha256"] != fingerprint(source["files"])
        ):
            raise ValueError("actual checkpoint/export differs from artifact")
        arms.append(_arm(name, artifact, source))
    if expected.get("existing_draft") is not None:
        artifact = expected["existing_draft"]
        source = sources["source:existing_draft"]
        if artifact["export_sha256"] != fingerprint(source["files"]):
            raise ValueError("existing draft export binding mismatch")
        arms.append(_arm("existing-draft", artifact, source, existing=True))
    return dict(target=target["path"], config=config, prompts=rows, arms=arms)


def _arm(name, artifact, source, existing=False):
    cfg = _read(Path(source["path"]) / "config.json")
    family = artifact["family"]
    if family not in _ARCH or cfg.get("architectures") != [_ARCH[family]]:
        raise ValueError("unsupported or substituted draft architecture")
    if family == "dflash2" and cfg.get("model_type") != "qwen3":
        raise ValueError("DFlash2 needs its native serving conversion before dispatch")
    if family == "dspark" and cfg.get("dflash_config", {}).get(
        "markov_rank", 0
    ) != cfg.get("dspark_config", {}).get("markov_rank", 0):
        raise ValueError("DSpark serving metadata mismatch")
    block = cfg.get("block_size", cfg.get("dflash_config", {}).get("block_size"))
    if type(block) is not int or block < 2:
        raise ValueError("export must declare real block size")
    return dict(
        name=name,
        family=family,
        path=source["path"],
        block_size=block,
        artifact=artifact,
        existing=existing,
    )


def _http(url, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return (
            json.loads(body) if payload is not None else None
        )  # health may be plain text


def measure(url, prompt, trial, config, deadline, *, speculative):
    count = config["max_new_tokens"]
    payload = dict(
        input_ids=prompt["input_ids"],
        sampling_params=dict(
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_new_tokens=count,
            sampling_seed=config["seed"],
            ignore_eos=True,
            skip_special_tokens=False,
        ),
        return_logprob=True,
        logprob_start_len=-1,
        return_text_in_logprobs=False,
    )
    start = time.perf_counter()
    response = _http(
        url + "/generate",
        payload,
        min(config["request_timeout_seconds"], _remaining(deadline)),
    )
    seconds = time.perf_counter() - start
    meta = response["meta_info"]
    ids = meta.get("output_token_ids")
    if ids is None:
        rows = meta.get("output_token_logprobs")
        if not isinstance(rows, list) or any(
            not isinstance(r, list) or len(r) < 2 for r in rows
        ):
            raise ValueError("actual output token IDs required")
        ids = [r[1] for r in rows]
    if (
        not isinstance(ids, list)
        or len(ids) != count
        or any(type(t) is not int or t < 0 for t in ids)
        or meta.get("completion_tokens") != count
    ):
        raise ValueError("fixed output token count/IDs mismatch")
    accepted = proposed = 0
    if speculative:
        accepted, proposed = meta.get("spec_num_correct_drafts"), meta.get(
            "spec_num_proposed_drafts"
        )
        verifies = meta.get("spec_verify_ct")
        if (
            type(accepted) is not int
            or type(proposed) is not int
            or type(verifies) is not int
            or verifies <= 0
            or proposed <= 0
            or not 0 <= accepted <= min(proposed, count)
        ):
            raise ValueError(
                "actual speculative verification/acceptance counters required"
            )
    return dict(
        row_index=prompt["row_index"],
        trial=trial,
        domain=prompt["domain"],
        prompt_sha256=prompt["prompt_sha256"],
        token_ids=ids,
        generated_tokens=count,
        seconds=seconds,
        accepted_tokens=accepted,
        proposed_tokens=proposed,
        request_sha256=fingerprint(payload),
        raw_meta=meta,
    )


def serve(directory, prepared, arm, deadline):
    config = prepared["config"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        prepared["target"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--context-length",
        str(config["context_length"]),
        "--mem-fraction-static",
        "0.72",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "4096",
        "--skip-server-warmup",
        "--disable-radix-cache",
        "--disable-cuda-graph",
    ]
    if arm:
        algorithm = "DSPARK" if arm["family"] == "dspark" else "DFLASH"
        flag = (
            "--speculative-dspark-block-size"
            if algorithm == "DSPARK"
            else "--speculative-dflash-block-size"
        )
        command += [
            "--speculative-algorithm",
            algorithm,
            "--speculative-draft-model-path",
            arm["path"],
            flag,
            str(arm["block_size"]),
        ]
    name = arm["name"] if arm else "target-only"
    # File names use a hash, never an untrusted arm name as a path.
    stem = fingerprint(name)[:16]
    _write(directory / (stem + "-command.json"), command)
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    with (directory / (stem + "-server.log")).open("w") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            pass_fds=_inherited_fds(_LOCK_FD_ENV),
        )
        try:
            url = f"http://127.0.0.1:{port}"
            startup = min(deadline, time.time() + config["startup_timeout_seconds"])
            while True:
                if process.poll() is not None:
                    raise RuntimeError("SGLang server exited before health")
                _remaining(startup)
                try:
                    _http(url + "/health", None, min(2, _remaining(startup)))
                    break
                except OSError:
                    time.sleep(min(0.2, _remaining(startup)))
            for _ in range(config["warmup_requests"]):
                measure(
                    url,
                    prepared["prompts"][0],
                    0,
                    config,
                    deadline,
                    speculative=arm is not None,
                )
            rows = []
            for trial in range(config["trials"]):
                for prompt in prepared["prompts"]:
                    rows.append(
                        measure(
                            url,
                            prompt,
                            trial,
                            config,
                            deadline,
                            speculative=arm is not None,
                        )
                    )
                    _write(directory / (stem + "-partial.json"), rows)
            return dict(
                name=name,
                status="complete",
                runs=rows,
                **({"artifact": arm["artifact"]} if arm else {}),
            )
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def assemble(request, results):
    expected = request["expected"]
    comparison = fingerprint(expected["bindings"])
    for arm in results:
        arm["comparison_fingerprint"] = comparison
    by_name = {r["name"]: r for r in results}
    if len(by_name) != len(results):
        raise ValueError("duplicate arm results")
    report = dict(
        schema=SCHEMA,
        status="complete",
        split="validation",
        production_engine_speed=True,
        speed_decisions_allowed=False,
        timing_scope=TIMING_SCOPE,
        acceptance_scope=ACCEPTANCE_SCOPE,
        bindings=deepcopy(expected["bindings"]),
        comparison_fingerprint=comparison,
        baseline=by_name.pop("target-only"),
        candidates=[by_name.pop(n) for n in expected["artifacts"]],
        existing_draft=by_name.pop("existing-draft", None),
        target_resident=False,
        measurement="short_smoke_not_production_speed_proof",
    )
    if by_name:
        raise ValueError("unexpected producer arm")
    normalize_serving_report(report, **{**expected, "allow_speed_decisions": False})
    return report


def _stage(args):
    if args._stage == "inspect":
        _write(Path(args.output), runtime_identity(args.deadline_unix))
        return
    directory = Path(args.job_dir)
    request = _read(directory / "request.json")
    if args._stage == "preflight":
        _write(
            directory / "producer-prepared.json", preflight(request, args.deadline_unix)
        )
    elif args._stage == "finish":
        _check_sources(request, args.deadline_unix)
        if (
            runtime_identity(args.deadline_unix)
            != request["expected"]["bindings"]["runtime"]
        ):
            raise ValueError("runtime changed during serving")
        prepared = _read(directory / "producer-prepared.json")
        names = ["target-only"] + [a["name"] for a in prepared["arms"]]
        report = assemble(
            request,
            [_read(directory / (fingerprint(n)[:16] + "-result.json")) for n in names],
        )
        _write(directory / "report.json", report)
    else:
        prepared = _read(directory / "producer-prepared.json")
        arm = (
            None
            if args.arm == "target-only"
            else next(a for a in prepared["arms"] if a["name"] == args.arm)
        )
        result = serve(directory, prepared, arm, args.deadline_unix)
        _write(directory / (fingerprint(args.arm)[:16] + "-result.json"), result)


def _inherited_fds(*names):
    fds = tuple(int(os.environ[name]) for name in names if name in os.environ)
    for fd in fds:
        os.fstat(fd)
    return fds


def _terminate_signal(signum, frame):
    # _supervise closes the owner pipe; its independent guardian does cleanup.
    raise SystemExit(128 + signum)


def _guard_stage(command, owner_fd, deadline):
    """Outlive an abruptly killed producer while holding its job lock.

    EOF on the private owner pipe also works after reparenting, without Linux
    parent-death APIs or PID polling. Only this guardian owns the read endpoint.
    """
    completion = _inherited_fds(_CLEANUP_FD_ENV)

    def notify(token):
        if completion:
            try:
                os.write(completion[0], token)
            except BrokenPipeError:
                pass  # Dispatcher died; still finish cleanup and retain the lock.

    process = None
    notify(b"+")
    try:
        # Never give a serving child the cleanup acknowledgement descriptor.
        env = dict(os.environ)
        env.pop(_CLEANUP_FD_ENV, None)
        _remaining(deadline)
        if select.select([owner_fd], [], [], 0)[0]:
            raise RuntimeError("producer exited before stage launch")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=_inherited_fds(_LOCK_FD_ENV),
            env=env,
        )
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError("producer stage deadline reached")
            if select.select([owner_fd], [], [], min(0.05, left))[0]:
                raise RuntimeError("producer exited before stage cleanup")
            code = process.poll()
            if code is not None:
                if code:
                    raise RuntimeError("producer stage failed")
                return 0
    finally:
        if process is not None:
            _kill_group(process)
        notify(b"-")
        os.close(owner_fd)


def _supervise(arguments, deadline, log):
    allowance = _remaining(deadline) - 3
    if allowance <= 0:
        raise TimeoutError("producer deadline cleanup reserve exhausted")
    owner_read, owner_write = os.pipe()
    guardian = None
    try:
        command = [sys.executable, "-m", __spec__.name, *arguments]
        with Path(log).open("w") as stream:
            guardian = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "draftfit.training.sglang_producer",
                    "--_supervisor-command",
                    json.dumps(command),
                    "--_owner-fd",
                    str(owner_read),
                    "--deadline-unix",
                    str(time.time() + allowance),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(owner_read, *_inherited_fds(_LOCK_FD_ENV, _CLEANUP_FD_ENV)),
            )
        os.close(owner_read)
        owner_read = None
        if guardian.wait(timeout=allowance + _CLEANUP_SECONDS):
            raise RuntimeError("producer stage failed; see " + str(log))
    finally:
        # On any producer exception/signal, request cleanup without killing the
        # guardian that owns the lock and the cleanup acknowledgement.
        os.close(owner_write)
        if owner_read is not None:
            os.close(owner_read)
        if guardian is not None:
            guardian.wait(timeout=_CLEANUP_SECONDS + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--inspect-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--_stage", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--_supervisor-command", help=argparse.SUPPRESS)
    parser.add_argument("--_owner-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._supervisor_command:
        return _guard_stage(
            json.loads(args._supervisor_command), args._owner_fd, args.deadline_unix
        )
    if args._stage:
        _stage(args)
        return 0
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _terminate_signal)
    if args.inspect_runtime:
        if args.output is None or args.deadline_unix is None:
            parser.error("inspect requires output and absolute deadline")
        _supervise(
            [
                "--_stage",
                "inspect",
                "--output",
                str(args.output),
                "--deadline-unix",
                str(args.deadline_unix),
            ],
            args.deadline_unix,
            args.output.with_suffix(".log"),
        )
        return 0
    if args.job_dir is None:
        parser.error("--job-dir required")
    root = args.job_dir.resolve()
    admission = _read(root / "admission.json")
    if admission.get("approved") is not True:
        raise ValueError("external admission required")
    for key in ("deadline_unix", "max_seconds"):
        if (
            type(admission.get(key)) not in (int, float)
            or not math.isfinite(admission[key])
            or admission[key] <= 0
        ):
            raise ValueError("invalid admission deadline/duration")
    deadline = min(admission["deadline_unix"], time.time() + admission["max_seconds"])
    common = ["--job-dir", str(root), "--deadline-unix", str(deadline)]
    receipt = dict(
        status="running", target_resident=False, speed_decisions_allowed=False
    )
    _write(root / "producer-receipt.json", receipt)
    try:
        _supervise(
            [*common, "--_stage", "preflight"],
            deadline,
            root / "producer-preflight.log",
        )
        prepared = _read(root / "producer-prepared.json")
        for name in ["target-only"] + [a["name"] for a in prepared["arms"]]:
            _supervise(
                [*common, "--_stage", "arm", "--arm", name],
                deadline,
                root / (fingerprint(name)[:16] + "-worker.log"),
            )
        _supervise(
            [*common, "--_stage", "finish"], deadline, root / "producer-finish.log"
        )
        receipt.update(
            status="complete", report_sha256=fingerprint(_read(root / "report.json"))
        )
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        _write(root / "producer-receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
