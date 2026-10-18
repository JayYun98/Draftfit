"""Persistent actual controller CLI gate: 3 families, forced plateau, new phase.

Run from the checkout with a fresh --output-dir and --device cpu|cuda. Optional
--target-model/--draft-config/--target-revision must all refer to an existing local
pretrained target and compatible draft config; downloads are disabled. Otherwise
use tiny random Qwen weights. All tokens and admission amounts are synthetic.
This is a lifecycle gate, not a provider budget guard or a production speed test.

The supervisor bounds the ENTIRE gate (setup, CLI runs, hashing and reload) to
--timeout-seconds, at most 180 seconds. An isolated process group contains the
gate and its CLI children; timeout kills that group. Receipts/logs/checkpoints
remain on disk. No live publication, GPU provisioning, or remote API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = ("dspark", "dflash", "dflash2")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def refresh_budget(root):
    write(
        root / "budget.json",
        dict(
            timestamp_unix=time.time(),
            spent_usd=0.0,
            reserved_usd=0.0,
            admission_kind="synthetic_local_test_only",
            provider_budget=False,
            provider_guard=False,
        ),
    )


def run_cli(root, label, deadline_unix, *, resume=False):
    refresh_budget(root)
    command = [
        sys.executable,
        "-m",
        "draftfit.cli",
        "unattended",
        "--config",
        str(root / "controller.json"),
        *(["--resume"] if resume else []),
    ]
    started = time.monotonic()
    with (root / f"{label}.log").open("w") as log:
        # Inherit the supervisor's process group, offline flags and credentials.
        process = subprocess.Popen(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT
        )
        next_refresh = time.monotonic() + 5
        while process.poll() is None:
            if time.time() >= deadline_unix:
                raise TimeoutError("gate controller CLI deadline reached")
            if time.monotonic() >= next_refresh:
                refresh_budget(root)
                next_refresh = time.monotonic() + 5
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
    require(process.returncode == 0, f"{label} CLI failed; see {label}.log")
    return dict(
        command=command,
        exit_code=process.returncode,
        seconds=time.monotonic() - started,
        log=str(root / f"{label}.log"),
    )


def checkpoint_receipt(path, state):
    optimizer = state["optimizer"]
    moments = optimizer["optimizer_state_dict"]["state"]
    require(
        state["step"] == 2 and state["consumed_samples"] == 4,
        "phase cursor/update mismatch",
    )
    require(
        state["epoch"] == 0 and state["next_batch"] == 4, "phase data cursor mismatch"
    )
    require(
        bool(moments) and bool(optimizer["fp32_params"]),
        "missing optimizer moments/masters",
    )
    require(
        bool(optimizer["scheduler_state_dict"]) and bool(state["rng"]),
        "missing scheduler/RNG",
    )
    require(
        all(float(row["step"]) == 2 for row in moments.values()),
        "optimizer step mismatch",
    )
    return dict(
        path=str(path),
        sha256=sha(path),
        schema=state["schema"],
        step=state["step"],
        epoch=state["epoch"],
        next_batch=state["next_batch"],
        consumed_samples=state["consumed_samples"],
        horizon=state["horizon"],
        optimizer_moment_entries=len(moments),
        fp32_master_count=len(optimizer["fp32_params"]),
        scheduler_state=optimizer["scheduler_state_dict"],
        rng_keys=sorted(state["rng"]),
    )


def exercise(args, root):
    source_paths = [
        Path(__file__),
        REPO / "draftfit/unattended.py",
        REPO / "draftfit/training/joint_family.py",
    ]
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in source_paths}
    import torch

    from draftfit.modeling.auto import AutoDraftModel
    from tests.test_joint_online_cpu import online_fixture
    from tests.test_joint_policy import policy

    started = time.monotonic()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU substitution forbidden")
    torch.set_num_threads(1)
    online_fixture(root, seeded=False)
    target = Path(args.target_model).resolve() if args.target_model else root / "target"
    weights = sorted(target.glob("*.safetensors"))
    require(bool(weights), "local safetensors target weights required")
    target_hashes = {p.name: sha(p) for p in [target / "config.json", *weights]}
    base = read(root / "base.json")
    if args.target_model:
        draft = read(args.draft_config)
        write(root / "draft.json", draft)
        base["model"].update(
            target_model_path=str(target),
            target_revision=args.target_revision,
            mask_token_id=draft["mask_token_id"],
        )
    # Enough unique rows for requested horizon 8; the deliberate plateau stops at 2.
    train = [
        dict(
            input_ids=list(range(1 + i, 9 + i)),
            loss_mask=[0, 1, 1, 1, 1, 1, 1, 0],
            domain="chat",
        )
        for i in range(12)
    ]
    holdout = [
        dict(
            input_ids=list(range(40 + i * 8, 48 + i * 8)),
            loss_mask=[0, 1, 1, 1, 1, 1, 1, 0],
            domain=domain,
        )
        for i, domain in enumerate(("chat", "code"))
    ]
    for name, rows in (("train", train), ("eval", holdout)):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    base["training"].update(
        num_epochs=4,
        max_steps=8,
        total_steps=8,
        batch_size=1,
        accumulation_steps=2,
        learning_rate=1e-4,
        eval_interval=1,
        save_interval=1,
        log_interval=1,
        max_checkpoints=10,
        joint_feature_queue_bytes=268435456 if args.target_model else 32768,
        joint_feature_queue_batches=4,
        joint_verification=dict(
            interval=1,
            max_prompts=2,
            max_new_tokens=4,
            draft_tokens=2,
            trials=1,
            offload_optimizer=True,
        ),
    )
    write(root / "base.json", base)
    manifest = read(root / "family.json")
    manifest["runs"] = [dict(name=name, strategy=name) for name in ARMS]
    write(root / "family.json", manifest)
    control_policy = policy(
        domains=["chat", "code"],
        patience=1,
        min_delta=1e6,
        min_requests=2,
        next_steps=1,
    )
    control_policy["limits"].update(
        max_usd=10.0,
        reserve_usd=1.0,
        max_steps=10,
        max_pilots=1,
        max_pilot_steps=2,
        max_pilot_usd=0.2,
        max_exploration_usd=0.2,
    )
    write(
        root / "controller.json",
        dict(
            schema="draftfit-unattended-v1",
            manifest="family.json",
            output_dir="controller",
            max_steps=10,
            deadline_unix=args._deadline_unix - 5,
            budget_path="budget.json",
            selection="loss",
            policy=control_policy,
            presets=[
                dict(
                    id="lower-lr",
                    when=["loss_plateau"],
                    reason="synthetic forced-plateau lifecycle gate",
                    overrides=["training.learning_rate=0.00001"],
                    pilot_steps=2,
                    pilot_cost_usd=0.1,
                )
            ],
        ),
    )
    cli = [run_cli(root, "initial", args._deadline_unix - 3)]
    campaign = root / "controller"
    state = read(campaign / "state.json")
    require(
        state["status"] == "complete" and state["total_steps"] == 4,
        "expected 4 completed updates",
    )
    require(
        [p["preset"] for p in state["phases"]] == ["initial", "lower-lr"],
        "expected registered second phase",
    )
    require(set(state["best"]) == set(ARMS), "missing family best exports")
    first, second = state["phases"]
    require(
        first["steps"] == 8 and first["branch"]["reason"] == "loss_plateau",
        "forced plateau did not branch",
    )
    require(
        first["branch"]["optimizer_mode"] == "weights_only", "branch migrated optimizer"
    )
    require(
        first["branch"]["new_phase_required"] is True,
        "branch did not request new phase",
    )
    require(
        second["offset"] == 2 and second["steps"] == 2, "incorrect second phase range"
    )
    checkpoints, queues, memory_peaks = {}, {}, []
    for phase in state["phases"]:
        require(len(phase["attempts"]) == 1, "unexpected extra training attempt")
        attempt = Path(phase["attempts"][0])
        checkpoint = Path(phase["checkpoint"])
        require(
            sha(checkpoint) == phase["checkpoint_sha256"],
            "phase checkpoint hash mismatch",
        )
        initial = torch.load(
            attempt / "joint-step0.pt", map_location="cpu", weights_only=True
        )
        final = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoints[phase["id"]] = checkpoint_receipt(checkpoint, final)
        require(
            initial["optimizer"]["optimizer_state_dict"]["state"] == {},
            "new phase optimizer not fresh",
        )
        for name in ARMS:
            require(
                any(
                    not torch.equal(v, initial["drafts"][name][k])
                    for k, v in final["drafts"][name].items()
                ),
                f"{phase['id']}/{name}: no real weight update",
            )
            steps = [
                r["artifact"]["step"]
                for r in state["history"][phase["id"] + ":" + name]
            ]
            require(
                steps == [phase["offset"] + 1, phase["offset"] + 2],
                "policy history step mismatch",
            )
        curves = [
            json.loads(line)
            for line in (attempt / "learning-curves.jsonl").read_text().splitlines()
        ]
        train_rows = [r for r in curves if r["kind"] == "train"]
        for name in ARMS:
            require(
                [r["step"] for r in train_rows if r["arm"] == name] == [1, 2],
                "unexpected training updates",
            )
        peaks = [r["cuda_peak_allocated_bytes"] for r in train_rows]
        require(
            (
                all(p is not None and p > 0 for p in peaks)
                if args.device == "cuda"
                else all(p is None for p in peaks)
            ),
            "actual training device evidence mismatch",
        )
        memory_peaks.extend(peaks)
        closed = [r for r in curves if r["kind"] == "queue_closed"][-1]
        require(
            closed["resident_bytes"] == closed["resident_items"] == 0,
            "queue did not close",
        )
        require(
            closed["consumed"] == 4
            and closed["peak_bytes"] <= base["training"]["joint_feature_queue_bytes"],
            "queue cursor/capacity mismatch",
        )
        queues[phase["id"]] = closed
        report = read(attempt / "verification-step2.json")
        require(
            report["tokenparity"] and not report["production_engine_speed"],
            "reference verification failed",
        )
        require(
            report["artifact_sha256"] == sha(checkpoint),
            "verification/checkpoint binding mismatch",
        )
        require(
            len(report["candidates"]) == 3 and report["domains"] == ["chat", "code"],
            "verification coverage mismatch",
        )
        del initial, final
    source = torch.load(first["checkpoint"], map_location="cpu", weights_only=True)
    restarted = torch.load(
        Path(second["attempts"][0]) / "joint-step0.pt",
        map_location="cpu",
        weights_only=True,
    )
    for name in ARMS:
        require(
            all(
                torch.equal(v, restarted["drafts"][name][k])
                for k, v in source["drafts"][name].items()
            ),
            "weights-only phase initialization changed weights",
        )
        config_second = read(
            Path(second["attempts"][0]) / name / "resolved_config.json"
        )
        require(
            config_second["training"]["learning_rate"] == 1e-5,
            "registered LR override missing",
        )
    del source, restarted
    exports = {}
    for name, best in state["best"].items():
        folder = Path(best["export"])
        require(
            sha(best["checkpoint"]) == best["checkpoint_sha256"],
            "best checkpoint reference mismatch",
        )
        file_hashes = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        require(file_hashes == best["files"], "best export inventory changed")
        saved = torch.load(best["checkpoint"], map_location="cpu", weights_only=True)[
            "drafts"
        ][name]
        loaded = AutoDraftModel.from_pretrained(str(folder), local_files_only=True)
        actual = loaded.state_dict()
        require(
            all(torch.equal(v, actual[k].cpu()) for k, v in saved.items()),
            "best export reload differs",
        )
        exports[name] = dict(
            path=str(folder),
            checkpoint=best["checkpoint"],
            checkpoint_sha256=best["checkpoint_sha256"],
            step=best["step"],
            strategy=best["strategy"],
            files=file_hashes,
            exact_reload=True,
        )
        del loaded, actual, saved
    before = {
        str(p.relative_to(campaign)): sha(p)
        for p in campaign.rglob("*")
        if p.is_file()
        and (p.suffix in (".pt", ".safetensors") or p.name == "learning-curves.jsonl")
    }
    cli.append(run_cli(root, "completed-resume", args._deadline_unix - 1, resume=True))
    require(
        read(campaign / "state.json") == state,
        "completed resume changed controller state",
    )
    after = {
        str(p.relative_to(campaign)): sha(p)
        for p in campaign.rglob("*")
        if p.is_file()
        and (p.suffix in (".pt", ".safetensors") or p.name == "learning-curves.jsonl")
    }
    require(before == after, "completed resume retrained/changed checkpoints or curves")
    require(
        {p.name: sha(p) for p in [target / "config.json", *weights]} == target_hashes,
        "teacher bytes changed",
    )
    require(
        {str(p.relative_to(REPO)): sha(p) for p in source_paths} == source_hashes,
        "gate/controller/trainer source changed during execution; rerun on a stable revision",
    )
    receipt = dict(
        schema="draftfit-unattended-gate-v1",
        status="pass",
        device=args.device,
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        execution_seconds=time.monotonic() - started,
        actual_updates=4,
        phase_updates=[2, 2],
        forced_plateau=True,
        forced_min_delta=1e6,
        data_kind="synthetic_token_sequences",
        admission_kind="synthetic_local_test_only",
        provider_budget=False,
        provider_guard=False,
        publication="not_run",
        production_speed_claim=False,
        real_generation=True,
        target_kind="local_pretrained" if args.target_model else "tiny_random_qwen",
        target_model=str(target),
        target_revision=args.target_revision,
        target_files_sha256=target_hashes,
        teacher_sha_unchanged=True,
        weights_only_initialization_exact=True,
        fresh_phase_optimizer=True,
        completed_resume_idempotent=True,
        interrupted_exact_resume="not_run_optional",
        checkpoint_references=checkpoints,
        queues=queues,
        cuda_allocator_peaks_bytes=memory_peaks,
        best_exports=exports,
        cli=cli,
        source_sha256=source_hashes,
        source_unchanged_during_gate=True,
    )
    write(root / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--target-model", help="Existing local pretrained target; no download"
    )
    parser.add_argument("--draft-config", help="Compatible local draft JSON")
    parser.add_argument(
        "--target-revision", help="Immutable 40-character source revision"
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_deadline-unix", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 10 <= args.timeout_seconds <= 180:
        parser.error("timeout-seconds must be between 10 and 180")
    provided = (args.target_model, args.draft_config, args.target_revision)
    if any(provided) and not all(provided):
        parser.error(
            "target-model, draft-config and target-revision must be supplied together"
        )
    if args.target_model:
        if (
            not Path(args.target_model).is_dir()
            or not Path(args.draft_config).is_file()
            or not re.fullmatch(r"[a-fA-F0-9]{40}", args.target_revision)
        ):
            parser.error(
                "local target/config and immutable 40-character revision required"
            )
    root = Path(args.output_dir).resolve()
    if args._worker:
        try:
            exercise(args, root)
        except Exception as exc:
            write(
                root / "receipt.json",
                dict(
                    schema="draftfit-unattended-gate-v1",
                    status="fail",
                    device=args.device,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
            raise
        return 0
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = time.time() + args.timeout_seconds - 2
    env = {
        **os.environ,
        "SPECFORGE_DEVICE": args.device,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(REPO),
    }
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--output-dir",
        str(root),
        "--_worker",
        "--_deadline-unix",
        str(deadline),
    ]
    if args.target_model:
        command.extend(
            [
                "--target-model",
                str(Path(args.target_model).resolve()),
                "--draft-config",
                str(Path(args.draft_config).resolve()),
            ]
        )
    timed_out = False
    with (root / "gate.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(
                timeout=max(
                    0.01, args.timeout_seconds - 2 - (time.monotonic() - started)
                )
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also clear any CLI child orphaned by a gate assertion/exception.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
    receipt_path = root / "receipt.json"
    receipt = (
        read(receipt_path)
        if receipt_path.exists()
        else dict(schema="draftfit-unattended-gate-v1", status="fail")
    )
    if timed_out or process.returncode:
        receipt.update(
            status="fail",
            error_type=(
                "HardTimeout"
                if timed_out
                else receipt.get("error_type", "WorkerFailure")
            ),
        )
    receipt.update(
        total_seconds=time.monotonic() - started,
        hard_timeout_seconds=args.timeout_seconds,
    )
    write(receipt_path, receipt)
    print(
        json.dumps(
            dict(
                status=receipt["status"],
                device=args.device,
                total_seconds=receipt["total_seconds"],
                receipt=str(receipt_path),
            )
        )
    )
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
