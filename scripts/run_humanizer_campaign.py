"""Resumable Humanizer generation, joint training, and verified HF exports.

Full runs need the pinned seed and either two token JSONL files or two input
draft files. Tiny runs take a local family manifest with a real tiny teacher.
The process never rents a GPU or claims a completed run from a checkpoint alone.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAMILIES = ("dflash", "dspark", "dflash2")
SEED_PATTERNS = ("config.json", "model.safetensors")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_tokens(path: Path, count: int | None) -> int:
    from draftfit.training.joint_online import _read_rows
    rows, _, _, _ = _read_rows(path, 512)
    if count is not None and len(rows) != count:
        raise ValueError(f"{path}: expected exactly {count} token rows, found {len(rows)}")
    return len(rows)


def check_limit(args, started: float, phase: str) -> float:
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        raise TimeoutError(f"deadline reached before {phase}")
    projected = args.prior_spend_usd + (time.monotonic() - started) * args.hourly_usd / 3600
    remaining = args.budget_usd - projected
    if remaining <= 0:
        raise TimeoutError(f"budget reached before {phase}: estimated ${projected:.3f}")
    return remaining


def seed_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    from draftfit.humanizer_campaign import SEED_ID, SEED_REVISION
    return Path(snapshot_download(SEED_ID, revision=SEED_REVISION,
                                  allow_patterns=list(SEED_PATTERNS), local_files_only=True))


def token_source(args, split: str, required: int, started: float) -> Path:
    supplied = getattr(args, f"{split}_tokens")
    if supplied is not None:
        source = supplied.resolve()
        count_tokens(source, None if args.stop_at else required)
        return source
    final = args.root / "pairs" / f"{split}.tokens.jsonl"
    manifest = final.with_name(f"{split}.manifest.json")
    if final.exists():
        if not manifest.exists() or json.loads(manifest.read_text()).get("count") != required:
            raise ValueError(f"{split}: final tokens need a matching ready manifest")
        count_tokens(final, required)
        return final
    source = getattr(args, f"{split}_input")
    if source is None:
        raise ValueError(f"{split}: provide --{split}-tokens or --{split}-input")
    remaining = check_limit(args, started, f"{split} generation")
    command = [sys.executable, str(Path(__file__).with_name("generate_humanizer_campaign_pairs.py")),
               "--input", str(source), "--output-dir", str(args.root / "pairs"),
               "--required-count", str(required), "--batch-size", str(args.generation_batch_size),
               "--hourly-usd", str(args.hourly_usd),
               "--max-projected-usd", str(min(args.generation_budget_usd, remaining)),
               "--local-files-only"]
    budget_deadline = time.time() + remaining * 3600 / args.hourly_usd
    command += ["--deadline-unix", str(min(args.deadline_unix or float("inf"), budget_deadline))]
    for reserve in getattr(args, f"{split}_reserve"):
        command += ["--reserve-input", str(reserve)]
    with (args.root / f"generation-{split}.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{split} generation exited {result.returncode}; see generation-{split}.log")
    count_tokens(final, required)
    return final


def prepare_manifest(args, train: Path, validation: Path) -> Path:
    if args.manifest:
        return args.manifest.resolve()
    if args.tiny:
        if not args.fixture_manifest:
            raise ValueError("--tiny requires --fixture-manifest with local tiny Gemma target and seed")
        return args.fixture_manifest.resolve()
    from draftfit.humanizer_campaign import prepare
    folder = args.root / "config"
    if not folder.exists():
        prepare(folder, seed_snapshot(), train_tokens=train, validation_tokens=validation)
    else:
        base = json.loads((folder / "base.json").read_text())
        if base["data"]["prompts_path"] != str(train) or base["data"]["eval_data_path"] != str(validation):
            raise ValueError("prepared config has different token sources")
    path = folder / "base.json"
    base = json.loads(path.read_text())
    base["training"].update(num_epochs=1000 if args.stop_at else 1, max_steps=30000, total_steps=30000,
                            batch_size=1, accumulation_steps=1, max_checkpoints=1,
                            log_interval=10, save_interval=250, eval_interval=1000)
    atomic_json(path, base)
    return folder / "family.json"


def configured_stages(manifest: Path, run_dir: Path, tiny: bool, diagnostic: bool = False,
                      staged: bool = False):
    from draftfit.family import build_family_plan
    from draftfit.training.joint_family import validate_joint_family
    _, stages = build_family_plan(str(manifest))
    if {stage.config.training.strategy for stage in stages} != set(FAMILIES) or len(stages) != 3:
        raise ValueError("campaign requires exactly DFlash, DSpark and DFlash2")
    adjusted = []
    for stage in stages:
        cfg = stage.config.model_copy(deep=True)
        cfg.output_dir = str(run_dir / stage.name)
        if staged:
            samples = count_tokens(Path(cfg.data.prompts_path), None)
            cfg.training.num_epochs = math.ceil((10 if tiny else 30000) / samples)
            cfg.training.max_steps = None
            cfg.training.total_steps = 10 if tiny else 30000
            cfg.training.batch_size = cfg.training.accumulation_steps = 1
            cfg.training.save_interval = cfg.training.eval_interval = 5 if tiny else 250
            cfg.training.log_interval = 1 if tiny else 10
            cfg.training.max_checkpoints = 1
        elif tiny:
            cfg.training.num_epochs = max(cfg.training.num_epochs, 3)
            cfg.training.max_steps = 10
            cfg.training.total_steps = 10
            cfg.training.batch_size = cfg.training.accumulation_steps = 1
            cfg.training.save_interval = cfg.training.eval_interval = 5
            cfg.training.log_interval = 1
            cfg.training.max_checkpoints = 1
        elif diagnostic:
            samples = count_tokens(Path(cfg.data.prompts_path), None)
            cfg.training.num_epochs = max(cfg.training.num_epochs, math.ceil(30000 / samples))
            cfg.training.max_steps = cfg.training.total_steps = 30000
            cfg.training.batch_size = cfg.training.accumulation_steps = 1
            cfg.training.save_interval = cfg.training.eval_interval = 5
            cfg.training.log_interval = 1
            cfg.training.max_checkpoints = 1
        else:
            cfg.training.num_epochs = 1
            cfg.training.max_steps = cfg.training.total_steps = 30000
            cfg.training.batch_size = cfg.training.accumulation_steps = 1
            cfg.training.save_interval = 250
            cfg.training.eval_interval = 1000
            cfg.training.log_interval = 10
            cfg.training.max_checkpoints = 1
        adjusted.append(replace(stage, config=cfg))
    validate_joint_family(adjusted)
    return adjusted


def latest_checkpoint(root: Path) -> tuple[int, Path | None]:
    found = [(int(p.stem.removeprefix("joint-step")), p)
             for p in root.glob("attempt-*/joint-step*.pt")]
    return max(found, default=(0, None), key=lambda item: item[0])


def train_worker(manifest: Path, run_dir: Path, resume: Path | None, stop: int,
                 tiny: bool, diagnostic: bool, deadline_unix: float | None,
                 data_phase: int | None = None) -> dict:
    from draftfit.training.joint_family import run_joint_family
    stages = configured_stages(manifest, run_dir, tiny, diagnostic, staged=data_phase is not None)
    for stage in stages:
        folder = Path(stage.config.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stage.draft_config.to_json_file(str(folder / "draft.json"))
        stage.config.model.draft_model_config = str(folder / "draft.json")
        atomic_json(folder / "resolved.json", stage.config.model_dump(mode="json"))
    options = {"resume_from": resume, "stop_at_step": stop, "deadline_unix": deadline_unix}
    if data_phase is not None:
        options["data_phase"] = data_phase
    result = run_joint_family(run_dir, stages, **options)
    atomic_json(run_dir / "worker-result.json", result)
    return result


def launch_training(args, manifest: Path, run_dir: Path, resume: Path | None,
                    stop: int, started: float, data_phase: int | None = None) -> None:
    check_limit(args, started, f"training to step {stop}")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--fixture-manifest", str(manifest),
               "--worker-output", str(run_dir), "--worker-stop", str(stop)]
    if args.tiny:
        command.append("--tiny")
    if resume:
        command += ["--worker-resume", str(resume)]
    if args.stop_at and not args.tiny:
        command.append("--worker-diagnostic")
    if data_phase is not None:
        command += ["--worker-data-phase", str(data_phase)]
    remaining_seconds = (args.budget_usd - args.prior_spend_usd) * 3600 / args.hourly_usd - (time.monotonic() - started)
    effective_deadline = min(args.deadline_unix or float("inf"), time.time() + remaining_seconds)
    command += ["--deadline-unix", str(effective_deadline)]
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(1)
            if time.time() > effective_deadline + 120:
                process.terminate()
                process.wait(timeout=30)
                raise TimeoutError("joint worker exceeded deadline grace; last checkpoint retained")
        if process.returncode:
            raise RuntimeError(f"joint worker exited {process.returncode}; see {run_dir / 'worker.log'}")


def export_and_verify(result: dict, run_dir: Path, destination: Path) -> list[dict]:
    import torch
    from draftfit.export.to_hf import export_to_hf
    from draftfit.modeling.auto import AutoDraftModel
    arms = []
    destination.mkdir(parents=True, exist_ok=True)
    for row in result["runs"]:
        name = row["name"]
        checkpoint = Path(row["latest"]).resolve()
        target = json.loads((run_dir / name / "resolved.json").read_text())["model"]["target_model_path"]
        output = destination / name
        export_to_hf(str(checkpoint), str(checkpoint / "config.json"), str(output), embedding_source=target)
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)["draft_state_dict"]
        restored = AutoDraftModel.from_pretrained(output).state_dict()
        for key, value in state.items():
            if key not in restored or not torch.equal(value.cpu(), restored[key].cpu()):
                raise ValueError(f"{name}: exported weight differs at {key}")
        arms.append({"name": name, "step": row["global_step"], "export": str(output),
                     "weights_sha256": sha(output / "model.safetensors")})
    return arms


def run(args) -> dict:
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = args.root / "status.json"
    previous_receipt = json.loads(receipt.read_text()) if receipt.exists() else None
    status = {"status": "running", "tiny": args.tiny, "completed_step": 0, "arms": []}
    atomic_json(receipt, status)
    try:
        if args.manifest and not any((args.train_tokens, args.validation_tokens,
                                      args.train_input, args.validation_input)):
            from draftfit.family import build_family_plan
            _, planned = build_family_plan(str(args.manifest))
            config = planned[0].config
            if not config.training.joint_online_capture:
                raise ValueError("--manifest must use joint online token sources")
            train, validation = Path(config.data.prompts_path), Path(config.data.eval_data_path)
            count_tokens(train, None if args.stop_at else (12 if args.tiny else 30000))
            count_tokens(validation, None if args.stop_at else (2 if args.tiny else 500))
        else:
            train = token_source(args, "train", 12 if args.tiny else 30000, started)
            validation = token_source(args, "validation", 2 if args.tiny else 500, started)
        status["tokens"] = {"train": {"path": str(train), "sha256": sha(train)},
                            "validation": {"path": str(validation), "sha256": sha(validation)}}
        if previous_receipt and previous_receipt.get("tokens") and previous_receipt["tokens"] != status["tokens"]:
            raise ValueError("existing run has different token sources; use a new output directory")
        atomic_json(receipt, status)
        check_limit(args, started, "configuration")
        manifest = prepare_manifest(args, train, validation)
        status["manifest_sha256"] = sha(manifest)
        if previous_receipt and previous_receipt.get("manifest_sha256") and previous_receipt["manifest_sha256"] != status["manifest_sha256"]:
            raise ValueError("existing run has a different family manifest")
        atomic_json(receipt, status)
        horizon = 10 if args.tiny else 30000
        requested = args.stop_at or horizon
        milestones = tuple(args.milestones) if args.milestones else ((5, 10) if args.tiny and not args.stop_at else (requested,))
        for stop in milestones:
            current, previous = latest_checkpoint(args.root / "run")
            if args.resume and previous is None:
                previous = args.resume.resolve()
                current = int(previous.stem.removeprefix("joint-step"))
            if current >= stop:
                continue
            run_dir = args.root / "run" / f"attempt-{stop}"
            if run_dir.exists():
                # An interrupted attempt may contain an atomic checkpoint.
                run_dir = args.root / "run" / f"attempt-{stop}-{int(time.time())}"
            launch_training(args, manifest, run_dir, previous, stop, started)
            current, _ = latest_checkpoint(args.root / "run")
            if current < stop:
                break
            status["completed_step"] = stop
            atomic_json(receipt, status)
        current, checkpoint = latest_checkpoint(args.root / "run")
        if checkpoint is None:
            raise RuntimeError("joint training checkpoint is missing")
        run_dir = checkpoint.parent
        result = json.loads((run_dir / "worker-result.json").read_text())
        status["arms"] = export_and_verify(result, run_dir, args.root / "exports")
        completed = current == horizon and not args.stop_at and result.get("status") != "stopped"
        status.update(status="complete" if completed else "diagnostic" if args.stop_at else "stopped",
                      completed_step=current,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      completed_at_unix=time.time())
        if result.get("status") == "stopped":
            status["reason"] = result.get("reason")
    except BaseException as exc:
        status.update(status="stopped" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        current, checkpoint = latest_checkpoint(args.root / "run")
        status["completed_step"] = current
        if checkpoint:
            status["checkpoint"] = str(checkpoint)
        atomic_json(receipt, status)
        raise
    atomic_json(receipt, status)
    return status


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", "--output-dir", type=Path)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--fixture-manifest", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--stop-at", type=int)
    p.add_argument("--milestones", type=int, nargs="+")
    p.add_argument("--resume", type=Path)
    for split in ("train", "validation"):
        p.add_argument(f"--{split}-tokens", type=Path)
        p.add_argument(f"--{split}-input", type=Path)
        p.add_argument(f"--{split}-reserve", type=Path, action="append", default=[])
    p.add_argument("--deadline-unix", type=float)
    p.add_argument("--budget-usd", type=float, default=3.0)
    p.add_argument("--prior-spend-usd", type=float)
    p.add_argument("--hourly-usd", type=float, default=0.4737314815)
    p.add_argument("--generation-budget-usd", type=float, default=1.10)
    p.add_argument("--generation-batch-size", type=int, default=16)
    p.add_argument("--staged", action="store_true")
    p.add_argument("--stage-rows", type=int, default=1000)
    p.add_argument("--stage-first-steps", type=int, default=250)
    p.add_argument("--stage-patience", type=int, default=3)
    p.add_argument("--stage-min-delta", type=float, default=0.0)
    p.add_argument("--continuation-journal", type=Path,
                   help="original train.partial.jsonl to extend an exhausted immutable train pool")
    p.add_argument("--continuation-start-row", type=int,
                   help="accepted journal rows reserved for the original pool and fixed held-out set")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-resume", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-stop", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker-diagnostic", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-data-phase", type=int, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.milestones:
        if args.milestones != sorted(set(args.milestones)) or args.milestones[0] < 1 or args.milestones[-1] > (10 if args.tiny else 30000):
            p.error("milestones must increase uniquely within the horizon")
        args.stop_at = args.milestones[-1] if args.milestones[-1] < (10 if args.tiny else 30000) else None
    if args.worker:
        train_worker(args.fixture_manifest, args.worker_output, args.worker_resume,
                     args.worker_stop, args.tiny, args.worker_diagnostic, args.deadline_unix,
                     args.worker_data_phase)
        return 0
    if args.root is None or (not args.tiny and args.prior_spend_usd is None):
        p.error("--root is required; full runs also require --prior-spend-usd")
    if args.manifest and args.fixture_manifest:
        p.error("choose --manifest or --fixture-manifest")
    if args.tiny and not (args.manifest or args.fixture_manifest):
        p.error("--tiny requires --manifest or --fixture-manifest")
    if args.stop_at is not None and not 1 <= args.stop_at <= (10 if args.tiny else 30000):
        p.error("--stop-at must be within the horizon")
    args.prior_spend_usd = 0.0 if args.prior_spend_usd is None else args.prior_spend_usd
    amounts = (args.prior_spend_usd, args.budget_usd, args.hourly_usd,
               args.generation_budget_usd)
    if (any(not math.isfinite(value) for value in amounts)
            or not 0 <= args.prior_spend_usd < args.budget_usd
            or args.hourly_usd <= 0 or args.generation_budget_usd <= 0
            or args.generation_batch_size < 1):
        p.error("budget, hourly rate, generation budget and batch size must be finite and positive")
    if args.deadline_unix is not None and not math.isfinite(args.deadline_unix):
        p.error("deadline must be a finite Unix timestamp")
    if args.staged and (args.stop_at or args.milestones or args.resume):
        p.error("staged campaign manages its own milestones and resume")
    if args.continuation_journal is not None:
        if (not args.staged or not args.train_tokens or not args.validation_tokens or not args.train_input
                or args.continuation_start_row is None or not 0 < args.continuation_start_row < 30000):
            p.error("continuation requires --staged, original train/validation tokens, --train-input and a start row within 30000")
    elif args.continuation_start_row is not None:
        p.error("--continuation-start-row requires --continuation-journal")
    if (args.stage_rows < 1 or args.stage_first_steps < 1 or args.stage_patience < 1
            or not math.isfinite(args.stage_min_delta) or args.stage_min_delta < 0):
        p.error("invalid staged campaign settings")
    try:
        if args.staged:
            from scripts.humanizer_staged import run_staged
            result = run_staged(args)
        else:
            result = run(args)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
