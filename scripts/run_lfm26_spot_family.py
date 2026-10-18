"""Restartable, budget-bounded LFM2.6 family experiment on an existing GPU.

This script never rents or destroys an instance. The caller supplies the actual
lease start and inclusive hourly price; its conservative wall-clock deadline
also counts time spent stopped or preparing the machine.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from draftfit.family import build_family_plan


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _source_digest(manifest: Path) -> str:
    raw = manifest.read_bytes()
    base = json.loads(raw)["base_config"]
    return hashlib.sha256(raw + (manifest.parent / base).read_bytes()).hexdigest()


def _last_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    last: dict = {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break  # A preemption can leave one torn final JSONL row.
            row = json.loads(line)
            if any(key.startswith("eval/") for key in row):
                last["eval_step"] = row["step"]
            if any(key.startswith("train/") for key in row):
                last["train_step"] = row["step"]
            last.update(row)
    return last


def _validate_matrix(stages) -> None:
    if [stage.config.training.strategy for stage in stages] != [
        "dspark", "dspark", "dflash", "dflash2"
    ]:
        raise ValueError("expected DSpark warm start, DSpark continual, DFlash, DFlash2")
    first = stages[0]
    if not first.config.model.draft_checkpoint_path or first.parent is not None:
        raise ValueError("first DSpark stage needs an official checkpoint")
    if any(stage.parent != first.name for stage in stages[1:]):
        raise ValueError("all later stages must hand off from the first DSpark stage")
    if not all(stage.config.training.retrain_anchor_alignment for stage in stages[2:]):
        raise ValueError("DFlash conversions require retrain_anchor_alignment=true")
    if any(stage.config.training.save_interval <= 0 for stage in stages):
        raise ValueError("each stage needs a positive save_interval for preemption recovery")
    if any(stage.config.training.eval_interval <= 0 for stage in stages):
        raise ValueError("each stage needs held-out evaluation")


def _latest(stage) -> Path | None:
    path = Path(stage.config.output_dir) / f"{stage.name}-latest"
    return path if path.is_dir() and (path / "training_state.pt").is_file() else None


def _completed(stage, state: dict) -> bool:
    if state.get("status") != "complete" or _latest(stage) is None:
        return False
    metrics = _last_metrics(Path(stage.config.output_dir) / "metrics.jsonl")
    return "eval/avg_loss" in metrics and "eval/avg_acc" in metrics


def _run_train(command: list[str], log_path: Path, deadline: float,
               lock_fd: int) -> tuple[int, bool]:
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n=== started {time.time():.3f}: {json.dumps(command)} ===\n")
        log.flush()
        os.fsync(log.fileno())
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=(lock_fd,),
        )
        def stop_process() -> None:
            if process.poll() is not None:
                return
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        try:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 0)
            return process.wait(timeout=remaining), False
        except subprocess.TimeoutExpired:
            stop_process()
            return process.returncode, True
        except BaseException:
            stop_process()
            raise
        finally:
            log.write(f"=== stopped {time.time():.3f}: exit={process.returncode} ===\n")
            log.flush()
            os.fsync(log.fileno())


def run(manifest: Path, *, hourly_usd: float, max_usd: float, lease_start_epoch: float,
        stop_reserve_seconds: float = 60, plan: bool = False) -> dict:
    if hourly_usd <= 0 or max_usd <= 0 or lease_start_epoch <= 0 or stop_reserve_seconds < 0:
        raise ValueError("hourly price, budget and lease start must be positive")
    manifest = manifest.expanduser().resolve()
    root, stages = build_family_plan(str(manifest))
    _validate_matrix(stages)
    deadline = lease_start_epoch + max_usd / hourly_usd * 3600 - stop_reserve_seconds
    summary = {"output_dir": str(root), "deadline_epoch": deadline,
               "runs": [{"name": stage.name, "strategy": stage.config.training.strategy,
                         "from": stage.parent} for stage in stages]}
    if plan:
        return summary
    if root.is_symlink():
        raise ValueError("output root must not be a symlink")
    if root.exists() and not (root / "orchestration.json").is_file():
        raise FileExistsError(f"refusing existing experiment output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = root / "orchestration.json"
        identity = {"source_sha256": _source_digest(manifest), "hourly_usd": hourly_usd,
                    "max_usd": max_usd, "lease_start_epoch": lease_start_epoch,
                    "stop_reserve_seconds": stop_reserve_seconds}
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError("manifest or cost contract changed; use a fresh output directory")
        else:
            record = {**identity, "stages": {}}
            _write_json(record_path, record)
        for stage in stages:
            if stage.parent and not _completed(
                next(item for item in stages if item.name == stage.parent),
                record["stages"].get(stage.parent, {}),
            ):
                raise RuntimeError(f"parent stage {stage.parent} is not complete")
            state = record["stages"].get(stage.name, {})
            if _completed(stage, state):
                continue
            if time.time() >= deadline:
                state.update(status="budget_stop", stopped_epoch=time.time())
                record["stages"][stage.name] = state
                _write_json(record_path, record)
                break
            output = Path(stage.config.output_dir)
            if output.is_symlink():
                raise ValueError(f"stage output must not be a symlink: {output}")
            if not output.exists():
                output.mkdir()
                stage.draft_config.to_json_file(str(output / "draft.json"))
                _write_json(output / "train.json", stage.config.model_dump(mode="json"))
            elif not (output / "train.json").is_file() or not (output / "draft.json").is_file():
                raise FileExistsError(f"refusing incomplete foreign stage directory: {output}")
            original = stage.config.model_dump(mode="json")
            if json.loads((output / "train.json").read_text()) != original:
                raise ValueError(f"stage config changed: {stage.name}")
            checkpoint = _latest(stage)
            if state and checkpoint is None and state.get("status") == "failed":
                raise RuntimeError(f"{stage.name} failed without a checkpoint; inspect train.log")
            if checkpoint:
                resume = stage.config.model_dump(mode="json")
                resume["model"]["draft_checkpoint_path"] = None
                resume["training"]["resume_from"] = str(checkpoint)
                _write_json(output / "resume.json", resume)
                config_path = output / "resume.json"
            else:
                config_path = output / "train.json"
            state.update(status="running", resumed_from=str(checkpoint) if checkpoint else None,
                         started_epoch=time.time(), attempts=state.get("attempts", 0) + 1)
            record["stages"][stage.name] = state
            _write_json(record_path, record)
            code, timed_out = _run_train(
                [sys.executable, "-u", "-m", "draftfit.cli", "train", "-c", str(config_path)],
                output / "train.log", deadline, lock.fileno(),
            )
            metrics = _last_metrics(output / "metrics.jsonl")
            state.update(exit_code=code, metrics=metrics, stopped_epoch=time.time())
            if timed_out:
                state["status"] = "budget_stop"
            elif code == 0 and _latest(stage) is not None and {
                "eval/avg_loss", "eval/avg_acc"
            } <= metrics.keys():
                state["status"] = "complete"
            else:
                state["status"] = "failed"
            _write_json(record_path, record)
            if state["status"] != "complete":
                break
        return {**summary, "stages": record["stages"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True,
                        help="inclusive spot GPU + disk hourly price")
    parser.add_argument("--max-usd", type=float, required=True,
                        help="maximum cost allocated to this instance")
    parser.add_argument("--lease-start-epoch", type=float, required=True,
                        help="Vast lease start UNIX timestamp, reused across restarts")
    parser.add_argument("--stop-reserve-seconds", type=float, default=60)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest, hourly_usd=args.hourly_usd, max_usd=args.max_usd,
                 lease_start_epoch=args.lease_start_epoch,
                 stop_reserve_seconds=args.stop_reserve_seconds, plan=args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.plan or all(
        value["status"] == "complete" for value in result["stages"].values()
    ) and len(result["stages"]) == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
