"""Bounded joint batch probes and capture-inclusive runtime forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from draftfit.training.campaign_budget import (
    conservative_quantile,
    estimate_runtime,
    read_joint_profile,
)
from draftfit.training.hardware_tuning import candidates, choose


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def forecast(
    profile,
    *,
    current_step,
    target_step,
    available_seconds,
    overhead_seconds,
    hourly_usd=None,
    remaining_usd=None,
    transfer_reserve_usd=0.0,
    checkpoint_seconds=None,
    evaluation_seconds=None,
    verification_seconds=None,
    **settings,
):
    """Use measured recurring costs; monetary ceilings can only reduce time."""
    for name, value in [
        ("hourly_usd", hourly_usd),
        ("remaining_usd", remaining_usd),
        ("transfer_reserve_usd", transfer_reserve_usd),
    ]:
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if (hourly_usd is None) != (remaining_usd is None) or hourly_usd == 0:
        raise ValueError(
            "cost forecasting needs positive hourly_usd and remaining_usd together"
        )
    if hourly_usd is not None:
        available_seconds = min(
            available_seconds,
            max(0.0, remaining_usd - transfer_reserve_usd) * 3600 / hourly_usd,
        )
    bounds = dict(
        checkpoint_seconds=checkpoint_seconds,
        evaluation_seconds=evaluation_seconds,
        verification_seconds=verification_seconds,
    )
    costs = {}
    sources = {}
    for name, bound in bounds.items():
        if bound is not None and (
            type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0
        ):
            raise ValueError(f"{name} must be a finite positive conservative bound")
        measured = conservative_quantile(profile[name]) if profile[name] else None
        costs[name] = max(
            value for value in (measured, bound, 0.0) if value is not None
        )
        sources[name] = (
            "measured"
            if measured is not None
            else ("explicit_bound" if bound is not None else "unmeasured")
        )
    estimate = estimate_runtime(
        samples=profile["step_seconds"],
        current_step=current_step,
        target_step=target_step,
        available_seconds=available_seconds,
        overhead_seconds=overhead_seconds,
        **costs,
        **settings,
    )
    estimate["unmeasured_operations"] = [
        k
        for k in ("checkpoint_seconds", "evaluation_seconds", "verification_seconds")
        if not profile[k]
    ]
    estimate["operation_cost_sources"] = sources
    missing = [
        name
        for name, source in sources.items()
        if source == "unmeasured"
        and estimate.get(name.replace("_seconds", "_events"), 0)
    ]
    if missing:
        estimate.update(
            feasible=False,
            reason="unmeasured_recurring_cost",
            lower_bound_estimated_seconds=estimate["guard_estimated_seconds"],
            guard_estimated_seconds=None,
            median_estimated_seconds=None,
            max_feasible_step=None,
            missing_cost_bounds=missing,
        )
    estimate["estimated_cost_usd"] = (
        (estimate["guard_estimated_seconds"] * hourly_usd / 3600 + transfer_reserve_usd)
        if hourly_usd is not None and estimate["guard_estimated_seconds"] is not None
        else None
    )
    return estimate


def _worker(args):
    import torch

    from draftfit.family import build_family_plan
    from draftfit.training.joint_family import run_joint_family

    torch.set_num_threads(1)
    _, stages = build_family_plan(args.manifest)
    adjusted = []
    for stage in stages:
        cfg = deepcopy(stage.config)
        arm = Path(args.output) / stage.name
        arm.mkdir()
        draft_path = arm / "draft.json"
        stage.draft_config.to_json_file(str(draft_path))
        cfg.model.draft_model_config = str(draft_path)
        cfg.output_dir = str(arm)
        cfg.training.batch_size = args.batch_size
        cfg.training.accumulation_steps = args.effective_batch // args.batch_size
        cfg.training.max_steps = cfg.training.total_steps = args.probe_steps
        cfg.training.joint_verification = cfg.training.joint_control = None
        adjusted.append(replace(stage, config=cfg))
    try:
        result = run_joint_family(
            args.output,
            adjusted,
            deadline_unix=args.deadline_unix,
            probe_steps=args.probe_steps,
        )
    except torch.cuda.OutOfMemoryError:
        result = dict(status="oom", batch_size=args.batch_size)
    write(args.result, result)
    return 0


def batch_probe(
    manifest,
    output,
    *,
    maximum,
    effective_batch,
    probe_steps=3,
    max_seconds=120,
    device="cpu",
):
    """Probe fresh subprocesses; never change a live optimizer or exact resume."""
    if type(probe_steps) is not int or probe_steps < 2:
        raise ValueError(
            "probe_steps must include warmup and at least one measured update"
        )
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be finite and positive")
    batches = candidates(maximum, effective_batch)
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + max_seconds
    trials = []
    from draftfit.family import build_family_plan

    _, stages = build_family_plan(str(manifest))
    contract = hashlib.sha256(
        json.dumps(
            [
                {
                    "config": s.config.model_dump(mode="json"),
                    "draft": s.draft_config.to_dict(),
                }
                for s in stages
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    receipt = dict(
        schema="draftfit-calibration-v1",
        contract=contract,
        effective_batch_size=effective_batch,
        probe_steps=probe_steps,
        device=device,
        trials=trials,
        selected=None,
        scope="isolated fresh training probes; no optimizer resume or production speed claim",
    )
    write(output / "calibration.json", receipt)
    for batch in batches:
        if time.time() >= deadline:
            break
        if any(
            s.config.training.joint_feature_queue_bytes
            and s.config.training.joint_feature_queue_batches < effective_batch // batch
            for s in stages
        ):
            trials.append(
                dict(
                    batch_size=batch,
                    status="incompatible",
                    reason="feature queue cannot hold this accumulation window",
                )
            )
            write(output / "calibration.json", receipt)
            continue
        folder = output / f"batch-{batch}"
        folder.mkdir()
        result_file = folder / "result.json"
        command = [
            sys.executable,
            "-m",
            "draftfit.calibrate",
            "_worker",
            "--manifest",
            str(manifest),
            "--output",
            str(folder),
            "--result",
            str(result_file),
            "--batch-size",
            str(batch),
            "--effective-batch",
            str(effective_batch),
            "--probe-steps",
            str(probe_steps),
            "--deadline-unix",
            str(deadline),
        ]
        env = os.environ.copy()
        env.update(SPECFORGE_DEVICE=device, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (folder / "probe.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=max(0.01, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                trials.append(dict(batch_size=batch, status="timeout"))
                write(output / "calibration.json", receipt)
                break
        if proc.returncode or not result_file.exists():
            receipt["failure"] = (
                f'worker failed for batch {batch}; see {folder / "probe.log"}'
            )
            write(output / "calibration.json", receipt)
            raise RuntimeError(receipt["failure"])
        row = json.loads(result_file.read_text())
        trials.append(row)
        write(output / "calibration.json", receipt)
        if row["status"] == "oom":
            break
    try:
        receipt["selected"] = choose(trials)
    except RuntimeError as error:
        receipt["failure"] = str(error)
        write(output / "calibration.json", receipt)
        raise
    write(output / "calibration.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="draftfit calibrate")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser(
        "forecast", help="estimate time/cost from an actual joint monitor log"
    )
    f.add_argument("--profile", required=True)
    for name in ("current-step", "target-step"):
        f.add_argument("--" + name, type=int, required=True)
    f.add_argument("--available-seconds", type=float, required=True)
    f.add_argument("--overhead-seconds", type=float, required=True)
    f.add_argument("--minimum-samples", type=int, default=5)
    f.add_argument("--safety-factor", type=float, default=1.25)
    f.add_argument("--hourly-usd", type=float)
    f.add_argument("--remaining-usd", type=float)
    f.add_argument("--transfer-reserve-usd", type=float, default=0.0)
    for name in ("checkpoint", "evaluation", "verification"):
        f.add_argument("--" + name + "-interval", type=int, default=0)
        f.add_argument("--" + name + "-seconds", type=float)
    f.add_argument("--output")
    b = sub.add_parser(
        "batch",
        help="measure fresh joint batch candidates without touching live training",
    )
    b.add_argument("--manifest", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--maximum", type=int, required=True)
    b.add_argument("--effective-batch", type=int, required=True)
    b.add_argument("--probe-steps", type=int, default=3)
    b.add_argument("--max-seconds", type=float, default=120)
    b.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    for name in ("manifest", "output", "result"):
        w.add_argument("--" + name, required=True)
    for name in ("batch-size", "effective-batch", "probe-steps"):
        w.add_argument("--" + name, type=int, required=True)
    w.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            return _worker(args)
        values = vars(args).copy()
        values.pop("command")
        if args.command == "batch":
            result = batch_probe(**values)
        else:
            profile_path = values.pop("profile")
            output = values.pop("output")
            profile = read_joint_profile(profile_path, through_step=args.current_step)
            result = forecast(profile, **values)
            result["profile_sha256"] = hashlib.sha256(
                Path(profile_path).read_bytes()
            ).hexdigest()
            if output:
                write(output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
