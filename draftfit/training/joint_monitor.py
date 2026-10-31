"""Append-only joint learning curves and checkpoint-bound MLX observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


class JointMonitor:
    def __init__(self, root, stages, start_step, resume_from):
        self.root = Path(root)
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.pending = {}
        self.log_interval = stages[0].config.training.log_interval
        self.stages = stages
        self.origin = (
            "existing" if stages[0].config.model.draft_checkpoint_path else "scratch"
        )
        self.meta = dict(
            schema="draftfit-joint-monitor-v1",
            run_id=self.run_id,
            origin=self.origin,
            start_step=start_step,
            resume_from=str(resume_from) if resume_from else None,
            source_checkpoint=stages[0].config.model.draft_checkpoint_path,
            arms={s.name: s.config.training.strategy for s in stages},
        )
        (self.root / "monitor.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def emit(self, kind, step, arm, **values):
        append(
            self.root / "learning-curves.jsonl",
            dict(
                schema="draftfit-joint-monitor-v1",
                run_id=self.run_id,
                origin=self.origin,
                kind=kind,
                step=step,
                arm=arm,
                strategy=self.meta["arms"][arm],
                timestamp_unix=time.time(),
                elapsed_seconds=time.monotonic() - self.started,
                **values,
            ),
        )

    def train(self, step, arm, *, final=False, **values):
        """Step-average losses; throughput from total work / total train time."""
        group = self.pending.setdefault(
            arm,
            dict(
                start=step,
                count=0,
                seconds=0.0,
                samples=0.0,
                tokens=0.0,
                sums={},
                peaks={},
            ),
        )
        seconds = values["joint_step_seconds"]
        group["count"] += 1
        group["seconds"] += seconds
        group["samples"] += seconds * values["joint_samples_per_second"]
        group["tokens"] += seconds * values["joint_valid_tokens_per_second"]
        for key in ("train_loss", "train_accuracy", "grad_norm"):
            group["sums"][key] = group["sums"].get(key, 0.0) + values[key]
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            value = values[key]
            if value is not None:
                group["peaks"][key] = max(group["peaks"].get(key, 0), value)
        if step % self.log_interval != 0 and not final:
            return
        self.emit(
            "train",
            step,
            arm,
            step_start=group["start"],
            steps_in_window=group["count"],
            aggregation="mean_of_optimizer_steps",
            **{k: v / group["count"] for k, v in group["sums"].items()},
            joint_step_seconds=group["seconds"] / group["count"],
            joint_window_seconds=group["seconds"],
            joint_samples_per_second=group["samples"] / group["seconds"],
            joint_valid_tokens_per_second=group["tokens"] / group["seconds"],
            learning_rate=values["learning_rate"],
            cuda_peak_allocated_bytes=group["peaks"].get("cuda_peak_allocated_bytes"),
            cuda_peak_reserved_bytes=group["peaks"].get("cuda_peak_reserved_bytes"),
        )
        del self.pending[arm]

    def checkpoint(self, step, results):
        for result in results:
            emit_arms = result.get("emit_arm_checkpoints", True)
            arm_path = (
                (
                    result.get("arm_checkpoint")
                    or str(Path(result["output_dir"]) / f'{result["name"]}-step{step}')
                )
                if emit_arms
                else None
            )
            self.emit(
                "checkpoint",
                step,
                result["name"],
                checkpoint_id=f'{self.run_id}:{step}:{result["name"]}',
                checkpoint_path=result.get("joint_checkpoint") or arm_path,
                joint_checkpoint=result.get("joint_checkpoint"),
                emit_arm_checkpoints=emit_arms,
                arm_checkpoint=arm_path,
            )


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def bind_manifest(root, step, manifest):
    """Bind user-selected converted exports to a saved joint step, by arm name.

    This records attribution, not a proof that conversion preserved weight bytes.
    """
    if type(step) is not int or step < 0:
        raise ValueError("joint step must be a nonnegative integer")
    root = Path(root)
    meta = json.loads((root / "monitor.json").read_text())
    checkpoints = {
        r["arm"]: r["checkpoint_id"]
        for r in read_rows(root / "learning-curves.jsonl")
        if r["kind"] == "checkpoint" and r["step"] == step
    }
    names = [a["name"] for a in manifest["arms"] if a["mode"] != "baseline"]
    if (
        not names
        or len(names) != len(set(names))
        or any(n not in checkpoints for n in names)
    ):
        raise ValueError("manifest draft names must match saved joint checkpoint arms")
    return {
        **manifest,
        "joint_monitor": {
            "run_id": meta["run_id"],
            "step": step,
            "checkpoints": {n: checkpoints[n] for n in names},
        },
    }


def ingest_mlx(root, source):
    """Import only complete, parity-passing, explicitly checkpoint-bound runs."""
    root, source = Path(root), Path(source)
    rows = read_rows(source)
    if (
        not rows
        or rows[0].get("kind") != "provenance"
        or rows[-1].get("kind") != "summary"
    ):
        raise ValueError("benchmark must include provenance and a final summary")
    provenance = rows[0]
    binding = provenance["manifest"].get("joint_monitor")
    meta = json.loads((root / "monitor.json").read_text())
    if not binding or binding.get("run_id") != meta["run_id"]:
        raise ValueError("benchmark manifest must bind to this monitor run_id")
    step = binding["step"]
    if type(step) is not int or step < 0:
        raise ValueError("invalid checkpoint step")
    checkpoints = {
        (r["step"], r["arm"]): r["checkpoint_id"]
        for r in read_rows(root / "learning-curves.jsonl")
        if r["kind"] == "checkpoint"
    }
    arms = provenance["manifest"]["arms"]
    baselines = [a["name"] for a in arms if a["mode"] == "baseline"]
    if len(baselines) != 1 or rows[-1].get("failed_arms"):
        raise ValueError("one baseline and no failed arms required")
    measured = [r for r in rows if r["kind"] == "measurement"]
    expected = {
        (a["name"], case, trial)
        for a in arms
        for case in provenance["selected_ids"]
        for trial in range(provenance["trials"])
    }
    keys = [(r["arm"], r["id"], r["trial"]) for r in measured]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("incomplete or duplicated measured request set")
    for r in measured:
        if (
            r.get("parity") is not True
            or len(r["result"]["token_ids"]) > provenance["max_new_tokens"]
        ):
            raise ValueError("measurement failed exact parity or output budget")
        for key in ("tok_s", "decode_tok_s"):
            if not math.isfinite(r[key]) or r[key] <= 0:
                raise ValueError("invalid generation throughput")
        for rd in r["rounds"]:
            if not (
                type(rd["accepted"]) is int
                and type(rd["drafted"]) is int
                and 0 <= rd["accepted"] <= rd["drafted"]
            ):
                raise ValueError("invalid acceptance counters")
    reference = {(r["id"], r["trial"]): r for r in measured if r["arm"] == baselines[0]}
    for row in measured:
        baseline = reference[(row["id"], row["trial"])]
        if (
            row["result"]["token_ids"] != baseline["result"]["token_ids"]
            or row["domain"] != baseline["domain"]
        ):
            raise ValueError("baseline tokens or domain differ")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = root / "generation-curves.jsonl"
    if destination.exists() and any(
        r["benchmark_sha256"] == digest for r in read_rows(destination)
    ):
        raise ValueError("benchmark already imported")
    output = []
    for arm in arms:
        name = arm["name"]
        if arm["mode"] == "baseline":
            continue
        identity = binding.get("checkpoints", {}).get(name)
        if identity is None or checkpoints.get((step, name)) != identity:
            raise ValueError(f"unknown checkpoint binding for {name}")
        for domain in [None] + sorted({r["domain"] for r in measured}):
            selected = [
                r
                for r in measured
                if r["arm"] == name and (domain is None or r["domain"] == domain)
            ]
            baseline = [
                r
                for r in measured
                if r["arm"] == baselines[0]
                and (domain is None or r["domain"] == domain)
            ]
            accepted = sum(x["accepted"] for r in selected for x in r["rounds"])
            proposed = sum(x["drafted"] for r in selected for x in r["rounds"])
            speed = statistics.median(r["tok_s"] for r in selected)
            base_speed = statistics.median(r["tok_s"] for r in baseline)
            output.append(
                dict(
                    schema="draftfit-joint-monitor-v1",
                    kind="generation",
                    run_id=meta["run_id"],
                    origin=meta["origin"],
                    step=step,
                    arm=name,
                    strategy=meta["arms"][name],
                    domain=domain,
                    checkpoint_id=identity,
                    requests=len(selected),
                    accepted_tokens=accepted,
                    proposed_tokens=proposed,
                    acceptance_rate=accepted / proposed if proposed else None,
                    median_tok_s=speed,
                    median_decode_tok_s=statistics.median(
                        r["decode_tok_s"] for r in selected
                    ),
                    baseline_median_tok_s=base_speed,
                    speedup_vs_target=speed / base_speed,
                    exact_token_parity=True,
                    benchmark_sha256=digest,
                    data_sha256=provenance["data_sha256"],
                    runtime_commit=provenance["runtime_commit"],
                    runtime_diff_sha256=provenance["runtime_diff_sha256"],
                    benchmark_provenance=provenance,
                    timestamp_unix=time.time(),
                )
            )
    if not output:
        raise ValueError("no draft observations")
    # Validate everything before appending; one writer per run directory.
    for row in output:
        append(destination, row)
    return output


def export_csv(root, output):
    rows = []
    for name in ("learning-curves.jsonl", "generation-curves.jsonl"):
        path = Path(root) / name
        if path.exists():
            rows.extend(read_rows(path))
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with Path(output).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["step"], r["arm"], r["kind"])):
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--import-mlx", type=Path)
    group.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.import_mlx:
        print(
            json.dumps({"observations": len(ingest_mlx(args.run_dir, args.import_mlx))})
        )
    else:
        export_csv(args.run_dir, args.csv)


if __name__ == "__main__":
    main()
