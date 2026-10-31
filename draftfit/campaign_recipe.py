"""Generate fixed-horizon joint milestones and optional ranked branch jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draftfit.campaign import read, validate, write


def build(spec):
    steps = spec.get("milestones", [1000, 3000, 10000, 20000, 30000])
    if (
        not steps
        or any(type(x) is not int or x < 1 or x > 30000 for x in steps)
        or sorted(set(steps)) != steps
    ):
        raise ValueError("milestones must increase, with a maximum of 30000")
    patience = spec.get("patience", 2)
    if type(patience) is not int or patience < 1:
        raise ValueError("patience must be positive")
    jobs = []
    targets = spec["targets"]
    if len({t["name"] for t in targets}) != len(targets):
        raise ValueError("duplicate target names")
    histories = {t["name"]: [] for t in targets}
    previous = {}
    # Round robin: while target A is being evaluated, B can occupy the GPU.
    for index, step in enumerate(steps):
        for target in targets:
            name = target["name"]
            train_id, eval_id = f"{name}-train-{step}", f"{name}-eval-{step}"
            command = [
                spec.get("python", sys.executable),
                "-m",
                "draftfit.campaign_joint",
                "--manifest",
                target["manifest"],
                "--stop-at",
                str(step),
            ]
            deps = []
            if name in previous:
                command += ["--resume-job", previous[name]]
                deps = [previous[name]] + histories[name][-patience - 1 :]
            train = dict(
                id=train_id,
                kind="train",
                resource=target.get("training_resource", "gpu"),
                argv=command,
                depends_on=deps,
                timeout_seconds=target.get("training_timeout_seconds", 21600),
            )
            if index:
                train["gate"] = dict(
                    evaluations=(
                        histories[name][-patience - 1 :]
                        if len(histories[name]) > patience
                        else histories[name][-1:]
                    ),
                    top_k=3,
                    min_speedup=spec.get("min_speedup", 1.0),
                    min_domain_speedup=spec.get("min_domain_speedup", 0.9),
                    min_relative_gain=spec.get("min_relative_gain", 0.01),
                )
            jobs.append(train)
            jobs.append(
                dict(
                    id=eval_id,
                    kind="evaluate",
                    resource=target.get("evaluation_resource", "validation"),
                    argv=target["evaluate"],
                    depends_on=[train_id],
                    train_job=train_id,
                    comparison_id=target["comparison_id"],
                    min_requests=spec.get("min_requests", 32),
                    timeout_seconds=target.get("evaluation_timeout_seconds", 7200),
                )
            )
            histories[name].append(eval_id)
            previous[name] = train_id
            if step == spec.get("branch_at", 3000):
                for branch in target.get("branches", []):
                    jobs.append(
                        dict(
                            id=f"{name}-branch-{branch['name']}",
                            kind="train",
                            resource=target.get("training_resource", "gpu"),
                            argv=branch["train"],
                            depends_on=[train_id, eval_id],
                            timeout_seconds=branch.get("timeout_seconds", 7200),
                            gate=dict(
                                evaluations=[eval_id],
                                family=branch["family"],
                                top_k=spec.get("branch_top_k", 2),
                                min_speedup=spec.get("branch_min_speedup", 1.05),
                                min_domain_speedup=spec.get("min_domain_speedup", 0.9),
                                min_relative_gain=0,
                            ),
                        )
                    )
                    branch_id = jobs[-1]["id"]
                    jobs.append(
                        dict(
                            id=branch_id + "-eval",
                            kind="evaluate",
                            resource=target.get("evaluation_resource", "validation"),
                            argv=branch["evaluate"],
                            depends_on=[branch_id],
                            train_job=branch_id,
                            comparison_id=target["comparison_id"],
                            min_requests=spec.get("min_requests", 32),
                            timeout_seconds=target.get(
                                "evaluation_timeout_seconds", 7200
                            ),
                        )
                    )
    config = dict(
        schema="draftfit-campaign-v1",
        budget=spec["budget"],
        fail_fast=spec.get("fail_fast", False),
        resources=spec.get("resources", {"gpu": 1, "validation": 1}),
        jobs=jobs,
    )
    validate(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    base = Path(args.spec).resolve().parent
    for target in spec["targets"]:
        target["manifest"] = str((base / target["manifest"]).resolve())
    output = Path(args.output)
    if output.exists():
        raise ValueError("refusing to overwrite campaign config")
    write(output, build(spec))


if __name__ == "__main__":
    main()
