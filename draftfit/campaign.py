"""Deterministic, restartable local campaign scheduler. No cloud mutations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

TERMINAL = {"complete", "failed", "skipped", "stopped", "interrupted"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def number(x, *, positive=False):
    if (
        isinstance(x, bool)
        or not isinstance(x, (float, int))
        or not math.isfinite(x)
        or x < 0
        or (positive and x == 0)
    ):
        raise ValueError(
            "expected a finite positive number"
            if positive
            else "expected a finite nonnegative number"
        )
    return x


def validate(config):
    if config.get("schema") != "draftfit-campaign-v1":
        raise ValueError("unsupported campaign schema")
    if type(config.get("fail_fast", False)) is not bool:
        raise ValueError("fail_fast must be boolean")
    budget = config["budget"]
    for key in ("max_hours", "max_usd"):
        number(budget[key], positive=True)
    for key in ("hourly_usd", "fixed_usd", "reserve_usd"):
        number(budget[key])
    if budget["fixed_usd"] + budget["reserve_usd"] >= budget["max_usd"]:
        raise ValueError("no budget remains after fixed charges and reserve")
    resources = config["resources"]
    if not resources or any(type(v) is not int or v < 1 for v in resources.values()):
        raise ValueError("resources require positive integer capacities")
    seen = set()
    for job in config["jobs"]:
        name = job["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name)
            or name in seen
        ):
            raise ValueError("invalid/duplicate job id")
        if (
            job["kind"] not in ("train", "evaluate", "prepare")
            or job["resource"] not in resources
        ):
            raise ValueError("unknown kind/resource")
        if (
            not isinstance(job["argv"], list)
            or not job["argv"]
            or any(not isinstance(x, str) for x in job["argv"])
        ):
            raise ValueError("argv must be a nonempty string array; no shell")
        deps = job.get("depends_on", [])
        if len(set(deps)) != len(deps) or any(x not in seen for x in deps):
            raise ValueError("dependencies must be unique earlier jobs")
        number(job["timeout_seconds"], positive=True)
        if job["kind"] == "evaluate":
            if job["train_job"] not in deps or not job.get("comparison_id"):
                raise ValueError(
                    "evaluation must depend on a training job and bind comparison_id"
                )
            if (
                type(job.get("min_requests", 32)) is not int
                or job.get("min_requests", 32) < 1
            ):
                raise ValueError("min_requests must be positive integer")
        if "gate" in job:
            gate = job["gate"]
            if not gate.get("evaluations") or any(
                x not in deps for x in gate["evaluations"]
            ):
                raise ValueError("gate evaluations must be direct dependencies")
            for key in ("min_speedup", "min_domain_speedup", "min_relative_gain"):
                number(gate[key])
            if type(gate["top_k"]) is not int or gate["top_k"] < 1:
                raise ValueError("top_k must be positive")
        seen.add(name)
    if not seen:
        raise ValueError("empty campaign")


def evaluate_report(job, report, state, root):
    """Trusted benchmark adapters supply measured aggregates, never loss proxies."""
    training = root / job["train_job"] / "result.json"
    if (
        report.get("schema") != "draftfit-campaign-evaluation-v1"
        or report.get("comparison_id") != job["comparison_id"]
        or not report.get("comparison_fingerprint")
        or report.get("split") != "validation"
        or report.get("training_result_sha256") != digest(training)
    ):
        raise ValueError("evaluation split/comparison/checkpoint binding mismatch")
    candidates = report["candidates"]
    receipt = read(training)
    if "arms" in receipt and {a["name"] for a in receipt["arms"]} != {
        c["name"] for c in candidates
    }:
        raise ValueError("evaluation omitted or substituted a trained family")
    if not candidates or len({x["name"] for x in candidates}) != len(candidates):
        raise ValueError("empty/duplicate candidates")
    normalized = []
    for candidate in candidates:
        domains = candidate["domains"]
        if (
            candidate.get("exact_token_parity") is not True
            or not domains
            or len({x["name"] for x in domains}) != len(domains)
        ):
            raise ValueError("parity failure or missing/duplicate domains")
        total_tokens = total_seconds = base_seconds = requests = 0
        ratios = {}
        for domain in domains:
            n = domain["requests"]
            if type(n) is not int or n < 1:
                raise ValueError("invalid request count")
            if domain.get("matched_requests") != n:
                raise ValueError("unmatched requests")
            tokens = number(domain["output_tokens"], positive=True)
            if tokens != domain["baseline_output_tokens"]:
                raise ValueError("candidate/baseline token counts differ")
            seconds = number(domain["seconds"], positive=True)
            baseline = number(domain["baseline_seconds"], positive=True)
            ratios[domain["name"]] = baseline / seconds
            requests += n
            total_tokens += tokens
            total_seconds += seconds
            base_seconds += baseline
        if requests < job.get("min_requests", 32):
            raise ValueError("insufficient validation requests")
        normalized.append(
            dict(
                name=candidate["name"],
                speedup=base_seconds / total_seconds,
                tok_s=total_tokens / total_seconds,
                domains=ratios,
            )
        )
    return dict(
        comparison_id=report["comparison_id"],
        comparison_fingerprint=report.get("comparison_fingerprint"),
        candidates=normalized,
        training_result_sha256=report["training_result_sha256"],
    )


def decide(gate, state):
    reports = [state["jobs"][key]["evaluation"] for key in gate["evaluations"]]
    if (
        len({(r["comparison_id"], r.get("comparison_fingerprint")) for r in reports})
        != 1
    ):
        raise ValueError("cannot compare different validation workloads")
    current = sorted(
        reports[-1]["candidates"], key=lambda c: (-c["speedup"], c["name"])
    )
    candidates = [
        c
        for c in current
        if c["speedup"] >= gate["min_speedup"]
        and min(c["domains"].values()) >= gate["min_domain_speedup"]
    ]
    selected = candidates[: gate["top_k"]]
    family = gate.get("family")
    if family:
        selected = [c for c in selected if c["name"] == family]
    observed = dict(
        candidates=current, selected=[c["name"] for c in selected], thresholds=gate
    )
    if not selected:
        return False, "below_speed_or_domain_threshold_or_not_top_k", observed
    # Patience is the number of consecutive evaluation transitions supplied.
    if len(reports) > 1:
        gains = []
        for previous, latest in zip(reports, reports[1:]):
            old = {c["name"]: c["speedup"] for c in previous["candidates"]}
            new = {c["name"]: c["speedup"] for c in latest["candidates"]}
            names = [c["name"] for c in selected]
            if any(n not in old or n not in new for n in names):
                raise ValueError("candidate missing from plateau history")
            gains.append(max(new[n] / old[n] - 1 for n in names))
        observed["relative_gains"] = gains
        if all(g < gate["min_relative_gain"] for g in gains):
            return False, "plateau", observed
    return True, "continue", observed


def progress(folder):
    latest = {}
    for path in folder.glob("attempt-*/*/metrics.jsonl"):
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 32768))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                    if "step" in row:
                        latest[path.parent.name] = row
                        break
                except (ValueError, UnicodeDecodeError):
                    continue
        except OSError:
            continue
    return latest


def dashboard(root, state):
    rows = []
    for name, job in state["jobs"].items():
        metrics = job.get("evaluation", {}).get("candidates", [])
        speeds = ", ".join(f"{m['name']}: {m['speedup']:.3f}×" for m in metrics)
        steps = ", ".join(
            f"{n}: step {r['step']}" for n, r in job.get("progress", {}).items()
        )
        cells = [name, job["status"], steps, job.get("reason", ""), speeds]
        rows.append(
            "<tr>"
            + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells)
            + f'<td><a href="{name}/stdout.log">log</a></td></tr>'
        )
    content = (
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
        "<title>Draftfit campaign</title><style>body{font:15px system-ui;margin:32px;background:#f6f8fb}"
        "table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}</style>"
        "<h1>"
        + html.escape(state.get("label", "Draftfit campaign"))
        + "</h1><p>Status: "
        + html.escape(state["status"])
        + f" · Estimated cost ${state.get('estimated_usd', 0):.3f} · Refreshes every 5 seconds</p>"
        + '<p><a href="state.json">State and decision details</a> · <a href="events.jsonl">Phase history</a></p>'
        + "<table><tr><th>Job</th><th>Status</th><th>Training progress</th><th>Decision reason</th><th>Verified speed / target</th><th>Logs</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    temp = root / "index.html.tmp"
    temp.write_text(content)
    temp.replace(root / "index.html")


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Also remove descendants if the leader exited first.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(config_path, root, *, retry_interrupted=False):
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    config = read(config_path)
    validate(config)
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "state.json"
        state = (
            read(state_path)
            if state_path.exists()
            else dict(
                schema="draftfit-campaign-state-v1",
                config_sha256=identity,
                started_at=time.time(),
                status="running",
                label=config.get("label", "Draftfit campaign"),
                jobs={
                    j["id"]: {"status": "pending", "attempts": 0}
                    for j in config["jobs"]
                },
            )
        )
        if state["config_sha256"] != identity:
            raise ValueError("campaign config changed; use a new output directory")
        for name, job in state["jobs"].items():
            if (
                job["status"] == "complete"
                and digest(root / name / "result.json") != job["result_sha256"]
            ):
                raise ValueError(
                    "completed result was modified; refusing stale decisions"
                )
            if job["status"] == "running":
                job.update(status="interrupted", reason="controller_interrupted")
            if retry_interrupted and job["status"] == "interrupted":
                job.update(status="pending", reason="explicit_retry")
        processes = {}
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True

        old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}

        def event(name, **data):
            row = dict(timestamp=time.time(), job=name, **data)
            with (root / "events.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

        def persist():
            for name in processes:
                state["jobs"][name]["progress"] = progress(root / name)
            write(state_path, state)
            dashboard(root, state)

        state["status"] = "running"
        try:
            while True:
                now = time.time()
                b = config["budget"]
                hours = max(0, now - state["started_at"]) / 3600
                state["estimated_usd"] = b["fixed_usd"] + b["hourly_usd"] * hours
                exhausted = (
                    hours >= b["max_hours"]
                    or state["estimated_usd"] + b["reserve_usd"] >= b["max_usd"]
                )
                stopping = stopped or exhausted or (root / "STOP").exists()
                for name, (process, stream, began) in list(processes.items()):
                    spec = next(j for j in config["jobs"] if j["id"] == name)
                    timed_out = time.monotonic() - began >= spec["timeout_seconds"]
                    if stopping or timed_out:
                        terminate(process)
                    code = process.poll()
                    if code is None:
                        continue
                    terminate(process)
                    stream.close()
                    job = state["jobs"][name]
                    job.update(
                        exit_code=code, finished_at=now, progress=progress(root / name)
                    )
                    if stopping or timed_out:
                        job.update(
                            status="interrupted" if stopped else "stopped",
                            reason="budget_or_stop" if stopping else "timeout",
                        )
                    elif code != 0:
                        job.update(status="failed", reason=f"exit_{code}")
                    else:
                        try:
                            result_path = root / name / "result.json"
                            result = read(result_path)
                            if result.get("status") != "complete":
                                raise ValueError("worker did not confirm completion")
                            if spec["kind"] == "evaluate":
                                job["evaluation"] = evaluate_report(
                                    spec, result, state, root
                                )
                            job.update(
                                status="complete",
                                reason="worker_result_valid",
                                result_sha256=digest(result_path),
                            )
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            job.update(status="failed", reason=f"invalid_result: {exc}")
                    event(name, status=job["status"], reason=job["reason"])
                    del processes[name]
                    persist()
                failed = config.get("fail_fast", False) and any(
                    j["status"] in {"failed", "stopped", "interrupted"}
                    for j in state["jobs"].values()
                )
                if failed and not stopping:
                    for name, (process, stream, _) in list(processes.items()):
                        terminate(process)
                        stream.close()
                        state["jobs"][name].update(
                            status="stopped",
                            reason="peer_failed",
                            exit_code=process.returncode,
                            finished_at=time.time(),
                        )
                        event(name, status="stopped", reason="peer_failed")
                        del processes[name]
                    for name, job in state["jobs"].items():
                        if job["status"] == "pending":
                            job.update(status="stopped", reason="peer_failed")
                            event(name, status="stopped", reason="peer_failed")
                    state["status"] = "failed"
                    persist()
                    return 1
                if stopping:
                    for name, job in state["jobs"].items():
                        if job["status"] == "pending" and not stopped:
                            job.update(status="stopped", reason="budget_or_stop")
                            event(name, status="stopped", reason=job["reason"])
                    state["status"] = "interrupted" if stopped else "stopped"
                    persist()
                    return 2
                for spec in config["jobs"]:
                    name = spec["id"]
                    job = state["jobs"][name]
                    if config.get("fail_fast", False) and any(
                        j["status"] == "failed" for j in state["jobs"].values()
                    ):
                        break
                    if job["status"] != "pending":
                        continue
                    deps = [state["jobs"][d] for d in spec.get("depends_on", [])]
                    if any(d["status"] in TERMINAL - {"complete"} for d in deps):
                        job.update(status="skipped", reason="dependency_not_complete")
                        event(name, status="skipped", reason=job["reason"])
                        continue
                    if any(d["status"] != "complete" for d in deps):
                        continue
                    if "gate" in spec:
                        try:
                            allow, reason, decision = decide(spec["gate"], state)
                        except (KeyError, ValueError, TypeError) as exc:
                            allow, reason, decision = False, f"invalid_gate: {exc}", {}
                        job["decision"] = decision
                        if not allow:
                            status = (
                                "failed"
                                if reason.startswith("invalid_gate:")
                                else "skipped"
                            )
                            job.update(status=status, reason=reason)
                            event(name, status=status, reason=reason, decision=decision)
                            continue
                    busy = sum(
                        1
                        for n in processes
                        if next(j for j in config["jobs"] if j["id"] == n)["resource"]
                        == spec["resource"]
                    )
                    if busy >= config["resources"][spec["resource"]]:
                        continue
                    folder = root / name
                    folder.mkdir(exist_ok=True)
                    # A retry cannot accidentally consume the previous attempt's receipt.
                    result = folder / "result.json"
                    if result.exists():
                        result.rename(folder / f"result-attempt-{job['attempts']}.json")
                    env = os.environ.copy()
                    env.update(
                        DRAFTFIT_JOB_OUTPUT=str(folder),
                        DRAFTFIT_JOB_RESULT=str(result),
                        DRAFTFIT_JOB_ATTEMPT=str(job["attempts"] + 1),
                        DRAFTFIT_INPUT_RESULTS=json.dumps(
                            {
                                d: str(root / d / "result.json")
                                for d in spec.get("depends_on", [])
                            }
                        ),
                    )
                    stream = (folder / "stdout.log").open("a")
                    job.update(
                        status="running",
                        reason="dependencies_and_gate_passed",
                        started_at=time.time(),
                        attempts=job["attempts"] + 1,
                    )
                    persist()
                    try:
                        process = subprocess.Popen(
                            spec["argv"],
                            cwd=config_path.parent,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(lock.fileno(),),
                        )
                    except OSError as exc:
                        stream.close()
                        job.update(status="failed", reason=f"spawn_failed: {exc}")
                        event(name, status="failed", reason=job["reason"])
                        continue
                    processes[name] = (process, stream, time.monotonic())
                    job["pid"] = process.pid
                    event(name, status="running", decision=job.get("decision"))
                persist()
                if all(j["status"] in TERMINAL for j in state["jobs"].values()):
                    state["status"] = (
                        "failed"
                        if any(
                            j["status"] in {"failed", "interrupted", "stopped"}
                            for j in state["jobs"].values()
                        )
                        else "complete"
                    )
                    persist()
                    return 0 if state["status"] == "complete" else 1
                time.sleep(0.2)
        finally:
            for process, stream, _ in processes.values():
                terminate(process)
                stream.close()
            for sig, handler in old.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("run")
    start.add_argument("--config", required=True)
    start.add_argument("--output", required=True)
    start.add_argument("--retry-interrupted", action="store_true")
    check = sub.add_parser("plan")
    check.add_argument("--config", required=True)
    status = sub.add_parser("status")
    status.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "run":
        raise SystemExit(
            run(args.config, args.output, retry_interrupted=args.retry_interrupted)
        )
    if args.action == "plan":
        config = read(args.config)
        validate(config)
        print(json.dumps(config, indent=2))
    else:
        print(json.dumps(read(Path(args.output) / "state.json"), indent=2))


if __name__ == "__main__":
    main()
