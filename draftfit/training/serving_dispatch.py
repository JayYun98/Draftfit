"""Optional, fail-closed external serving boundary; native verification stays required.

No default engine command or model offloader is supplied. The caller owns a
supported ``memory_handoff`` context-manager factory: enter must quiesce capture
and release training GPU memory, exit must restore identities, optimizer, modes,
RNG and cursor even on failure. It yields a receipt with ``target_resident=False``
and ``mode='released_training_memory'`` (or ``'trainer_process_exited'``). A plain
assertion is not evidence of release: integration must test that implementation.
Default resident native verification must not be routed through this module.

A trusted explicit command receives ``--job-dir ATTEMPT``. Its request.json
contains independent strict-report expectations, source paths and native evidence;
it writes report.json using serving_report.SCHEMA. No shell, credentials in argv,
network threads, algorithm substitutions, or implicit GPU/API calls are used.
The caller independently binds runtime/target/tokenizer/data in ``expected``;
copying expectations from the worker report defeats validation. File inventories
bind exact checkpoint and all export bytes, plus caller-supplied other sources.

Admission is externally reserved, not calculated here. Each real attempt needs a
unique reservation_id and explicit absolute deadline/max_seconds. Failure remains
retryable only with a new admission. read_result revalidates a completed receipt
without launching or consuming admission. policy_update returns a detached state
proposal; caller atomically commits it with decisions/reservations before acting.
No native history is mixed into production history; old drafts stay ineligible.
Smoke never enables speed decisions. Nothing here claims resident-target SGLang
or makes the existing controller a production-serving controller by itself.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from draftfit.training.joint_policy import decide, fingerprint, observation_key
from draftfit.training.serving_report import normalize_serving_report


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _remaining(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("finite absolute serving deadline required")
    left = deadline - time.time()
    if left <= 0:
        raise TimeoutError("serving admission deadline reached")
    return left


def _sha(path, deadline):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _remaining(deadline)
            result.update(block)
    return result.hexdigest()


def _source(path, deadline):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        return dict(path=str(path), sha256=_sha(path, deadline))
    files = {
        p.relative_to(path).as_posix(): _sha(p, deadline)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }
    if not files:
        raise ValueError("empty serving source directory")
    return dict(path=str(path), files=files)


def _native(report, expected, checkpoint_sha):
    candidates = report.get("candidates", [])
    if (
        report.get("schema") != "draftfit-joint-generation-v1"
        or report.get("production_engine_speed") is not False
        or report.get("tokenparity") is not True
        or report.get("artifact_sha256") != checkpoint_sha
        or report.get("parity_failures")
        or len(candidates) != len(expected["artifacts"])
        or {c["name"] for c in candidates} != set(expected["artifacts"])
        or any(c.get("tokenparity") is not True for c in candidates)
        or any(
            c.get("tokenparity") is not True for c in report.get("existing_drafts", [])
        )
    ):
        raise ValueError(
            "current successful native verification required before external serving"
        )


def _check_sources(request, deadline):
    for value in request["sources"].values():
        if _source(value["path"], deadline) != value:
            raise ValueError("serving source bytes or file inventory changed")


def _normalize(raw, expected):
    # Smoke evidence never authorizes a speed policy, even if a worker asks for it.
    return normalize_serving_report(raw, **{**expected, "allow_speed_decisions": False})


# Private inherited-descriptor contract with the trusted SGLang supervisor.
_LOCK_FD_ENV = "DRAFTFIT_SERVING_LOCK_FD"
_CLEANUP_FD_ENV = "DRAFTFIT_SERVING_CLEANUP_FD"
_CLEANUP_SECONDS = 2.0


def _kill_group(process):
    """Kill and confirm all live members, including reparented grandchildren.

    Trusted serving children must stay in their stage process group. Zombies
    have released their resources and cannot execute or retain a flock.
    """
    import psutil

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1)
    end = time.monotonic() + 1
    while time.monotonic() < end:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    # Some container init processes leave orphan zombies unreaped. Those no
    # longer execute or hold descriptors; distinguish them from live members.
    for child in psutil.process_iter():
        try:
            if (
                os.getpgid(child.pid) == process.pid
                and child.status() != psutil.STATUS_ZOMBIE
            ):
                raise RuntimeError("serving process group termination unconfirmed")
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue


def _await_cleanup(fd):
    """Require every registered independent guardian to confirm cleanup and exit."""
    end = time.monotonic() + _CLEANUP_SECONDS + 1
    pending = 0
    while True:
        left = end - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], max(0, left))[0]:
            raise RuntimeError("serving guardian cleanup timed out")
        data = os.read(fd, 4096)
        if not data:
            if pending:
                raise RuntimeError(
                    "serving guardian exited without cleanup confirmation"
                )
            return
        for token in data:
            if token == ord("+"):
                pending += 1
            elif token == ord("-") and pending:
                pending -= 1
            else:
                raise RuntimeError("invalid serving guardian cleanup confirmation")


def _run(command, directory, deadline, lock_fd):
    allowance = _remaining(deadline) - (2 * _CLEANUP_SECONDS + 1)
    if allowance <= 0:
        raise TimeoutError("insufficient serving admission for bounded cleanup")
    read_fd, write_fd = os.pipe()
    process = None
    try:
        env = dict(
            os.environ, **{_LOCK_FD_ENV: str(lock_fd), _CLEANUP_FD_ENV: str(write_fd)}
        )
        process = subprocess.Popen(
            [*command, "--job-dir", str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=(lock_fd, write_fd),
            env=env,
        )
        os.close(write_fd)
        write_fd = None
        try:
            code = process.wait(timeout=allowance)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("external serving command deadline reached") from exc
        if code:
            raise RuntimeError(f"external serving command exited {code}")
    finally:
        try:
            if process is not None:
                try:
                    _kill_group(process)
                    _await_cleanup(read_fd)
                except BaseException:
                    # Unwinding memory_handoff would restore training beside an
                    # unconfirmed serving process. Fail-stop this worker instead.
                    try:
                        os.write(
                            2,
                            b"Serving cleanup unconfirmed; refusing training restore.\n",
                        )
                    finally:
                        os._exit(70)
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)


def dispatch(
    job_dir,
    *,
    command,
    expected,
    checkpoint,
    exports,
    native_report,
    memory_handoff,
    admission,
    additional_sources=None,
):
    """Run one admitted verifier attempt, normalize or raise; never change models.

    expected is the independently supplied normalize_serving_report kwargs.
    Every candidate must refer to the same authoritative joint checkpoint here.
    exports contains actual engine-loadable export directories, including config.
    additional_sources should cover target, tokenizer, validation data and engine
    code. Their semantic SHA/runtime expectations remain the caller's obligation.
    memory_handoff must be a bounded, already tested context-manager factory.
    Its release/restore operations themselves are caller-owned, not killable here.
    """
    if not callable(memory_handoff):
        raise ValueError(
            "supported explicit memory_handoff required; native stays resident"
        )
    if (
        not isinstance(command, (list, tuple))
        or not command
        or any(not isinstance(x, str) or not x or "\0" in x for x in command)
    ):
        raise ValueError("explicit trusted command argv required")
    if (
        admission.get("approved") is not True
        or not isinstance(admission.get("reservation_id"), str)
        or not admission["reservation_id"]
    ):
        raise ValueError("external serving admission/reservation required")
    for key in ("deadline_unix", "max_seconds"):
        value = admission.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("finite positive serving deadline/max_seconds required")
    deadline = min(admission["deadline_unix"], time.time() + admission["max_seconds"])
    _remaining(deadline)
    expected = deepcopy(expected)
    if expected.get("allow_speed_decisions", False) is not False:
        raise ValueError("dispatcher smoke cannot authorize speed decisions")
    if set(exports) != set(expected["artifacts"]):
        raise ValueError("exports must match all expected candidates")
    sources = {"checkpoint": _source(checkpoint, deadline)}
    checkpoint_sha = sources["checkpoint"]["sha256"]
    _native(native_report, expected, checkpoint_sha)
    for name, path in exports.items():
        source = _source(path, deadline)
        if "config.json" not in source.get("files", {}):
            raise ValueError("engine export requires config and full file inventory")
        artifact = expected["artifacts"][name]
        if (
            artifact["artifact_sha256"] != checkpoint_sha
            or artifact["checkpoint_sha256"] != checkpoint_sha
            or artifact["export_sha256"] != fingerprint(source["files"])
        ):
            raise ValueError("independent checkpoint/export binding mismatch")
        sources["export:" + name] = source
    for name, path in (additional_sources or {}).items():
        sources["source:" + name] = _source(path, deadline)
    native_binding = {
        k: deepcopy(native_report[k])
        for k in ("schema", "artifact_sha256", "production_engine_speed", "tokenparity")
    }
    native_binding["candidates"] = [
        dict(name=c["name"], tokenparity=c["tokenparity"])
        for c in native_report["candidates"]
    ]
    native_binding["existing_drafts"] = [
        dict(name=c["name"], tokenparity=c["tokenparity"])
        for c in native_report.get("existing_drafts", [])
    ]
    request = dict(
        schema="draftfit-serving-dispatch-v1",
        expected=expected,
        sources=sources,
        native_report=native_binding,
        target_resident=False,
    )
    root = Path(job_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("serving job already owned") from None
        manifest = root / "request.json"
        if manifest.exists() and _read(manifest) != request:
            raise ValueError("durable serving job expectations changed")
        _write(manifest, request)
        attempts = sorted(root.glob("attempt-*"))
        for attempt in [*attempts, *root.glob(".initializing-*")]:
            if not (attempt / "admission.json").exists():
                # Legacy/unpublished initialization never reached worker launch.
                continue
            prior = _read(attempt / "admission.json")
            if prior["reservation_id"] == admission["reservation_id"]:
                raise ValueError(
                    "reservation already consumed; retry needs fresh external admission"
                )
        directory = root / f"attempt-{len(attempts):06d}"
        # No visible attempt can exist without both durable launch inputs.
        temporary = Path(tempfile.mkdtemp(prefix=".initializing-", dir=root))
        _write(temporary / "admission.json", admission)
        _write(temporary / "request.json", request)
        temporary.rename(directory)
        receipt = dict(
            status="running",
            attempt=directory.name,
            request_sha256=fingerprint(request),
            target_resident=False,
            speed_decisions_allowed=False,
        )
        _write(root / "receipt.json", receipt)
        try:
            _remaining(deadline)
            with memory_handoff() as handoff:
                if (
                    not isinstance(handoff, dict)
                    or handoff.get("target_resident") is not False
                    or handoff.get("mode")
                    not in ("released_training_memory", "trainer_process_exited")
                ):
                    raise ValueError(
                        "memory handoff must explicitly release training target"
                    )
                receipt["memory_handoff"] = deepcopy(handoff)
                _run(command, directory, deadline, lock.fileno())
                raw = _read(directory / "report.json")
                result = _normalize(raw, expected)
                _check_sources(request, deadline)
            # Restore must succeed before verified status is durable/eligible.
            _check_sources(request, deadline)
            _remaining(deadline)
            receipt.update(status="verified", report_sha256=fingerprint(raw))
            _write(root / "receipt.json", receipt)
            return result
        except BaseException as exc:
            receipt.update(status="failed", error_type=type(exc).__name__)
            _write(root / "receipt.json", receipt)
            raise


def read_result(job_dir, *, expected, deadline_unix):
    """Revalidate a prior success against independent expectations; no subprocess."""
    root = Path(job_dir)
    request, receipt = _read(root / "request.json"), _read(root / "receipt.json")
    if (
        receipt["status"] != "verified"
        or receipt["request_sha256"] != fingerprint(request)
        or fingerprint(request["expected"]) != fingerprint(expected)
    ):
        raise ValueError("no matching verified serving receipt")
    _check_sources(request, deadline_unix)
    raw = _read(root / receipt["attempt"] / "report.json")
    if fingerprint(raw) != receipt["report_sha256"]:
        raise ValueError("serving report changed after verification")
    return _normalize(raw, expected)


def policy_update(normalized, *, policies, losses, histories, budget, tried_presets=()):
    """Pure proposal for separate production history; commit atomically in caller.

    Input must be dispatch/read_result output. Does not select publication winners,
    reserve money, execute a branch, change an optimizer or mutate caller state.
    Native loss-only/history decisions remain outside this optional boundary.
    """
    rows = normalized["candidates"]
    names = {r["name"] for r in rows}
    if names != set(policies) or names != set(losses) or set(histories) - names:
        raise ValueError(
            "policy/loss/history candidates do not match verified candidates"
        )
    if normalized["speed_decisions_allowed"] is not False:
        raise ValueError("smoke dispatcher does not authorize speed policy")
    updated, decisions = deepcopy(histories), {}
    for serving in rows:
        name = serving["name"]
        current = {
            k: serving[k]
            for k in (
                "artifact_sha256",
                "family",
                "preset",
                "step",
                "training_fingerprint",
                "objective_fingerprint",
                "capture_fingerprint",
                "checkpoint_sha256",
                "export_sha256",
            )
        }
        observation = dict(
            artifact=current, loss=losses[name], serving=deepcopy(serving)
        )
        history = updated.setdefault(name, [])
        for row in history:
            if row.get("serving", {}).get("production_engine_speed") is not True:
                raise ValueError(
                    "separate production history required; native history cannot be relabeled"
                )
        matches = [
            r
            for r in history
            if observation_key(r["artifact"]) == observation_key(current)
        ]
        if matches:
            if len(matches) != 1 or fingerprint(matches[0]) != fingerprint(observation):
                raise ValueError("conflicting production observation replay")
        else:
            history.append(observation)
        cfg = deepcopy(policies[name])
        cfg.update(
            comparison_fingerprint=normalized["comparison_fingerprint"],
            require_serving=True,
        )
        decision = decide(
            cfg, history, current=current, budget=budget, tried_presets=tried_presets
        )
        if not decision["current_serving_validated"]:
            raise ValueError(
                "policy did not validate current production evidence: "
                + str(decision["reason"])
            )
        decisions[name] = decision
    return dict(
        history=updated,
        decisions=decisions,
        existing_draft=deepcopy(normalized.get("existing_draft")),
        source_report_sha256=normalized["source_report_sha256"],
        speed_decisions_allowed=False,
    )
