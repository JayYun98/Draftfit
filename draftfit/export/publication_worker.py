"""Bounded POSIX subprocess owner for already prepared, durable HF jobs.

with PublicationWorker() as worker:
    worker.start(job_dir)  # prepared by hf_publish.prepare_publication
    ...                   # training may continue; poll() never waits for upload
    outcome = worker.wait(deadline_unix=time.time() + remaining_seconds)

One child at a time per owner. start/poll/wait/close are owner-thread APIs.
The default command is this interpreter -m draftfit.export.hf_publish --job-dir
PATH. ``command`` replaces the executable prefix for local process tests; the
same --job-dir PATH arguments are appended. Never place credentials in command
arguments. Credentials are inherited through the environment, never serialized.
All child stdio goes to DEVNULL, so network/library output cannot leak tokens or
fill a pipe. The only journal is publication-worker.json (no command/env/output).

wait converts the explicit Unix deadline to a monotonic timeout. Expiry/close
terminates the child's process group, then kills it after bounded grace. Terminal
poll also kills leftover descendants. No network thread or unbounded join is
created. Normal cleanup may exceed the deadline by terminate_timeout+kill_timeout.
OS-level unkillable processes return termination_unconfirmed, block new starts,
and do not prevent controller exit. Context-manager/atexit cleanup is provided.

Jobs, payloads and hf_publish's receipt.json are never overwritten or removed.
Killed/failed workers can retry the same directory. An inherited flock prevents
another wrapper using that job while its worker survives a controller crash;
old journal PIDs are NEVER used to signal a process. Process exit zero means
worker success, not an independently verified HF artifact: the caller still
consumes hf_publish's verified receipt. The caller owns retry policy/admission.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("expected finite nonnegative seconds")
    return value


class PublicationWorker:
    def __init__(self, *, command=None, terminate_timeout=0.25, kill_timeout=1.0):
        self.terminate_timeout = _seconds(terminate_timeout)
        self.kill_timeout = _seconds(kill_timeout)
        if command is not None and (
            not isinstance(command, (list, tuple))
            or not command
            or any(
                not isinstance(arg, str) or not arg or "\0" in arg for arg in command
            )
        ):
            raise ValueError("command must be a nonempty argv sequence")
        self._command = (
            tuple(command)
            if command is not None
            else (sys.executable, "-m", "draftfit.export.hf_publish")
        )
        self._process = None
        self._lock = None
        self._directory = None
        self._closed = False
        self._state = dict(status="idle", retryable=False, returncode=None)
        atexit.register(self.close)

    def _save(self):
        """Atomic diagnostic journal; disk failure never bypasses process cleanup."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._directory,
                prefix=".publication-worker-",
                delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(self._state, stream, allow_nan=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._directory / "publication-worker.json")
            return True
        except OSError as exc:
            self._state["journal_error"] = type(exc).__name__
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _release(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _signal_group(self, sig):
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            pass

    def _finish(self, status):
        # A successful/failed parent may have left children in its session.
        try:
            self._signal_group(signal.SIGKILL)
        except OSError as exc:
            self._state.update(
                status="termination_unconfirmed",
                retryable=False,
                error_type=type(exc).__name__,
            )
            self._save()
            return deepcopy(self._state)
        self._state.update(
            status=status,
            retryable=status != "succeeded",
            returncode=self._process.returncode,
            finished_at=time.time(),
        )
        self._process = None
        self._save()
        self._release()
        return deepcopy(self._state)

    def start(self, job_dir):
        """Launch one existing prepared job; return running or spawn-failure state."""
        if self._closed:
            raise RuntimeError("worker is closed")
        if self._process is not None:
            self.poll()
            if self._process is not None:
                raise RuntimeError("publication worker already active")
        directory = Path(job_dir).resolve()
        if (
            not (directory / "publication.json").is_file()
            or not (directory / "payload").is_dir()
        ):
            raise ValueError("prepared durable publication job required")
        lock = (directory / ".publication-worker.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError("publication job already owned") from None
        self._directory, self._lock = directory, lock
        self._state = dict(
            status="starting", retryable=False, returncode=None, started_at=time.time()
        )
        if not self._save():
            self._state.update(status="failed", retryable=True)
            self._release()
            return deepcopy(self._state)
        try:
            self._process = subprocess.Popen(
                [*self._command, "--job-dir", str(directory)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(lock.fileno(),),
                # No env override: preserve process-provided credentials exactly.
            )
        except (OSError, ValueError) as exc:
            self._state.update(
                status="failed",
                retryable=True,
                error_type=type(exc).__name__,
                finished_at=time.time(),
            )
            self._save()
            self._release()
            return deepcopy(self._state)
        self._state.update(status="running", pid=self._process.pid)
        if not self._save():
            return self._terminate("failed")
        return deepcopy(self._state)

    def poll(self):
        """Nonblocking process status. It does not interpret publication receipts."""
        if self._process is not None and self._process.poll() is not None:
            status = self._state.get("termination_reason")
            return self._finish(
                status or ("succeeded" if self._process.returncode == 0 else "failed")
            )
        return deepcopy(self._state)

    def _terminate(self, reason):
        self._state["termination_reason"] = reason
        try:
            self._signal_group(signal.SIGTERM)
            try:
                self._process.wait(timeout=self.terminate_timeout)
            except subprocess.TimeoutExpired:
                pass
            self._signal_group(signal.SIGKILL)
            self._process.wait(timeout=self.kill_timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._state.update(
                status="termination_unconfirmed",
                retryable=False,
                error_type=type(exc).__name__,
            )
            self._save()
            return deepcopy(self._state)
        return self._finish(reason)

    def wait(self, deadline_unix):
        """Wait until completion or Unix deadline, then boundedly terminate."""
        deadline_unix = _seconds(deadline_unix)
        self.poll()
        if self._process is None:
            return deepcopy(self._state)
        # Popen.wait implements its timeout with a monotonic clock on POSIX.
        remaining = max(0.0, deadline_unix - time.time())
        try:
            self._process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return self._terminate("timed_out")
        return self.poll()

    def close(self):
        """Idempotent bounded cleanup; preserves job files for a new owner."""
        self._closed = True
        self.poll()
        if self._process is not None:
            result = self._terminate("cancelled")
        else:
            result = deepcopy(self._state)
        if self._process is None:
            atexit.unregister(self.close)
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
