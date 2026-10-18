"""Stop a lease, recover disk files through host rsync, verify, then delete.

The caller supplies a pre-stop inventory of immutable files and an independent
hard deadline guard. Failed transfers never authorize early deletion.

Stopped instances support host-disk copy. Use the structured C.<instance_id>
module in both the API request and rsync source. "Unknown module" alone does
not establish that stopping removed disk access. An API success or a zero
Vast CLI exit code does not establish a completed transfer. This module uses the
native rsync exit status and verifies every inventoried hash. It never starts
or restarts an instance; any recovery requiring compute is the owner's explicit
budgeted decision. Keep the independent hard-stop active even if copying fails.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath

CHUNK = 4 * 1024**2
STDERR_LIMIT = 4096


def _check(deadline, expired):
    if time.time() >= deadline or (expired is not None and expired()):
        raise TimeoutError("backup budget or deadline reached")
    # The callback itself may take time.
    if time.time() >= deadline:
        raise TimeoutError("backup deadline reached")


def _paths(root, files):
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("unsafe backup root")
    root = root.resolve()
    seen = set()
    paths = []
    for row in files:
        name = row["path"]
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "\0" in name
            or str(relative) != name
            or name in seen
            or type(row["size"]) is not int
            or row["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("unsafe or invalid inventory")
        path = root.joinpath(*relative.parts)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("unsafe backup destination")
        seen.add(name)
        paths.append(path)
    return root, paths


def verify(root, files, *, deadline=float("inf"), expired=None):
    """Hash every expected file, checking the owner's budget between chunks."""
    _check(deadline, expired)
    root, paths = _paths(root, files)
    if not files:
        return False
    for row, path in zip(files, paths):
        _check(deadline, expired)
        if not path.is_file() or path.stat().st_size != row["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                _check(deadline, expired)
                chunk = stream.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
        _check(deadline, expired)
        if digest.hexdigest() != row["sha256"]:
            return False
    return True


def cleanup(
    *, stop, stopped, recover, destroy, expired, record, sleep=time.sleep, interval=15
):
    """Dependency-injected lifecycle; hard cap is the sole unverified-delete path."""
    while not expired():
        try:
            stop()
            if not stopped():
                record("waiting_for_stop", {})
                sleep(interval)
                continue
            record("stopped", {})
            if recover():
                record("backup_verified", {})
                destroy("backup_verified")
                return "deleted_verified"
            record("backup_pending", {})
        except Exception as error:
            record(
                "recovery_error",
                {"type": type(error).__name__, "detail": str(error)[:500]},
            )
        sleep(interval)
    record("backup_incomplete_at_limit", {})
    destroy("absolute_limit_backup_incomplete")
    return "deleted_at_limit"


def _host_endpoint(instance_id, remote_root, local_root, timeout):
    # Installed SDK: VastAI resolves VAST_API_KEY / existing SDK key files.
    # storage.copy selects /commands/rsync/ ONLY when one ID is None.
    from vastai.api import storage
    from vastai.sdk import VastAI

    client = VastAI(retry=1).client
    client.timeout = timeout
    response = storage.copy(
        client,
        src_id=f"C.{int(instance_id)}",
        dst_id=None,
        src_path=remote_root,
        dst_path=str(local_root),
    )
    if response.get("success") is not True:
        raise RuntimeError("Vast host rsync endpoint rejected request")
    host = response["src_addr"]
    port = int(response["src_port"])
    if (
        not isinstance(host, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", host)
        or host.startswith("-")
        or not 0 < port < 65536
    ):
        raise ValueError("invalid Vast rsync endpoint")
    return host, port


def _stop_transfer(process):
    # TERM lets rsync publish its partial file. KILL is only the fallback.
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def copy_verified(
    instance_id,
    remote_root,
    local_root,
    files,
    deadline,
    *,
    expired=None,
    record=None,
    attempt_seconds=120,
    ssh_identity=None,
    rsync_remote_config=None,
    compress=False,
):
    """Fetch exactly the inventory via host rsync, then hash before success.

    expired() returns true at the owner's budget limit (or may raise to abort).
    It is called during hashes and at most one second between rsync polls;
    callers may cache account reads. record(state, details) receives a bounded
    stderr tail on failures. The independent hard-stop remains mandatory.
    Existing positional callers remain supported; they enforce deadline only.
    A stopped host may reject access with "Unknown module"; failure is reported
    without an automatic restart or any other remote lifecycle action.
    rsync_remote_config optionally selects an already provisioned, caller-owned
    absolute <numeric-prefix>_C.<instance_id>_rsync.conf path on that host. It
    creates or changes no host configuration; the default command is unchanged.
    compress=True enables rsync wire compression (-z); stored bytes are unchanged.
    """

    def report(state, details):
        if record is not None:
            try:
                record(state, details)
            except Exception:
                pass  # A failed receipt must not prevent transfer termination.

    try:
        _check(deadline, expired)
        root, _ = _paths(local_root, files)
        remote = PurePosixPath(remote_root)
        if not remote.is_absolute() or ".." in remote.parts or "\0" in remote_root:
            raise ValueError("unsafe remote root")
        if int(instance_id) <= 0 or attempt_seconds <= 0:
            raise ValueError("invalid transfer limits or instance ID")
        if rsync_remote_config is not None:
            config = str(rsync_remote_config)
            config_path = PurePosixPath(config)
            if (
                not config_path.is_absolute()
                or config.startswith("//")
                or ".." in config_path.parts
                or str(config_path) != config
                or any(ord(char) < 32 or ord(char) == 127 for char in config)
                or not re.fullmatch(
                    rf"[0-9]+_C\.{int(instance_id)}_rsync\.conf", config_path.name
                )
            ):
                raise ValueError("unsafe or mismatched owned rsync config path")
        if not files:
            return False
        if verify(root, files, deadline=deadline, expired=expired):
            return True
        root.mkdir(parents=True, exist_ok=True)
        _check(deadline, expired)
        host, port = _host_endpoint(
            int(instance_id), str(remote) + "/", root, min(25, deadline - time.time())
        )
        _check(deadline, expired)
        # No recursive flag: --files-from lists only inventoried regular files.
        # --checksum also repairs same-size/same-mtime corrupt local files.
        with tempfile.TemporaryFile() as names:
            names.write(b"".join(os.fsencode(row["path"]) + b"\0" for row in files))
            names.seek(0)
            transport = [
                "ssh",
                "-p",
                str(port),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=2",
            ]
            if ssh_identity is not None:
                identity = Path(ssh_identity).expanduser().resolve()
                if not identity.is_file():
                    raise ValueError("SSH identity file missing")
                transport[1:1] = ["-i", str(identity), "-o", "IdentitiesOnly=yes"]
            command = [
                "rsync",
                "-dt",
                "--relative",
                "--partial",
                "--checksum",
                "--from0",
                "--files-from=-",
                "-e",
                shlex.join(transport),
                "--",
                f"vastai_kaalia@{host}::C.{int(instance_id)}/{str(remote)}/",
                str(root) + "/",
            ]
            if compress:
                command.insert(1, "-z")
            if rsync_remote_config is not None:
                command.insert(
                    1,
                    "--rsync-path="
                    + shlex.join(["sudo", "-n", "rsync", f"--config={config}"]),
                )
            process = subprocess.Popen(
                command,
                stdin=names,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            tail = bytearray()

            def drain():
                while chunk := process.stderr.read(1024):
                    tail.extend(chunk)
                    del tail[:-STDERR_LIMIT]

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            end = min(deadline, time.time() + attempt_seconds)
            try:
                while True:
                    _check(end, expired)
                    try:
                        returncode = process.wait(
                            timeout=max(0, min(1, end - time.time()))
                        )
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                _stop_transfer(process)
                reader.join(timeout=3)
                if not reader.is_alive():
                    process.stderr.close()
                report(
                    "rsync_result",
                    {
                        "returncode": process.returncode,
                        "stderr": bytes(tail).decode("utf-8", errors="replace"),
                    },
                )
            _check(deadline, expired)
            if returncode:
                return False
        valid = verify(root, files, deadline=deadline, expired=expired)
        if not valid:
            report("hash_mismatch", {})
        return valid
    except TimeoutError:
        report("backup_limit", {})
        return False
