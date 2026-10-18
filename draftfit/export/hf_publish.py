"""Immutable HF publication jobs, independent of training and checkpoint cleanup.

Call prepare_publication before releasing/replacing the selected best export,
then submit publish(job_dir) to a ThreadPoolExecutor or run this module's CLI.
Retry the same job directory after failure; it never reads the live export again.
Only payload/ is uploaded. Resume metadata and receipts stay local. Job directories
are retained after success/failure and may be removed explicitly by their owner.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time


MANIFEST = "draftfit-sha256.json"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class PublishInputError(ValueError):
    """Validation text safe to expose without remote exception details."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise PublishInputError("Export must be a regular directory.")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PublishInputError("Export symlinks are not supported.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublishInputError("Export contains a non-regular file.")
        name = path.relative_to(root).as_posix()
        if name in {MANIFEST, ".draftfit-hf-publish.json"}:
            continue
        result[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def prepare_publication(folder, repo_id, job_dir, run_id, *, private=True, resume_from=None):
    """Synchronously copy a completed export; return a durable job directory.

Use a new job_dir for every selected export. The source must remain stable until
this returns. A resume checkpoint is referenced and hashed, never uploaded or
deleted; the trainer remains responsible for retaining that checkpoint.
"""
    if not isinstance(repo_id, str) or not _REPO_ID.fullmatch(repo_id):
        raise PublishInputError("HF repo ID must have owner/name format.")
    if type(private) is not bool:
        raise PublishInputError("private must be a boolean.")
    source = Path(folder).absolute()
    files = _inventory(source)
    if "config.json" not in files or not any(
        "/" not in name and name.startswith(("model", "pytorch_model"))
        and Path(name).suffix in {".safetensors", ".bin"} for name in files
    ):
        raise PublishInputError("Export needs config.json and model weights.")
    source = source.resolve()
    destination = Path(job_dir).absolute()
    if destination.resolve().is_relative_to(source):
        raise PublishInputError("Publication job must be outside the source export.")
    if destination.exists():
        raise PublishInputError("Publication job already exists; retry it with publish.")
    resume = None
    if resume_from is not None:
        checkpoint = Path(resume_from).resolve(strict=True)
        resume = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hf-prepare-", dir=destination.parent))
    try:
        payload = staging / "payload"
        payload.mkdir()
        for name in files:
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        # Detect in-place writes, atomic replacements, additions and removals
        # during the copy. No hardlinks: later training writes must be isolated.
        if _inventory(payload) != files or _inventory(source) != files:
            raise PublishInputError("Export changed while creating the snapshot; prepare again.")
        manifest = {"format": "draftfit-sha256-v1", "files": files}
        _write(payload / MANIFEST, manifest)
        _write(staging / "publication.json", {
            "schema": "draftfit-hf-publication-v1", "repo_id": repo_id,
            "private": private, "run_id": str(run_id), "source": str(source),
            "resume": resume, "manifest_sha256": _sha256(payload / MANIFEST),
        })
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _model_artifact(name):
    # Unmanaged root weights/indexes can change Transformers' loader selection.
    return "/" not in name and (
        name.endswith((".safetensors", ".bin", ".safetensors.index.json", ".bin.index.json"))
        or name in {"tf_model.h5", "model.ckpt.index", "flax_model.msgpack", "adapter_config.json"}
    )


def _publication_operations(api, job, payload, parent):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    remote = set(api.list_repo_files(job["repo_id"], repo_type="model", revision=parent))
    current = set(json.loads((payload / MANIFEST).read_text())["files"])
    owned = set()
    if MANIFEST in remote:
        with tempfile.TemporaryDirectory(prefix="hf-previous-") as cache:
            downloaded = api.hf_hub_download(
                job["repo_id"], filename=MANIFEST, repo_type="model", revision=parent,
                cache_dir=cache, force_download=True,
            )
            try:
                previous = json.loads(Path(downloaded).read_text())
                if previous["format"] != "draftfit-sha256-v1" or not isinstance(previous["files"], dict):
                    raise ValueError
                for name, metadata in previous["files"].items():
                    if (not name or "\\" in name or name == MANIFEST
                            or any(part in {"", ".", "..", ".git"} for part in name.split("/"))
                            or type(metadata["bytes"]) is not int or metadata["bytes"] < 0
                            or not re.fullmatch(r"[a-f0-9]{64}", metadata["sha256"])):
                        raise ValueError
                owned = set(previous["files"])
            except (ValueError, KeyError, TypeError):
                raise PublishInputError("Previous publication manifest is invalid; refusing replacement.") from None
    unmanaged = remote - owned - {MANIFEST}
    if any(_model_artifact(name) for name in unmanaged):
        raise PublishInputError("Unmanaged model files exist; explicit migration is required.")
    # Refuse collisions rather than taking ownership of a user's model card or
    # any other existing content (including file/directory prefix collisions).
    if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
           for a in current | {MANIFEST} for b in unmanaged):
        raise PublishInputError("Publication would overwrite unmanaged repository files.")
    removed = sorted((owned & remote) - current)
    operations = [CommitOperationDelete(path_in_repo=name, is_folder=False) for name in removed]
    operations.extend(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(payload / name))
                      for name in sorted(current | {MANIFEST}))
    return operations, removed


def _verify(api, job, payload, commit, removed=()):
    manifest = json.loads((payload / MANIFEST).read_text())
    expected = dict(manifest["files"])
    expected[MANIFEST] = {"bytes": (payload / MANIFEST).stat().st_size,
                          "sha256": job["manifest_sha256"]}
    remote = set(api.list_repo_files(job["repo_id"], repo_type="model", revision=commit))
    if remote.intersection(removed) or any(_model_artifact(name) for name in remote - expected.keys()):
        raise RuntimeError("Remote commit retains conflicting or obsolete publication files.")
    rows = api.get_paths_info(job["repo_id"], paths=sorted(expected),
                              revision=commit, repo_type="model")
    actual = {row.path: row.size for row in rows if hasattr(row, "path")}
    if actual != {name: info["bytes"] for name, info in expected.items()}:
        raise RuntimeError("Remote commit file sizes differ.")
    # Size alone cannot detect corruption or equal-length weight replacement.
    # Download at the returned immutable commit, never at a moving branch.
    with tempfile.TemporaryDirectory(prefix="hf-verify-") as cache:
        for name, info in expected.items():
            downloaded = api.hf_hub_download(
                job["repo_id"], filename=name, repo_type="model", revision=commit,
                cache_dir=cache, force_download=True,
            )
            if _sha256(downloaded) != info["sha256"]:
                raise RuntimeError("Remote commit SHA256 differs.")
    return {"file_count": len(expected), "bytes": sum(v["bytes"] for v in expected.values())}


def publish(job_dir, *, token=None, api_factory=None, max_attempts=3, retry_delay=1.0):
    """Upload/verify one prepared job; safe to run in a background worker.

Token defaults to HF_TOKEN, then HUGGINGFACE_HUB_TOKEN. Exceptions and local
receipts never contain transport exception messages. A known commit is retained
on verification failure, so retry verifies it without uploading a second commit.
Concurrent publication of the same job fails immediately (POSIX flock).
Only files owned by the previous manifest may be replaced/deleted; unmanaged
collisions require explicit migration. Additions and removals share one commit.
The first observed HEAD is pinned across retries; a concurrent update requires
review and a new job, never an automatic rebase over someone else's changes.
"""
    if type(max_attempts) is not int or max_attempts < 1:
        raise PublishInputError("max_attempts must be a positive integer.")
    if not math.isfinite(retry_delay) or retry_delay < 0:
        raise PublishInputError("retry_delay must be finite and nonnegative.")
    directory = Path(job_dir)
    with (directory / ".publish.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PublishInputError("Publication is already running for this job.") from None
        job = json.loads((directory / "publication.json").read_text())
        payload = directory / "payload"
        manifest = payload / MANIFEST
        if _sha256(manifest) != job["manifest_sha256"] or _inventory(payload) != json.loads(manifest.read_text())["files"]:
            raise PublishInputError("Publication snapshot changed; refusing upload.")
        receipt_path = directory / "receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt = {"repo_id": job["repo_id"], "private": job["private"],
                   "run_id": job["run_id"], "manifest_sha256": job["manifest_sha256"],
                   "commit": previous.get("commit"), "attempts": previous.get("attempts", 0),
                   "parent_commit": previous.get("parent_commit"),
                   "removed_files": previous.get("removed_files", [])}
        credential = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not credential and api_factory is None:
            raise PublishInputError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
        for attempt in range(max_attempts):
            receipt.update(status="pending", attempts=receipt["attempts"] + 1)
            _write(receipt_path, receipt)
            try:
                if api_factory is None:
                    from huggingface_hub import HfApi
                    factory = HfApi
                else:
                    factory = api_factory
                api = factory(token=credential)
                api.create_repo(job["repo_id"], repo_type="model", private=job["private"], exist_ok=True)
                info = api.repo_info(job["repo_id"], repo_type="model")
                if getattr(info, "private", None) is not job["private"]:
                    raise PublishInputError("Existing repository visibility differs from the job.")
                if not receipt["commit"]:
                    head = getattr(info, "sha", None)
                    if not isinstance(head, str) or not re.fullmatch(r"[a-fA-F0-9]{40,64}", head):
                        raise PublishInputError("Repository HEAD is unavailable; refusing an unguarded commit.")
                    if receipt["parent_commit"] is not None and receipt["parent_commit"] != head:
                        raise PublishInputError("Repository HEAD changed; review the conflict and prepare a new job.")
                    receipt["parent_commit"] = head
                    _write(receipt_path, receipt)
                    operations, removed = _publication_operations(api, job, payload, head)
                    receipt["removed_files"] = removed
                    _write(receipt_path, receipt)
                    result = api.create_commit(
                        repo_id=job["repo_id"], repo_type="model", operations=operations,
                        commit_message="Draftfit immutable export", revision="main", parent_commit=head,
                    )
                    commit = getattr(result, "oid", None)
                    if not isinstance(commit, str) or not re.fullmatch(r"[a-fA-F0-9]{40,64}", commit):
                        raise RuntimeError("Upload returned no valid commit ID.")
                    receipt["commit"] = commit
                    _write(receipt_path, receipt)
                counts = _verify(api, job, payload, receipt["commit"], receipt["removed_files"])
                receipt.update(status="verified", finished_at=time.time(), **counts)
                receipt.pop("error_type", None)
                _write(receipt_path, receipt)
                return receipt
            except Exception as exc:
                receipt.update(status="failed", error_type=type(exc).__name__)
                _write(receipt_path, receipt)
                if isinstance(exc, PublishInputError):
                    raise
                if attempt + 1 < max_attempts:
                    time.sleep(retry_delay)
        raise RuntimeError("HF publication failed; retry the preserved job directory.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Prepared publication directory")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = publish(args.job_dir, max_attempts=args.max_attempts)
    except Exception:
        # Do not render arbitrary exception strings (including local path data).
        print("HF publication failed; inspect the local job receipt and retry.")
        return 1
    print(json.dumps({key: result[key] for key in ("status", "commit", "file_count", "bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
