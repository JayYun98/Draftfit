"""Real files/workers; only the Hub transport is replaced with a disk-backed fake."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from draftfit.export.hf_publish import (
    MANIFEST,
    PublishInputError,
    prepare_publication,
    publish,
)


class DiskHub:
    def __init__(self, root):
        self.root = root
        self.uploads = 0
        self.download_revisions = []
        self.upload_failures = 0
        self.verify_failures = 0
        self.corrupt = False
        self.private = True
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.history = root.with_name(root.name + "-history")
        self.head_file = self.history / "HEAD"
        self.lock = threading.Lock()
        self.barrier = None
        self.commits = []

    @property
    def commit(self):
        return self.head_file.read_text()

    def _record(self, tree, parent):
        contents = inventory(tree)
        digest = hashlib.sha1(
            json.dumps(
                [
                    parent,
                    [
                        (name, hashlib.sha256(data).hexdigest())
                        for name, data in sorted(contents.items())
                    ],
                ]
            ).encode()
        ).hexdigest()
        shutil.copytree(tree, self.history / digest)
        self.head_file.write_text(digest)
        return digest

    def __call__(self, *, token):
        return self

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        assert repo_id == "tester/draft" and repo_type == "model" and exist_ok
        with self.lock:
            if not self.head_file.exists():
                self.root.mkdir(parents=True, exist_ok=True)
                self.history.mkdir()
                self._record(self.root, None)

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private, sha=self.commit)

    def list_repo_files(self, repo_id, *, revision, repo_type):
        assert repo_id == "tester/draft" and repo_type == "model"
        return sorted(inventory(self.history / revision))

    def create_commit(
        self, *, repo_id, repo_type, operations, commit_message, revision, parent_commit
    ):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        assert repo_id == "tester/draft" and repo_type == "model" and revision == "main"
        self.uploads += 1
        self.entered.set()
        assert self.release.wait(10), "test worker was not released"
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.upload_failures:
            self.upload_failures -= 1
            raise RuntimeError("Authorization: Bearer synthetic-secret-value")
        with self.lock:
            if parent_commit != self.commit:
                raise RuntimeError("409 parent commit mismatch")
            with tempfile.TemporaryDirectory(dir=self.history) as staging:
                tree = Path(staging) / "tree"
                shutil.copytree(self.history / parent_commit, tree)
                deleted, added = [], []
                for operation in operations:
                    target = tree / operation.path_in_repo
                    if isinstance(operation, CommitOperationDelete):
                        assert operation.is_folder is False
                        target.unlink()
                        deleted.append(operation.path_in_repo)
                    else:
                        assert isinstance(operation, CommitOperationAdd)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(operation.path_or_fileobj, target)
                        added.append(operation.path_in_repo)
                if self.corrupt:
                    path = tree / "model.safetensors"
                    data = path.read_bytes()
                    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                commit = self._record(tree, parent_commit)
                shutil.rmtree(self.root)
                shutil.copytree(tree, self.root)
                self.commits.append(
                    {
                        "commit": commit,
                        "parent": parent_commit,
                        "deleted": deleted,
                        "added": added,
                    }
                )
        return SimpleNamespace(oid=commit)

    def get_paths_info(self, repo_id, *, paths, revision, repo_type):
        assert (self.history / revision).is_dir() and repo_type == "model"
        if self.verify_failures:
            self.verify_failures -= 1
            raise RuntimeError("Authorization: Bearer synthetic-secret-value")
        return [
            SimpleNamespace(path=p, size=(self.history / revision / p).stat().st_size)
            for p in paths
            if (self.history / revision / p).is_file()
        ]

    def hf_hub_download(
        self, repo_id, *, filename, revision, repo_type, cache_dir, force_download
    ):
        assert repo_type == "model" and force_download
        self.download_revisions.append(revision)
        destination = Path(cache_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.history / revision / filename, destination)
        return str(destination)


@pytest.fixture
def case(tmp_path):
    export = tmp_path / "best-export"
    export.mkdir()
    (export / "config.json").write_text('{"model_type":"draft"}')
    (export / "model.safetensors").write_bytes(b"original-best-weights")
    (export / "assets").mkdir()
    (export / "assets/tokenizer.json").write_text("{}")
    checkpoint = tmp_path / "joint-step5.pt"
    checkpoint.write_bytes(b"resumable-optimizer-and-rng-state")
    (tmp_path / "resume.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
    return SimpleNamespace(
        root=tmp_path,
        export=export,
        checkpoint=checkpoint,
        job=tmp_path / "publication",
        hub=DiskHub(tmp_path / "fake-hub"),
    )


def prepare(case):
    return prepare_publication(
        case.export, "tester/draft", case.job, "run-1", resume_from=case.checkpoint
    )


def inventory(folder):
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in folder.rglob("*")
        if p.is_file()
    }


def test_background_upload_uses_snapshot_after_best_export_replacement(case):
    expected = inventory(case.export)
    job = prepare(case)
    case.hub.release.clear()
    with ThreadPoolExecutor(1) as worker:
        future = worker.submit(publish, job, api_factory=case.hub, retry_delay=0)
        try:
            assert case.hub.entered.wait(10)
            assert not future.done()
            (case.export / "model.safetensors").write_bytes(b"replaced-best-weight")
            shutil.rmtree(case.export)
            case.export.mkdir()
            (case.export / "model.safetensors").write_bytes(b"another-generation")
        finally:
            case.hub.release.set()
        result = future.result(10)
    assert result["status"] == "verified"
    assert {
        k: v for k, v in inventory(case.hub.root).items() if k != MANIFEST
    } == expected
    assert set(case.hub.download_revisions) == {case.hub.commit}
    manifest = json.loads((case.hub.root / MANIFEST).read_text())
    for name, data in expected.items():
        assert manifest["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
    assert case.checkpoint.read_bytes() == b"resumable-optimizer-and-rng-state"


def test_failed_job_retries_original_snapshot_without_source_and_preserves_resume(
    case, capsys
):
    before = inventory(case.root)
    prepare(case)
    case.hub.upload_failures = 1
    with ThreadPoolExecutor(1) as worker:
        future = worker.submit(publish, case.job, api_factory=case.hub, max_attempts=1)
        with pytest.raises(RuntimeError, match="retry the preserved") as error:
            future.result(10)
    assert "synthetic-secret" not in str(error.value)
    failed = json.loads((case.job / "receipt.json").read_text())
    assert failed["status"] == "failed" and failed["commit"] is None
    for name, data in before.items():
        assert (case.root / name).read_bytes() == data
    metadata = json.loads((case.job / "publication.json").read_text())
    assert metadata["resume"] == {
        "path": str(case.checkpoint),
        "sha256": hashlib.sha256(case.checkpoint.read_bytes()).hexdigest(),
    }
    shutil.rmtree(case.export)
    with ThreadPoolExecutor(1) as worker:
        result = worker.submit(publish, case.job, api_factory=case.hub).result(10)
    assert result["status"] == "verified" and result["attempts"] == 2
    assert case.hub.uploads == 2
    assert case.checkpoint.read_bytes() == before[case.checkpoint.name]
    assert (case.root / "resume.json").read_bytes() == before["resume.json"]
    assert "resume.json" not in inventory(case.hub.root)
    assert "publication.json" not in inventory(case.hub.root)
    assert "synthetic-secret" not in (case.job / "receipt.json").read_text()
    assert "synthetic-secret" not in str(capsys.readouterr())


@pytest.mark.parametrize("failure", ["upload_failures", "verify_failures"])
def test_automatic_bounded_retry(case, failure):
    prepare(case)
    setattr(case.hub, failure, 1)
    with ThreadPoolExecutor(1) as worker:
        result = worker.submit(
            publish, case.job, api_factory=case.hub, retry_delay=0
        ).result(10)
    assert result["attempts"] == 2 and result["status"] == "verified"
    assert case.hub.uploads == (2 if failure == "upload_failures" else 1)


def test_retry_budget_exhaustion_retains_export_snapshot_and_resume(case):
    original = inventory(case.root)
    prepare(case)
    snapshot = inventory(case.job / "payload")
    case.hub.upload_failures = 10
    with ThreadPoolExecutor(1) as worker:
        future = worker.submit(
            publish, case.job, api_factory=case.hub, max_attempts=2, retry_delay=0
        )
        with pytest.raises(RuntimeError):
            future.result(10)
    assert case.hub.uploads == 2
    assert inventory(case.job / "payload") == snapshot
    for name, contents in original.items():
        assert (case.root / name).read_bytes() == contents
    assert json.loads((case.job / "receipt.json").read_text())["status"] == "failed"


def test_verification_retry_uses_saved_commit_without_reupload(case):
    prepare(case)
    case.hub.verify_failures = 1
    with pytest.raises(RuntimeError):
        publish(case.job, api_factory=case.hub, max_attempts=1)
    saved_commit = case.hub.commit
    assert json.loads((case.job / "receipt.json").read_text())["commit"] == saved_commit
    # HEAD may advance after the upload while this job awaits verification.
    (case.export / "model.safetensors").write_bytes(b"newer-weights")
    newer = prepare_publication(
        case.export, "tester/draft", case.root / "newer-job", "newer"
    )
    publish(newer, api_factory=case.hub)
    assert case.hub.commit != saved_commit
    # Create a fresh transport object to exercise recovery from the disk receipt.
    fresh = DiskHub(case.hub.root)
    result = publish(case.job, api_factory=fresh)
    assert result["status"] == "verified" and fresh.uploads == 0
    assert result["commit"] == saved_commit
    assert set(fresh.download_revisions) == {saved_commit}


def test_same_size_remote_corruption_is_never_verified(case):
    prepare(case)
    case.hub.corrupt = True
    with pytest.raises(RuntimeError):
        publish(case.job, api_factory=case.hub, max_attempts=2, retry_delay=0)
    assert case.hub.uploads == 1
    receipt = json.loads((case.job / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["attempts"] == 2
    assert case.checkpoint.is_file() and case.export.is_dir()


def test_snapshot_mutation_refused_before_transport(case):
    prepare(case)
    (case.job / "payload/model.safetensors").write_bytes(b"tampered")
    with pytest.raises(PublishInputError, match="snapshot changed"):
        publish(case.job, api_factory=case.hub)
    assert case.hub.uploads == 0


def test_visibility_failure_preserves_job_and_does_not_upload(case):
    prepare(case)
    case.hub.private = False
    with pytest.raises(PublishInputError, match="visibility"):
        publish(case.job, api_factory=case.hub)
    assert case.hub.uploads == 0
    assert json.loads((case.job / "receipt.json").read_text())["status"] == "failed"
    assert case.checkpoint.is_file() and case.export.is_dir()


def test_same_job_concurrent_publish_rejected(case):
    prepare(case)
    case.hub.release.clear()
    with ThreadPoolExecutor(2) as worker:
        first = worker.submit(publish, case.job, api_factory=case.hub)
        try:
            assert case.hub.entered.wait(10)
            second = worker.submit(publish, case.job, api_factory=case.hub)
            with pytest.raises(PublishInputError, match="already running"):
                second.result(5)
        finally:
            case.hub.release.set()
        assert first.result(10)["status"] == "verified"
    assert case.hub.uploads == 1


def test_prepare_rejects_symlinks_nested_job_and_existing_job(case):
    (case.export / "link").symlink_to(case.checkpoint)
    with pytest.raises(PublishInputError, match="symlinks"):
        prepare(case)
    (case.export / "link").unlink()
    with pytest.raises(PublishInputError, match="outside"):
        prepare_publication(case.export, "tester/draft", case.export / "job", "run-1")
    prepare(case)
    original = inventory(case.job)
    with pytest.raises(PublishInputError, match="already exists"):
        prepare(case)
    assert inventory(case.job) == original


def test_cli_real_library_adapter_with_only_transport_replaced(
    case, monkeypatch, capsys
):
    import huggingface_hub

    from draftfit.export.hf_publish import main

    prepare(case)
    monkeypatch.setattr(huggingface_hub, "HfApi", case.hub)
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-value")
    assert main(["--job-dir", str(case.job), "--max-attempts", "1"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "verified"
    assert "synthetic-secret" not in output.out + output.err


def test_prepared_job_publishes_from_fresh_cli_process(case):
    prepare(case)
    # A fresh interpreter must need only durable files, not parent worker state.
    code = """
import os, runpy, sys
from pathlib import Path
from tests.test_hf_publication import DiskHub
import huggingface_hub
huggingface_hub.HfApi = DiskHub(Path(sys.argv[1]))
os.environ['HF_TOKEN'] = 'synthetic-secret-value'
sys.argv = ['hf_publish', '--job-dir', sys.argv[2], '--max-attempts', '1']
runpy.run_module('draftfit.export.hf_publish', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(case.hub.root), str(case.job)],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "verified"
    assert "synthetic-secret" not in result.stdout + result.stderr
    assert (case.hub.root / "model.safetensors").read_bytes() == (
        case.export / "model.safetensors"
    ).read_bytes()


@pytest.mark.parametrize(
    "old_shard_size,new_shard_size",
    [
        ("10GB", "1KB"),
        ("1KB", "10GB"),
        ("1KB", "2KB"),
    ],
)
def test_republication_reloads_current_weights_and_removes_only_owned_files(
    tmp_path,
    old_shard_size,
    new_shard_size,
):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            tie_word_embeddings=False,
        )
    )
    hub = DiskHub(tmp_path / "hub")
    hub.root.mkdir()
    (hub.root / "README.md").write_text("User-owned model card\n")
    (hub.root / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (hub.root / "notes.txt").write_text("Keep me")
    unmanaged = inventory(hub.root)
    generations = []
    for generation, shard_size in enumerate((old_shard_size, new_shard_size)):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(generation))
        export = tmp_path / f"export-{generation}"
        model.save_pretrained(export, max_shard_size=shard_size)
        job = prepare_publication(
            export, "tester/draft", tmp_path / f"job-{generation}", "reload"
        )
        result = publish(job, api_factory=hub, retry_delay=0)
        assert result["status"] == "verified"
        generations.append(
            set(json.loads((job / "payload" / MANIFEST).read_text())["files"])
        )
    reloaded = Qwen3ForCausalLM.from_pretrained(hub.root, local_files_only=True)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], weight, rtol=0, atol=0)
    assert not (generations[0] - generations[1]) & set(inventory(hub.root))
    for name, contents in unmanaged.items():
        assert (hub.root / name).read_bytes() == contents
    assert len(hub.commits) == 2
    replacement = hub.commits[-1]
    assert set(replacement["deleted"]) == generations[0] - generations[1]
    assert set(replacement["added"]) == generations[1] | {MANIFEST}
    assert replacement["parent"] == hub.commits[0]["commit"]
    # The old revision stays complete; deletes/additions appear in one successor.
    old_tree = inventory(hub.history / replacement["parent"])
    assert generations[0] <= old_tree.keys()


@pytest.mark.parametrize("with_manifest", [False, True])
@pytest.mark.parametrize(
    "legacy_file",
    [
        "pytorch_model.bin",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
    ],
)
def test_unmanaged_legacy_weights_refused_without_deleting_anything(
    case, with_manifest, legacy_file
):
    case.hub.root.mkdir()
    (case.hub.root / legacy_file).write_bytes(b"legacy model")
    (case.hub.root / "README.md").write_text("User model card")
    if with_manifest:
        (case.hub.root / MANIFEST).write_text(
            json.dumps({"format": "draftfit-sha256-v1", "files": {}})
        )
    before = inventory(case.hub.root)
    prepare(case)
    with pytest.raises(PublishInputError, match="Unmanaged model files"):
        publish(case.job, api_factory=case.hub)
    assert case.hub.uploads == 0 and inventory(case.hub.root) == before


def test_existing_user_readme_collision_refused(case):
    case.hub.root.mkdir()
    (case.hub.root / "README.md").write_text("User model card")
    (case.export / "README.md").write_text("Export model card")
    prepare(case)
    with pytest.raises(PublishInputError, match="overwrite unmanaged"):
        publish(case.job, api_factory=case.hub)
    assert case.hub.uploads == 0
    assert inventory(case.hub.root) == {"README.md": b"User model card"}


def test_replacement_failure_is_atomic_and_deletion_paths_are_literal(case):
    (case.export / "obsolete[1].txt").write_text("Owned obsolete file")
    case.hub.root.mkdir()
    (case.hub.root / "obsolete1.txt").write_text("Unmanaged glob lookalike")
    prepare(case)
    publish(case.job, api_factory=case.hub)
    before, parent = inventory(case.hub.root), case.hub.commit
    (case.export / "obsolete[1].txt").unlink()
    (case.export / "model.safetensors").rename(
        case.export / "model-00001-of-00001.safetensors"
    )
    replacement = prepare_publication(
        case.export, "tester/draft", case.root / "replacement", "next"
    )
    case.hub.upload_failures = 1
    with pytest.raises(RuntimeError):
        publish(replacement, api_factory=case.hub, max_attempts=1)
    assert case.hub.commit == parent and inventory(case.hub.root) == before
    result = publish(replacement, api_factory=case.hub)
    assert result["status"] == "verified"
    assert set(case.hub.commits[-1]["deleted"]) == {
        "model.safetensors",
        "obsolete[1].txt",
    }
    assert (case.hub.root / "obsolete1.txt").read_text() == "Unmanaged glob lookalike"


def test_different_jobs_racing_same_head_cannot_overwrite_winner_even_on_retry(case):
    prepare(case)
    (case.export / "model.safetensors").write_bytes(b"competing weights")
    other = prepare_publication(
        case.export, "tester/draft", case.root / "other-job", "other"
    )
    case.hub.barrier = threading.Barrier(2)
    with ThreadPoolExecutor(2) as workers:
        futures = {
            job: workers.submit(publish, job, api_factory=case.hub, retry_delay=0)
            for job in (case.job, other)
        }
        winners, losers = [], []
        for job, future in futures.items():
            try:
                result = future.result(10)
                assert result["status"] == "verified"
                winners.append(job)
            except PublishInputError as error:
                assert "HEAD changed" in str(error)
                losers.append(job)
    assert len(winners) == len(losers) == 1
    assert len(case.hub.commits) == 1 and case.hub.uploads == 2
    assert inventory(case.hub.root) == inventory(winners[0] / "payload")
    # A fresh process/transport still respects the losing job's saved parent.
    fresh = DiskHub(case.hub.root)
    with pytest.raises(PublishInputError, match="HEAD changed"):
        publish(losers[0], api_factory=fresh)
    assert fresh.uploads == 0
    assert inventory(case.hub.root) == inventory(winners[0] / "payload")
