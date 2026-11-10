from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["pending", "deferred", "rejected", "pr_open", "integrated"]
PAPER_NOTE = re.compile(r"^(\d{4}\.\d{4,5})_v([1-9]\d*)(?:[_ .-]|$)")
PR_URL = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/pull/[1-9]\d*\Z")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Note(_Model):
    path: str
    sha256: str


class Decision(_Model):
    fingerprint: str
    version: int = Field(ge=1)
    notes: list[Note]
    status: Status
    reason: str
    pr_url: str | None = None
    recorded_at: str


class Paper(_Model):
    paper_id: str
    version: int = Field(ge=1)
    title: str
    notes: list[Note]
    fingerprint: str
    status: Status = "pending"
    available: bool = True
    decisions: list[Decision] = Field(default_factory=list)


class Ledger(_Model):
    schema_version: Literal[1] = 1
    source_id: str
    papers: dict[str, Paper] = Field(default_factory=dict)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> Ledger:
    return Ledger.model_validate_json(path.read_text(encoding="utf-8"))


@contextmanager
def _lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("intake state is in use by another process") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _save(path: Path, ledger: Ledger):
    content = json.dumps(ledger.model_dump(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _discover(root: Path):
    papers: dict[str, Paper] = {}
    context_notes = []
    note_count = 0
    for path in sorted(root.rglob("*.md")):
        if path.is_symlink() or any(
            (root / parent).is_symlink()
            for parent in path.relative_to(root).parents
            if parent != Path(".")
        ):
            raise ValueError(f"symlink notes are not supported: {path.name}")
        relative = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"note is outside the literature directory: {relative}")
        note_count += 1
        match = PAPER_NOTE.match(path.name)
        if match is None:
            context_notes.append(relative)
            continue
        paper_id, version = match[1], int(match[2])
        data = path.read_bytes()
        text = data.decode("utf-8")
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        note = Note(path=relative, sha256=_digest(data))
        current = papers.get(paper_id)
        if current is None or version > current.version:
            papers[paper_id] = Paper(
                paper_id=paper_id,
                version=version,
                title=title,
                notes=[note],
                fingerprint="",
            )
        elif version == current.version:
            current.notes.append(note)
    for paper in papers.values():
        source = {
            "version": paper.version,
            "notes": [n.model_dump() for n in paper.notes],
        }
        paper.fingerprint = _digest(json.dumps(source, sort_keys=True).encode("utf-8"))
    return papers, context_notes, note_count


def _scan(root: Path, state: Path, limit: int, dry_run: bool):
    source_id = _digest(str(root).encode("utf-8"))
    ledger = _load(state) if state.exists() else Ledger(source_id=source_id)
    if ledger.source_id != source_id:
        raise ValueError("state belongs to a different literature directory")
    discovered, context_notes, note_count = _discover(root)
    if not discovered and not ledger.papers:
        raise ValueError("no versioned paper notes found in the literature directory")
    changed = []
    for identifier, previous in ledger.papers.items():
        previous.available = (
            identifier in discovered
            and discovered[identifier].version >= previous.version
        )
    for identifier, current in discovered.items():
        previous = ledger.papers.get(identifier)
        if previous is not None and previous.version > current.version:
            continue
        if previous is None or previous.fingerprint != current.fingerprint:
            if previous is not None:
                current.decisions = previous.decisions
            ledger.papers[identifier] = current
            changed.append(identifier)
    if not dry_run:
        _save(state, ledger)
    pending = sorted(
        (
            paper
            for paper in ledger.papers.values()
            if paper.available and paper.status == "pending"
        ),
        key=lambda paper: paper.paper_id,
        reverse=True,
    )
    candidates = []
    for paper in pending[:limit]:
        candidate = paper.model_dump(exclude={"decisions"})
        candidate["paper_url"] = (
            f"https://arxiv.org/abs/{paper.paper_id}v{paper.version}"
        )
        candidate["previous_pr_urls"] = sorted(
            {d.pr_url for d in paper.decisions if d.pr_url}
        )
        candidates.append(candidate)
    return {
        "schema_version": 1,
        "note_count": note_count,
        "paper_count": len(ledger.papers),
        "pending_count": len(pending),
        "changed": sorted(changed),
        "unavailable": sorted(
            p.paper_id for p in ledger.papers.values() if not p.available
        ),
        "context_notes": context_notes,
        "candidates": candidates,
    }


def _paths(notes_dir, state_path):
    root = Path(notes_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("notes-dir must be a literature directory")
    state = Path(state_path).resolve()
    if state.is_relative_to(root):
        raise ValueError("state must be outside the read-only literature directory")
    return root, state


def scan(notes_dir, state_path, limit=1, dry_run=False):
    """Discover latest sources without completing or claiming any candidate."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200")
    root, state = _paths(notes_dir, state_path)
    if dry_run:
        return _scan(root, state, limit, True)
    with _lock(state):
        return _scan(root, state, limit, False)


def record(notes_dir, state_path, paper_id, fingerprint, status, reason, pr_url=None):
    """Record a decision against the exact source fingerprint returned by scan."""
    if status not in get_args(Status):
        raise ValueError("unknown intake status")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a decision requires a nonempty reason")
    if (pr_url is not None and not PR_URL.fullmatch(pr_url)) or (
        status == "pr_open" and pr_url is None
    ):
        raise ValueError(
            "pr_open requires a GitHub PR URL; supplied PR URLs must be valid"
        )
    root, state = _paths(notes_dir, state_path)
    if not state.exists():
        raise ValueError("state does not exist; scan the literature directory first")
    with _lock(state):
        ledger = _load(state)
        if ledger.source_id != _digest(str(root).encode("utf-8")):
            raise ValueError("state belongs to a different literature directory")
        if paper_id not in ledger.papers:
            raise ValueError("unknown paper ID; scan the literature directory first")
        paper = ledger.papers[paper_id]
        observed, _, _ = _discover(root)
        current = observed.get(paper_id)
        if (
            not paper.available
            or paper.fingerprint != fingerprint
            or current is None
            or current.fingerprint != fingerprint
        ):
            raise ValueError(
                "source fingerprint is stale or unavailable; scan and review again"
            )
        decision = Decision(
            fingerprint=fingerprint,
            version=paper.version,
            notes=paper.notes,
            status=status,
            reason=reason.strip(),
            pr_url=pr_url,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        if not paper.decisions or paper.decisions[-1].model_dump(
            exclude={"recorded_at"}
        ) != decision.model_dump(exclude={"recorded_at"}):
            paper.decisions.append(decision)
        paper.status = status
        _save(state, ledger)
        return paper.model_dump()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Track experimental literature decisions locally"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("scan", help="Find pending paper revisions")
    discover.add_argument("--notes-dir", required=True)
    discover.add_argument("--state", required=True)
    discover.add_argument("--limit", type=int, default=1)
    discover.add_argument("--dry-run", action="store_true")
    decision = commands.add_parser("record", help="Record a reviewed source decision")
    decision.add_argument("--notes-dir", required=True)
    decision.add_argument("--state", required=True)
    decision.add_argument("--paper-id", required=True)
    decision.add_argument("--fingerprint", required=True)
    decision.add_argument("--status", choices=get_args(Status), required=True)
    decision.add_argument("--reason", required=True)
    decision.add_argument("--pr-url")
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            result = scan(args.notes_dir, args.state, args.limit, args.dry_run)
        else:
            result = record(
                args.notes_dir,
                args.state,
                args.paper_id,
                args.fingerprint,
                args.status,
                args.reason,
                args.pr_url,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
