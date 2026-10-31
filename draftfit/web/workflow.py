"""Render a small allowlist of existing engine commands, without a shell."""

from __future__ import annotations

import json
from pathlib import Path


def unattended_config(engine, value):
    path = engine.file(value)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != "draftfit-unattended-v1":
        raise ValueError("Select an existing draftfit-unattended-v1 controller config.")
    # This adapter never adds a publication or cloud-resource execution path.
    if raw.get("publication") is not None:
        raise ValueError("Publication is not supported by the web adapter.")
    for name in ("manifest", "budget_path", "output_dir"):
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"The unattended config requires {name}.")
        engine.file(str(path.parent / raw[name]), required=name != "output_dir")
    return path, raw


def build_commands(engine, action, *, config_path=None, params=None):
    params = params or {}
    if action in ("plan", "train"):
        if params or config_path is None:
            raise ValueError(
                "Training uses the saved config without execution overrides."
            )
        return [
            engine.command(
                "train", "-c", config_path, *(["--plan"] if action == "plan" else [])
            )
        ]
    if action == "validate_tokens":
        if set(params) != {"expected", "actual"}:
            raise ValueError(
                "Token validation requires expected and actual JSON paths."
            )
        paths = [engine.file(params[key]) for key in ("expected", "actual")]
        return [
            engine.command(
                "validate", "tokens", "--expected", paths[0], "--actual", paths[1]
            )
        ]
    if action in ("unattended_plan", "unattended", "unattended_resume"):
        if set(params) != {"config_path"}:
            raise ValueError("Unattended actions require only config_path.")
        path, _ = unattended_config(engine, params["config_path"])
        flags = {
            "unattended_plan": ["--plan"],
            "unattended": [],
            "unattended_resume": ["--resume"],
        }[action]
        return [engine.command("unattended", "--config", path, *flags)]
    raise ValueError(
        f"Unsupported web action: {action}. Use the current engine CLI for other workflows."
    )


def read_tail(path, limit=256 * 1024):
    """Bound reads of growing engine logs and ignore a partially written last row."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - limit)
        stream.seek(start)
        if start:
            stream.readline()
        return stream.read().decode("utf-8", errors="replace")


def unattended_status(engine, config_path):
    path, raw = unattended_config(engine, config_path)
    root = engine.file(str(path.parent / raw["output_dir"]), required=False)
    state_path = engine.file(str(root / "state.json"), required=False)
    state = json.loads(state_path.read_text()) if state_path.is_file() else None
    rows = []
    # Only the latest engine attempt is displayed. Older successful phases must
    # not mask a failed or interrupted current attempt.
    phases = state.get("phases", []) if isinstance(state, dict) else []
    attempts = phases[-1].get("attempts", []) if phases else []
    if attempts:
        curves = engine.file(
            str(Path(attempts[-1]) / "learning-curves.jsonl"), required=False
        )
        if curves.is_file():
            for line in read_tail(curves).splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    queue = next(
        (
            r.get("queue") if r.get("kind") == "capture" else r
            for r in reversed(rows)
            if r.get("kind") in ("capture", "queue_closed")
        ),
        None,
    )
    return {
        "config_path": str(path),
        "output_dir": str(root),
        "state": state,
        "events": rows[-500:],
        "queue": queue,
        "evidence_scope": "Saved engine observations only; process completion is not a GPU or speedup claim.",
    }
