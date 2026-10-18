"""Staged Humanizer generation, continual joint training, held-out decisions and exports."""
from __future__ import annotations

import hashlib
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts import run_humanizer_campaign as campaign
from scripts.humanizer_staged_policy import (FAMILIES, accepted_slice, atomic_json, decide,
                                            publish_slice, publish_pool_slice, token_content_hash)


def _accepted_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(json.loads(line).get("status") == "accepted" for line in source)


def _pool_count(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(bool(line.strip()) for line in source)


def _token_hashes(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as source:
        return {token_content_hash(json.loads(line))
                for line in source if line.strip()}


def _continuation_contract(args, validation: Path) -> dict:
    """Bind the old frozen pool to the reserved prefix without rewriting generation state."""
    from scripts.generate_humanizer_campaign_pairs import read_sources, load_journal
    journal = args.continuation_journal.resolve()
    if journal.name != "train.partial.jsonl":
        raise ValueError("continuation journal must be the original train.partial.jsonl")
    inputs, fingerprint = read_sources([args.train_input, *args.train_reserve])
    progress = json.loads(journal.with_name("train.progress.json").read_text())
    if progress.get("required_count") != 30000 or progress.get("input_sha256") != fingerprint:
        raise ValueError("continuation generation input fingerprint or required_count differs")
    load_journal(journal, inputs)
    tokens, ids = accepted_slice(journal, 0, args.continuation_start_row)
    def canonical(row):
        return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = []
    for path in (args.train_tokens, validation):
        with path.open(encoding="utf-8") as source:
            expected.extend(canonical(json.loads(line)) for line in source if line.strip())
    if Counter(map(canonical, tokens)) != Counter(expected):
        raise ValueError("continuation prefix must equal the entire original train pool plus fixed held-out set")
    contract = {"journal": str(journal), "start_row": args.continuation_start_row,
                "input_sha256": fingerprint, "required_count": 30000,
                "pool_sha256": campaign.sha(args.train_tokens),
                "validation_sha256": campaign.sha(validation),
                "prefix_sha256": hashlib.sha256(json.dumps([ids, tokens], sort_keys=True).encode()).hexdigest()}
    path = args.root / "continuation.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("continuation provenance changed")
    atomic_json(path, contract)
    return contract


def _generate(args, target: int, started: float, journal: Path | None = None) -> None:
    journal = journal or args.root / "pairs/train.partial.jsonl"
    if _accepted_count(journal) >= target:
        return
    if args.train_input is None:
        raise ValueError("more rows needed; provide --train-input to resume the original journal")
    remaining = campaign.check_limit(args, started, "next generation slice")
    command = [sys.executable, str(Path(__file__).with_name("generate_humanizer_campaign_pairs.py")),
               "--input", str(args.train_input), "--output-dir", str(journal.parent),
               "--required-count", "30000",
               "--batch-size", str(args.generation_batch_size),
               "--hourly-usd", str(args.hourly_usd), "--max-projected-usd",
               str(min(args.generation_budget_usd, remaining)), "--local-files-only"]
    if target < 30000:
        command += ["--stop-after-accepted", str(target)]
    deadline = min(args.deadline_unix or math.inf,
                   time.time() + remaining * 3600 / args.hourly_usd)
    command += ["--deadline-unix", str(deadline)]
    for source in args.train_reserve:
        command += ["--reserve-input", str(source)]
    with (args.root / "generation-train.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    # The generator's bounded partial exit is expected, but its journal must
    # actually contain the requested accepted rows before training can begin.
    if result.returncode not in (0, 2) or _accepted_count(journal) < target:
        raise RuntimeError("generation did not reach the next immutable stage; inspect generation-train.log")


def _manifest(args, train: Path, validation: Path, folder: Path) -> Path:
    template = args.manifest or args.fixture_manifest
    if template is None:
        from draftfit.humanizer_campaign import prepare
        template = args.root / "template/family.json"
        if not template.exists():
            campaign.check_limit(args, time.monotonic(), "pinned seed configuration")
            prepare(template.parent, campaign.seed_snapshot(), train_tokens=train,
                    validation_tokens=validation)
    source = Path(template).resolve()
    template_data = json.loads(source.read_text())
    base_path = source.parent / template_data["base_config"]
    base = json.loads(base_path.read_text())
    def original_path(value):
        if not isinstance(value, str):
            return value
        candidate = source.parent / value
        return str(candidate.resolve()) if candidate.exists() else value
    for key in ("target_model_path", "draft_model_config", "draft_checkpoint_path"):
        if key in base.get("model", {}):
            base["model"][key] = original_path(base["model"][key])
    for entry in template_data["runs"]:
        for key in ("checkpoint", "draft_config"):
            if key in entry:
                entry[key] = original_path(entry[key])
    base["data"]["prompts_path"] = str(train.resolve())
    base["data"]["eval_data_path"] = str(validation.resolve())
    folder.mkdir(parents=True, exist_ok=True)
    desired = {"base.json": base, "family.json": {**template_data, "base_config": "base.json"}}
    for name, value in desired.items():
        path = folder / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"immutable stage configuration differs: {path}")
        else:
            atomic_json(path, value)
    return folder / "family.json"


def _validation(args, started: float) -> tuple[Path, set[str], int]:
    if not args.validation_tokens and not args.validation_input:
        count = 2 if args.tiny else 500
        folder = args.root / "fixed-validation"
        data = publish_slice(args.root / "pairs/train.partial.jsonl", folder, 0, count, set())
        return Path(data["tokens"]), set(data["ids"]), count
    source = campaign.token_source(args, "validation", 2 if args.tiny else 500, started)
    ids = set()
    manifest = (source.with_name(source.name.replace(".tokens.jsonl", ".manifest.json"))
                if source.name.endswith(".tokens.jsonl") else source.with_suffix(".manifest.json"))
    journal = args.root / "pairs/validation.partial.jsonl"
    if manifest.exists():
        ids = set(json.loads(manifest.read_text()).get("ids", []))
    elif journal.exists():
        with journal.open(encoding="utf-8") as stream:
            ids = {row["id"] for line in stream if (row := json.loads(line)).get("status") == "accepted"}
    return source, ids, 0


def _attempt(root: Path, phase: int) -> tuple[Path, tuple[int, Path | None]]:
    stage_root = root / "run" / f"stage-{phase:03d}"
    current = campaign.latest_checkpoint(stage_root)
    number = len(list(stage_root.glob("attempt-*")))
    return stage_root / f"attempt-{number:03d}", current


def _losses(result: dict) -> dict[str, float]:
    values = {}
    for row in result["runs"]:
        value = row.get("metrics", {}).get("eval/avg_loss")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"missing fixed held-out loss for {row['name']}")
        values[row["name"]] = float(value)
    if set(values) != set(FAMILIES):
        raise ValueError("joint evaluation did not cover all three families")
    return values


def _prune_exports(root: Path, history: list[dict]) -> None:
    latest = history[-1]
    keep = {(latest["phase"], name) for name in FAMILIES}
    for name in FAMILIES:
        best = min(history, key=lambda row: row["losses"][name])
        keep.add((best["phase"], name))
    best = {name: min(history, key=lambda row: row["losses"][name])["exports"][name]
            for name in FAMILIES}
    selected = (latest["exports"], best)
    for index in selected:
        if set(index) != set(FAMILIES):
            raise ValueError("export index is missing a family")
        for name, arm in index.items():
            weight = Path(arm["export"]) / "model.safetensors"
            if arm["name"] != name or not weight.is_file() or campaign.sha(weight) != arm["weights_sha256"]:
                raise ValueError(f"selected {name} export checksum differs; refusing cleanup")
    atomic_json(root / "exports/latest.json", latest["exports"])
    atomic_json(root / "exports/best.json", best)
    for folder in (root / "exports").glob("stage-*/*"):
        if folder.is_dir() and (int(folder.parent.name.removeprefix("stage-")), folder.name) not in keep:
            shutil.rmtree(folder)


def _prune_checkpoints(root: Path, history: list[dict]) -> None:
    """Retain the only exact-resume state and small learning-curve evidence."""
    latest = Path(history[-1]["checkpoint"]).resolve()
    if not latest.is_file() or campaign.sha(latest) != history[-1]["checkpoint_sha256"]:
        raise ValueError("latest exact-resume checkpoint changed; refusing cleanup")
    for phase_root in (root / "run").glob("stage-*/*"):
        if not phase_root.is_dir():
            continue
        for checkpoint in phase_root.glob("joint-step*.pt"):
            if checkpoint.resolve() != latest:
                checkpoint.unlink()
        for arm in FAMILIES:
            folder = phase_root / arm
            if not folder.is_dir():
                continue
            for state in folder.glob(f"{arm}-step*/training_state.pt"):
                # HF exports are already verified and indexed before cleanup;
                # this per-arm state duplicates large draft weights.
                state.unlink()


def run_staged(args) -> dict:
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state_path = args.root / "staged-status.json"
    prior = json.loads(state_path.read_text()) if state_path.exists() else None
    history = list(prior.get("history", [])) if prior else []
    if prior and prior.get("status") in {"complete", "early_stopped"}:
        existing_validation, _, _ = _validation(args, started)
        if prior.get("validation_sha256") != campaign.sha(existing_validation):
            raise ValueError("fixed held-out validation changed")
        for row in history:
            staged_data = args.root / f"stages/stage-{row['phase']:03d}/train.tokens.jsonl"
            if campaign.sha(staged_data) != row["data_sha256"]:
                raise ValueError("existing staged token file changed")
            receipt = json.loads(staged_data.with_name("data.json").read_text())
            if "pool_sha256" in receipt and (not args.train_tokens or
                    receipt["pool_sha256"] != campaign.sha(args.train_tokens)):
                raise ValueError("original train pool changed")
        _prune_exports(args.root, history)
        return prior
    state = {"status": "running", "history": history, "horizon": 10 if args.tiny else 30000}
    atomic_json(state_path, state)
    try:
        validation, heldout_ids, reserved_rows = _validation(args, started)
        validation_hash = campaign.sha(validation)
        heldout_hashes = _token_hashes(validation)
        if prior and prior.get("validation_sha256") not in (None, validation_hash):
            raise ValueError("fixed held-out validation changed")
        state["validation_sha256"] = validation_hash
        used_ids = set(heldout_ids)
        used_hashes = set(heldout_hashes)
        if args.train_tokens:
            used_hashes.update(_token_hashes(args.train_tokens))
        for row in history:
            receipt = json.loads((args.root / f"stages/stage-{row['phase']:03d}/data.json").read_text())
            if receipt["sha256"] != row["data_sha256"] or used_ids.intersection(receipt["ids"]):
                raise ValueError("existing stage data identity or disjointness differs")
            staged_data = args.root / f"stages/stage-{row['phase']:03d}/train.tokens.jsonl"
            if campaign.sha(staged_data) != row["data_sha256"]:
                raise ValueError("existing staged token file changed")
            if "pool_sha256" in receipt and (not args.train_tokens or
                    receipt["pool_sha256"] != campaign.sha(args.train_tokens)):
                raise ValueError("original train pool changed")
            used_ids.update(receipt["ids"])
            used_hashes.update(_token_hashes(staged_data))
            if row.get("source_kind") == "continuation":
                publish_slice(Path(receipt["journal"]), staged_data.parent, receipt["start"],
                              receipt["count"], set())
        continuation = None
        if getattr(args, "continuation_journal", None):
            continuation = _continuation_contract(args, validation)
            _, prefix_ids = accepted_slice(Path(continuation["journal"]), 0, continuation["start_row"])
            used_ids.update(prefix_ids)
        elif any(row.get("source_kind") == "continuation" for row in history):
            raise ValueError("resuming journal stages requires the original continuation arguments")
        step = history[-1]["step"] if history else 0
        pool_offset = reserved_rows + sum(row["rows"] for row in history
                                          if row.get("source_kind") != "continuation")
        journal_offset = (continuation["start_row"] + sum(row["rows"] for row in history
                          if row.get("source_kind") == "continuation")) if continuation else 0
        previous = Path(history[-1]["checkpoint"]) if history else None
        budget = history[-1]["next_step_budget"] if history else args.stage_first_steps
        phase = len(history)
        while step < state["horizon"]:
            campaign.check_limit(args, started, "stage start")
            available = _pool_count(args.train_tokens) if args.train_tokens else 30000
            continuing = bool(continuation and pool_offset >= available)
            offset = journal_offset if continuing else pool_offset
            if continuing:
                available = 30000
            rows = min(args.stage_rows, available - offset)
            if continuing:
                rows = min(rows, budget, state["horizon"] - step)
            if rows < 1:
                state["status"] = "data_exhausted"
                break
            stage = args.root / f"stages/stage-{phase:03d}"
            if continuing:
                journal = Path(continuation["journal"])
                _generate(args, offset + rows, started, journal=journal)
                data = publish_slice(journal, stage, offset, rows, used_ids, used_hashes)
            elif args.train_tokens:
                data = publish_pool_slice(args.train_tokens.resolve(), stage, offset, rows,
                                          heldout_hashes)
            else:
                _generate(args, offset + rows, started)
                data = publish_slice(args.root / "pairs/train.partial.jsonl", stage,
                                     offset, rows, used_ids)
            used_ids.update(data["ids"])
            used_hashes.update(_token_hashes(Path(data["tokens"])))
            manifest = _manifest(args, Path(data["tokens"]), validation, stage / "config")
            stop = min(state["horizon"], step + budget)
            run_dir, (current, current_checkpoint) = _attempt(args.root, phase)
            if current_checkpoint is not None and current >= stop:
                run_dir = current_checkpoint.parent
            else:
                campaign.launch_training(args, manifest, run_dir, current_checkpoint or previous,
                                         stop, started, data_phase=phase)
                current, current_checkpoint = campaign.latest_checkpoint(run_dir.parent)
            if current_checkpoint is None:
                raise RuntimeError("stage training has no resumable checkpoint")
            result_path = current_checkpoint.parent / "worker-result.json"
            if not result_path.exists():
                raise RuntimeError("stage checkpoint exists but worker result is missing")
            result = json.loads(result_path.read_text())
            destination = args.root / f"exports/stage-{phase:03d}"
            if destination.exists():
                # Only an uncommitted phase reaches here; regenerate it from
                # the verified checkpoint after a crash during conversion.
                shutil.rmtree(destination)
            exports = campaign.export_and_verify(result, current_checkpoint.parent, destination)
            if result.get("status") == "stopped" or current < stop:
                state.update(status="stopped", reason=result.get("reason", "stage_incomplete"),
                             latest_checkpoint=str(current_checkpoint), completed_step=current,
                             latest_exports={arm["name"]: arm for arm in exports})
                break
            losses = _losses(result)
            decision = decide(history, losses, step=current,
                              patience=args.stage_patience, min_delta=args.stage_min_delta)
            next_budget = min(1000, budget * 2) if all(decision["improved"].values()) else \
                250 if not any(decision["improved"].values()) else 500
            record = {**decision, "phase": phase, "rows": rows, "start_row": offset,
                      "source_kind": "continuation" if continuing else "pool" if args.train_tokens else "journal",
                      "planned_step_budget": stop - step,
                      "actual_train_steps": current - step,
                      "data_sha256": data["sha256"], "validation_sha256": validation_hash,
                      "checkpoint": str(current_checkpoint),
                      "checkpoint_sha256": campaign.sha(current_checkpoint),
                      "exports": {arm["name"]: arm for arm in exports},
                      "next_step_budget": next_budget}
            history.append(record)
            state.update(history=history, completed_step=current,
                         status="early_stopped" if decision["stop"] else
                                "complete" if current == state["horizon"] else "running")
            atomic_json(state_path, state)
            _prune_exports(args.root, history)
            _prune_checkpoints(args.root, history)
            if state["status"] != "running":
                break
            if continuing:
                journal_offset += rows
            else:
                pool_offset += rows
            previous, step, budget, phase = current_checkpoint, current, next_budget, phase + 1
    except BaseException as exc:
        state.update(status="stopped" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        atomic_json(state_path, state)
        raise
    atomic_json(state_path, state)
    return state
