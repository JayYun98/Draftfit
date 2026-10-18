"""DFlash-family runs, handoffs, and joint offline/online training."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.config import Config, apply_overrides, load_config
from draftfit.training.model_loading import (
    _apply_draft_overrides,
    convert_family_draft_config,
    load_draft_config_source,
    resolve_draft_config,
)


@dataclass(frozen=True)
class FamilyStage:
    name: str
    config: Config
    draft_config: object
    parent: str | None


def _local_path(value: str, base: Path) -> str:
    local_uri = value.startswith("file://")
    if local_uri:
        value = value[7:]
    if value.startswith("~"):
        return str(Path(value).expanduser().resolve())
    if local_uri or value.startswith((".", "/")) or (base / value).exists():
        return str((base / value).expanduser().resolve())
    return value  # Hugging Face repository ID


def build_family_plan(manifest_path: str) -> tuple[Path, list[FamilyStage]]:
    path = Path(manifest_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or set(manifest) != {"base_config", "output_dir", "runs"}:
        raise ValueError("family manifest requires exactly base_config, output_dir and runs")
    base_dir = path.parent
    root = (base_dir / manifest["output_dir"]).expanduser().resolve()
    base = load_config(str((base_dir / manifest["base_config"]).expanduser().resolve()))
    online = base.training.joint_online_capture
    train_source = base.data.prompts_path if online else base.data.hidden_states_path
    eval_source = base.data.eval_data_path if online else base.data.eval_hidden_states_path
    if not train_source or not eval_source or not base.training.eval_interval:
        raise ValueError("family runs require offline train and held-out eval features with training.eval_interval")
    train_path = Path(train_source).resolve()
    eval_path = Path(eval_source).resolve()
    if train_path == eval_path or train_path in eval_path.parents or eval_path in train_path.parents:
        raise ValueError("family train and held-out eval feature paths must differ")
    if base.training.resume_from:
        raise ValueError("family stages use weights-only handoff; clear training.resume_from")
    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("family runs must be a nonempty array")
    stages: list[FamilyStage] = []
    by_name: dict[str, FamilyStage] = {}
    protected = {"output_dir", "run_id", "training.strategy", "training.resume_from",
                 "model.target_model_path", "model.target_revision", "model.draft_checkpoint_path",
                 "model.draft_model_config", "data.hidden_states_path", "data.eval_hidden_states_path",
                 "training.joint_online_capture", "data.prompts_path", "data.eval_data_path"}
    for entry in runs:
        if not isinstance(entry, dict) or set(entry) - {"name", "strategy", "from", "checkpoint", "draft_config", "overrides"}:
            raise ValueError("each family run accepts name, strategy, from, checkpoint, draft_config and overrides")
        name, strategy = entry.get("name"), entry.get("strategy")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) or name in by_name:
            raise ValueError(f"invalid or duplicate family run name: {name!r}")
        if strategy not in ("dflash", "dspark", "dflash2"):
            raise ValueError(f"unsupported family strategy: {strategy!r}")
        parent = entry.get("from")
        if parent is not None and parent not in by_name:
            raise ValueError(f"family run {name!r} must reference an earlier run")
        if parent is not None and entry.get("checkpoint"):
            raise ValueError("choose either from or checkpoint for a family run")
        overrides = entry.get("overrides", [])
        if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
            raise ValueError("family overrides must be a list of dotted path=value strings")
        if any(item.split("=", 1)[0] in protected for item in overrides):
            raise ValueError(f"family run {name!r} overrides a protected target, data or handoff field")
        raw = base.model_dump(mode="json")
        raw["run_id"] = name
        raw["output_dir"] = str(root / name)
        raw["training"]["strategy"] = strategy
        raw["training"]["resume_from"] = None
        custom_config = entry.get("draft_config")
        if custom_config is not None and not isinstance(custom_config, str):
            raise ValueError("family draft_config must be a path or repository ID")
        if parent is not None:
            raw["model"]["draft_checkpoint_path"] = str(root / parent)
            raw["model"]["draft_model_config"] = None
        elif entry.get("checkpoint"):
            raw["model"]["draft_checkpoint_path"] = _local_path(entry["checkpoint"], base_dir)
            raw["model"]["draft_model_config"] = None
        if custom_config:
            raw["model"]["draft_model_config"] = _local_path(custom_config, base_dir)
        cfg = apply_overrides(Config.model_validate(raw), overrides)
        if cfg.data.hidden_states_path != base.data.hidden_states_path or cfg.data.eval_hidden_states_path != base.data.eval_hidden_states_path:
            raise ValueError("family runs must share training and held-out feature sources")
        registration = builtin_algorithm_registry().resolve(strategy)
        if custom_config:
            draft = load_draft_config_source(cfg.model.draft_model_config)
            provider = registration.providers.model.draft_config
            compatible = provider.compatible_architectures or {provider.architecture}
            if len(draft.architectures) != 1 or draft.architectures[0] not in compatible:
                raise ValueError(f"family run {name!r} draft_config architecture differs from strategy")
        elif parent is not None:
            draft = convert_family_draft_config(
                by_name[parent].draft_config, strategy,
                retrain_anchor_alignment=cfg.training.retrain_anchor_alignment,
                mask_token_id=cfg.model.mask_token_id,
            )
        elif cfg.model.draft_model_config:
            # The common scratch config defines geometry; each arm selects its
            # own family architecture. An explicit per-arm config stays strict.
            draft = convert_family_draft_config(
                load_draft_config_source(cfg.model.draft_model_config), strategy,
                retrain_anchor_alignment=cfg.training.retrain_anchor_alignment,
                mask_token_id=cfg.model.mask_token_id,
            )
        else:
            draft = resolve_draft_config(cfg, provider=registration.providers.model.draft_config)
        if custom_config or parent is not None or cfg.model.draft_model_config:
            _apply_draft_overrides(cfg, draft, registration.providers.model.draft_config)
        method = dict(getattr(draft, "dflash_config", None) or {})
        configured_mask = cfg.model.mask_token_id
        if configured_mask is not None:
            if method.get("mask_token_id") not in (None, configured_mask):
                raise ValueError(f"family run {name!r} has conflicting mask_token_id")
            method["mask_token_id"] = configured_mask
            draft.dflash_config = method
        if parent is not None and method.get("mask_token_id") is None:
            raise ValueError("family handoff requires model.mask_token_id or a source draft mask_token_id")
        if parent is not None and by_name[parent].config.training.strategy != strategy:
            parent_draft = by_name[parent].draft_config
            if method.get("sample_from_anchor", False) is not False:
                raise ValueError("family cross-strategy handoff requires sample_from_anchor=false")
            for field in ("block_size", "vocab_size", "hidden_size", "num_hidden_layers"):
                if getattr(draft, field) != getattr(parent_draft, field):
                    raise ValueError(f"family run {name!r} has incompatible {field}")
            parent_method = dict(getattr(parent_draft, "dflash_config", None) or {})
            for field in ("target_layer_ids", "mask_token_id"):
                if method.get(field) != parent_method.get(field):
                    raise ValueError(f"family run {name!r} has incompatible {field}")
        persisted = cfg.model_dump(mode="json")
        persisted["model"]["draft_model_config"] = str(root / name / "draft.json")
        cfg = Config.model_validate(persisted)
        # Validate the exact persisted config shape before starting any worker.
        from draftfit.application import resolve_run
        from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader

        if not online:
            resolve_run(cfg)
        else:
            from draftfit.application.planning import validate_resolved_run
            validate_resolved_run(cfg, registration)
        keys = ("input_ids", "loss_mask", "hidden_states")
        if strategy == "dspark":
            keys += ("target_last_hidden_states",)
        capture_versions = []
        capture_revisions = []
        for feature_path in (() if online else (cfg.data.hidden_states_path, cfg.data.eval_hidden_states_path)):
            reader = OfflineManifestReader(feature_path, strategy=strategy, feature_keys=keys)
            if reader.capture_manifest is not None:
                capture = reader.capture_manifest
                capture_versions.append(capture["tokenizer"]["version"])
                capture_revisions.append(capture["target"]["revision"])
                contract = capture["contract"]
                # Native producers name the algorithm capture layout; older
                # family caches use the shared target_hidden method.
                capture_strategy = contract["strategy"]
                expected_method = "dspark" if capture_strategy == "dspark" else "dflash"
                if contract.get("capture_method") not in (expected_method, "target_hidden"):
                    raise ValueError(f"family run {name!r} has unsupported capture method")
                has_target = "target_last_hidden_states" in contract["feature_names"]
                expected_repr = "hidden_state" if has_target else None
                expected_feature = "target_last_hidden_states" if has_target else None
                if (contract.get("target_repr") != expected_repr
                        or contract.get("target_feature") != expected_feature):
                    raise ValueError(f"family run {name!r} has unsupported capture target representation or feature")
                if capture["target"]["model"] != cfg.model.target_model_path:
                    raise ValueError(f"family run {name!r} target differs from capture manifest")
                if cfg.model.target_revision and capture["target"]["revision"] != cfg.model.target_revision:
                    raise ValueError(f"family run {name!r} target revision differs from capture manifest")
                if capture["contract"]["capture_layers"] != list(method.get("target_layer_ids", [])):
                    raise ValueError(f"family run {name!r} target layers differ from capture manifest")
        if len(capture_revisions) == 2 and capture_revisions[0] != capture_revisions[1]:
            raise ValueError(f"family run {name!r} train/holdout target revisions differ")
        if len(capture_versions) == 2 and capture_versions[0] != capture_versions[1]:
            raise ValueError(f"family run {name!r} train/holdout tokenizer versions differ")
        stage = FamilyStage(name, cfg, draft, parent)
        stages.append(stage)
        by_name[name] = stage
    return root, stages


def _summary(stage: FamilyStage) -> dict:
    return {"name": stage.name, "strategy": stage.config.training.strategy,
            "from": stage.parent, "checkpoint": stage.config.model.draft_checkpoint_path,
            "output_dir": stage.config.output_dir,
            "eval_interval": stage.config.training.eval_interval,
            "eval_features": stage.config.data.eval_hidden_states_path,
            "online_capture": stage.config.training.joint_online_capture,
            "eval_data": stage.config.data.eval_data_path}


def run_family(manifest_path: str, *, plan: bool = False, jobs: int = 1,
               joint: bool = False, resume_from: str | None = None,
               stop_at_step: int | None = None) -> dict:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("family jobs must be a positive integer")
    if stop_at_step is not None and (not joint or type(stop_at_step) is not int or stop_at_step < 1):
        raise ValueError("family --stop-at-step requires --joint and a positive step")
    if resume_from and not joint:
        raise ValueError("family --resume-from requires --joint")
    if joint and jobs != 1:
        raise ValueError("family --joint uses one shared optimizer; do not combine with --jobs")
    root, stages = build_family_plan(manifest_path)
    if any(s.config.training.joint_online_capture for s in stages) and not joint:
        raise ValueError("online family capture requires --joint")
    if joint:
        from draftfit.training.joint_family import validate_joint_family
        validate_joint_family(stages)
        if resume_from:
            resume_from = str(Path(resume_from).expanduser().resolve())
            if not Path(resume_from).is_file():
                raise FileNotFoundError(f"joint resume checkpoint file does not exist: {resume_from}")
    if jobs > 1:
        if any(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
            raise ValueError("concurrent family must run outside an existing torchrun worker")
        for stage in stages:
            cfg = stage.config
            if (cfg.mode != "offline" or cfg.deployment.mode != "local_colocated"
                    or cfg.deployment.trainer.nnodes != 1
                    or cfg.deployment.trainer.nproc_per_node != 1
                    or cfg.training.tp_size != 1
                    or cfg.training.sp_ulysses_size != 1 or cfg.training.sp_ring_size != 1):
                raise ValueError("concurrent family currently requires single-rank colocated offline runs")
    if plan:
        return {"output_dir": str(root), "jobs": jobs,
                "execution": "joint" if joint else ("concurrent" if jobs > 1 else "sequential"),
                **({"resume_from": resume_from, "stop_at_step": stop_at_step} if joint else {}),
                "runs": [_summary(stage) for stage in stages]}
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"family output already exists: {root}")
    root.mkdir(parents=True)
    if joint:
        from draftfit.training.joint_family import run_joint_family
        for stage in stages:
            output = Path(stage.config.output_dir)
            output.mkdir()
            stage.draft_config.to_json_file(str(output / "draft.json"))
            (output / "train.json").write_text(
                json.dumps(stage.config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
            )
        return run_joint_family(root, stages, resume_from=resume_from, stop_at_step=stop_at_step)
    if jobs > 1:
        return _run_concurrent(root, stages, jobs)
    results = []
    for stage in stages:
        output = Path(stage.config.output_dir)
        output.mkdir()
        stage.draft_config.to_json_file(str(output / "draft.json"))
        (output / "train.json").write_text(
            json.dumps(stage.config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        result = _summary(stage)
        result["status"] = "running"
        results.append(result)
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "draftfit.cli", "train", "-c", str(output / "train.json")],
            check=False,
        )
        result["status"] = "complete" if completed.returncode == 0 else "failed"
        result["exit_code"] = completed.returncode
        result["metrics"] = _last_metrics(output / "metrics.jsonl")
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if completed.returncode:
            break
        latest = output / f"{stage.name}-latest"
        if not (latest.is_dir() and (latest / "training_state.pt").is_file()):
            result["status"] = "failed"
            result["reason"] = "training produced no readable latest checkpoint"
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            break
        if "eval/avg_loss" not in result["metrics"] or "eval/avg_acc" not in result["metrics"]:
            result["status"] = "failed"
            result["reason"] = "training produced no held-out evaluation metrics"
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            break
    return {"output_dir": str(root), "runs": results}


def _run_concurrent(root: Path, stages: list[FamilyStage], jobs: int) -> dict:
    """Bound independent workers; a handoff starts only after its parent validates."""
    results = {stage.name: {**_summary(stage), "status": "queued"} for stage in stages}
    active = {}

    def signal_group(process, signum):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate() if signum == signal.SIGTERM else process.kill()

    def persist():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(list(results.values()), indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "results.json")

    def stop_workers():
        for process, _log in active.values():
            signal_group(process, signal.SIGTERM)
        deadline = time.monotonic() + 5
        for name, (process, log) in active.items():
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_group(process, signal.SIGKILL)
                process.wait()
            # A parent may exit while one of its loader children ignores SIGTERM.
            signal_group(process, signal.SIGKILL)
            log.close()
            results[name].update(status="cancelled", exit_code=process.returncode,
                                 finished_at=time.time())
        active.clear()
        for result in results.values():
            if result["status"] == "queued":
                result["status"] = "blocked"
        persist()

    try:
        for stage in stages:
            output = Path(stage.config.output_dir)
            output.mkdir()
            stage.draft_config.to_json_file(str(output / "draft.json"))
            (output / "train.json").write_text(
                json.dumps(stage.config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
            )
        persist()
        while any(result["status"] == "queued" for result in results.values()) or active:
            # Reap failures before allocating another training worker.
            failed = False
            for name, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                # A failed or successful worker must not leave background children.
                signal_group(process, signal.SIGKILL)
                log.close()
                del active[name]
                result = results[name]
                result.update(exit_code=code, finished_at=time.time(),
                              status="complete" if code == 0 else "failed")
                output = Path(result["output_dir"])
                result["metrics"] = _last_metrics(output / "metrics.jsonl")
                latest = output / f"{name}-latest"
                if code == 0 and not (latest.is_dir() and (latest / "training_state.pt").is_file()):
                    result.update(status="failed", reason="training produced no readable latest checkpoint")
                if code == 0 and not {"eval/avg_loss", "eval/avg_acc"} <= result["metrics"].keys():
                    result.update(status="failed", reason="training produced no held-out evaluation metrics")
                failed |= result["status"] == "failed"
                persist()
            if failed:
                stop_workers()
                break
            for stage in stages:
                result = results[stage.name]
                if len(active) >= jobs:
                    break
                if result["status"] != "queued":
                    continue
                if stage.parent is not None and results[stage.parent]["status"] != "complete":
                    continue
                output = Path(stage.config.output_dir)
                log = (output / "worker.log").open("wb")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-m", "draftfit.cli", "train", "-c", str(output / "train.json")],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"),
                    )
                except BaseException:
                    log.close()
                    result.update(status="failed", reason="training worker could not start")
                    raise
                active[stage.name] = process, log
                result.update(status="running", pid=process.pid, started_at=time.time(),
                              worker_log=str(output / "worker.log"))
                persist()
            if active:
                time.sleep(0.05)
    except BaseException:
        stop_workers()
        raise
    return {"output_dir": str(root), "jobs": jobs, "execution": "concurrent",
            "runs": list(results.values())}


def _last_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    last: dict = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if any(key.startswith("eval/") for key in row):
                last["eval_step"] = row["step"]
            if any(key.startswith("train/") for key in row):
                last["train_step"] = row["step"]
            last.update(row)
    return last
