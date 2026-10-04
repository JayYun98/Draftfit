"""One offline feature cache, independent DFlash-family runs and handoffs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.config import Config, apply_overrides, load_config
from draftfit.training.model_loading import (
    convert_family_draft_config, load_draft_config_source, resolve_draft_config,
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
    if not base.data.hidden_states_path or not base.data.eval_hidden_states_path or not base.training.eval_interval:
        raise ValueError("family runs require offline train and held-out eval features with training.eval_interval")
    train_path = Path(base.data.hidden_states_path).resolve()
    eval_path = Path(base.data.eval_hidden_states_path).resolve()
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
                 "model.draft_model_config", "data.hidden_states_path", "data.eval_hidden_states_path"}
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
            if draft.architectures != [registration.providers.model.draft_config.architecture]:
                raise ValueError(f"family run {name!r} draft_config architecture differs from strategy")
        elif parent is not None:
            draft = convert_family_draft_config(
                by_name[parent].draft_config, strategy,
                retrain_anchor_alignment=cfg.training.retrain_anchor_alignment,
            )
        else:
            draft = resolve_draft_config(cfg, provider=registration.providers.model.draft_config)
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

        resolve_run(cfg)
        keys = ("input_ids", "loss_mask", "hidden_states")
        if strategy == "dspark":
            keys += ("target_last_hidden_states",)
        capture_versions = []
        for feature_path in (cfg.data.hidden_states_path, cfg.data.eval_hidden_states_path):
            reader = OfflineManifestReader(feature_path, strategy=strategy, feature_keys=keys)
            if reader.capture_manifest is not None:
                capture = reader.capture_manifest
                capture_versions.append(capture["tokenizer"]["version"])
                if capture["target"]["model"] != cfg.model.target_model_path:
                    raise ValueError(f"family run {name!r} target differs from capture manifest")
                if cfg.model.target_revision and capture["target"]["revision"] != cfg.model.target_revision:
                    raise ValueError(f"family run {name!r} target revision differs from capture manifest")
                if capture["contract"]["capture_layers"] != list(method.get("target_layer_ids", [])):
                    raise ValueError(f"family run {name!r} target layers differ from capture manifest")
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
            "eval_features": stage.config.data.eval_hidden_states_path}


def run_family(manifest_path: str, *, plan: bool = False) -> dict:
    root, stages = build_family_plan(manifest_path)
    if plan:
        return {"output_dir": str(root), "runs": [_summary(stage) for stage in stages]}
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"family output already exists: {root}")
    root.mkdir(parents=True)
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
