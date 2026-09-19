"""Prepare editable draft-training projects using existing providers, not target models."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.application.project_validation import (  # noqa: F401 - legacy import
    validate_conversation_file as _validate_conversation_file,
)
from draftfit.application.project_validation import (
    validate_draft_metadata,
    validate_project_data,
)
from draftfit.config import SGLANG_CAPTURE_CONTEXT_HEADROOM, Config, apply_overrides
from draftfit.target_inspector import inspect_target
from draftfit.target_spec import TargetSpec


def algorithm_catalog():
    """Report registered capabilities without downloading or constructing models."""
    return [
        {
            "algorithm": item.name,
            "draft_architectures": sorted(item.spec.draft.compatible_architectures),
            "target_derived_config": item.providers.model.draft_config.target_defaults
            is not None,
            "draft_overrides": sorted(item.spec.draft.supported_overrides),
            "attention_backends": sorted(item.spec.capabilities.attention_backends),
            "features": [
                {
                    "mode": contract.mode.value,
                    "modality": contract.modality,
                    "required_tensors": sorted(contract.required_tensors),
                }
                for contract in item.spec.feature_contracts
            ],
            "support_scope": "registered implementation; target/backend GPU validation is separate",
        }
        for item in builtin_algorithm_registry()
    ]


def prepare_project(
    source: str,
    output_dir: str,
    *,
    strategy="dspark",
    revision="main",
    local_only=False,
    train_data=None,
    hidden_states=None,
    draft_config=None,
    draft_checkpoint=None,
    overrides=(),
    target_backend="sglang",
):
    """Validate metadata/configs first, then write into a NEW project directory.

    No weights, tokenizer execution, target modeling code, GPU allocation, or
    cloud resources are involved. Passing an existing draft preserves its shape
    unless the user explicitly overrides it. Stage evidence never implies GPU support.
    """
    from transformers import PretrainedConfig

    from draftfit.application.composition import resolve_run
    from draftfit.training.model_loading import resolve_draft_config

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"project already exists; refusing to overwrite {destination}"
        )
    if bool(train_data) == bool(hidden_states):
        raise ValueError("choose exactly one of --train-data or --hidden-states")
    if train_data and not Path(train_data).expanduser().is_file():
        raise ValueError(f"training data file does not exist: {train_data}")
    # Sources must be resolved before scaffold defaults are chosen.
    source_fields = {
        "model.draft_model_config",
        "model.draft_checkpoint_path",
        "data.train_data_path",
        "data.hidden_states_path",
    }
    if any(
        field == item.split("=", 1)[0] or field.startswith(item.split("=", 1)[0] + ".")
        for item in overrides
        for field in source_fields
    ):
        raise ValueError(
            "select draft and data sources through command arguments, not --set"
        )
    try:
        registration = builtin_algorithm_registry().resolve(strategy)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    inspection = inspect_target(source, revision=revision, local_only=local_only)
    spec = TargetSpec.from_inspection(inspection)
    is_local = Path(source).expanduser().is_dir()
    if not is_local and not re.fullmatch(r"[0-9a-fA-F]{40}", spec.revision):
        raise ValueError(
            "cannot resolve immutable target revision; check model access or pass a commit SHA"
        )

    raw = spec.scaffold_config(strategy=strategy)
    model = raw["model"]
    model["target_backend"] = target_backend
    if target_backend != "sglang":
        for key in list(model):
            if key.startswith("sglang_"):
                del model[key]
    model["target_revision"] = None if is_local else spec.revision
    model["draft_model_config"] = draft_config
    model["draft_checkpoint_path"] = draft_checkpoint
    if draft_config or draft_checkpoint:
        # Existing draft structure, not generic suggestions, is authoritative.
        for key in (
            "draft_num_hidden_layers",
            "draft_block_size",
            "target_layer_ids",
            "aux_hidden_state_layer_ids",
        ):
            model.pop(key, None)
    keys = inspection.get("weight_keys", {})
    embeddings = keys.get("embedding_candidates", [])
    heads = keys.get("head_candidates", [])
    if len(embeddings) == 1:
        model["embedding_key"] = embeddings[0]
    if len(heads) == 1:
        model["lm_head_key"] = heads[0]
    metadata = inspection["config"]
    text_metadata = metadata.get("text_config", metadata)
    if not heads and len(embeddings) == 1 and text_metadata.get("tie_word_embeddings"):
        model["lm_head_key"] = embeddings[0]
    raw["output_dir"] = str(destination / "checkpoints")
    raw["data"]["cache_dir"] = str(destination / "cache")
    raw["data"]["max_length"] = 1024
    raw["training"].update(
        max_steps=100, save_interval=25, attention_backend="flex_attention"
    )
    if hidden_states:
        raw["data"]["hidden_states_path"] = str(
            Path(hidden_states).expanduser().absolute()
        )
    else:
        raw["data"].pop("hidden_states_path", None)
        raw["data"]["train_data_path"] = str(Path(train_data).expanduser().absolute())
        if target_backend == "sglang":
            model.update(
                sglang_attention_backend="triton",
                sglang_linear_attn_backend="triton",
                sglang_linear_attn_verify_backend="triton",
                sglang_disable_cuda_graph=True,
                sglang_enable_deterministic_inference=True,
                sglang_disable_radix_cache=True,
                sglang_context_length=1024 + SGLANG_CAPTURE_CONTEXT_HEADROOM,
                sglang_max_total_tokens=1024 + SGLANG_CAPTURE_CONTEXT_HEADROOM,
                sglang_max_running_requests=1,
            )
        raw["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": str(destination / "control"),
                "backend": "mooncake",
                "managed_local": {
                    "trainer_cuda_visible_devices": ["1"],
                    "capture_servers": [
                        {"port": 30000, "cuda_visible_devices": ["0"], "tp_size": 1}
                    ],
                },
            },
        }
    cfg = apply_overrides(Config.model_validate(raw), list(overrides))
    if cfg.model.trust_remote_code:
        raise ValueError(
            "target prepare is metadata-only and requires model.trust_remote_code=false; "
            "supply a local registered draft config, then explicitly enable remote code "
            "in train.json for training if required"
        )
    if (
        cfg.training.strategy != strategy
        or cfg.model.target_model_path != spec.model_id
        or cfg.model.target_revision != model["target_revision"]
    ):
        raise ValueError(
            "select target, revision and strategy through their command arguments, not --set"
        )
    data_rows = validate_project_data(cfg.data, inspection) if train_data else None
    resolved = resolve_run(cfg)
    target = PretrainedConfig.from_dict(text_metadata)
    draft = resolve_draft_config(
        cfg, provider=registration.providers.model.draft_config, target_config=target
    )
    layers = list(
        registration.providers.model.resolve_capture_layers(cfg, draft, target)
    )
    validate_draft_metadata(
        strategy=strategy,
        target_depth=spec.num_hidden_layers,
        target_metadata=text_metadata,
        draft=draft,
        layers=layers,
        mask_token_id=cfg.model.mask_token_id,
    )
    # Source paths remain explicit even when the CLI is invoked from another CWD.
    final_config = cfg.model_dump(mode="json")
    final_config["model"]["draft_model_config"] = str(destination / "draft.json")
    checkpoint_source = cfg.model.draft_checkpoint_path
    if checkpoint_source and Path(checkpoint_source).expanduser().exists():
        final_config["model"]["draft_checkpoint_path"] = str(
            Path(checkpoint_source).expanduser().absolute()
        )
    contract_mode = "streaming" if train_data else "offline"
    contract = next(
        c
        for c in registration.spec.feature_contracts
        if c.mode.value == contract_mode and c.modality == cfg.model.input_modality
    )
    metadata_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "draft_project_v1",
        "target": spec.to_dict(),
        "target_config_sha256": metadata_hash,
        "algorithm": resolved.algorithm.name,
        "capture_layers": layers,
        "required_features": sorted(contract.required_tensors),
        "initialization": "weights_only" if checkpoint_source else "from_scratch",
        "input_validation": {
            "conversation_rows": data_rows,
            "tokenizer_masks": "not_run",
        },
        "validation": {
            "configuration": "passed",
            "capture": "not_run",
            "training": "not_run",
            "export": "not_run",
            "serving": "not_run",
        },
        "required_gates": list(spec.recommendations.get("required_gates", [])),
        "notes": [
            "Configuration validation is not target/backend compatibility certification.",
            "Verify embedding/head names, tokenizer masks and feature normalization before training.",
            "Generic hf rendering requires native assistant-generation spans; otherwise choose an explicit renderer.",
            "Inference backends own target architecture execution; this project does not implement target models.",
        ],
    }
    # Exclusive create preserves all prior projects. Leave any partial write visible
    # on I/O failure rather than deleting a directory another process may be reading.
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in (
        ("train.json", final_config),
        ("draft.json", draft.to_dict()),
        ("manifest.json", manifest),
        ("inspection.json", inspection),
    ):
        with (destination / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return {
        "project": str(destination),
        "config": str(destination / "train.json"),
        "manifest": str(destination / "manifest.json"),
        "validation": manifest["validation"],
    }
