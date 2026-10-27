# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draft configuration and weights-only initialization for unified training.

This module deliberately keeps model initialization separate from training
resume.  A warm start can read draft weights from a Hugging Face checkpoint or
from SpecForge's shared ``training_state.pt`` payload, but it has no access to
an optimizer, scheduler, controller counters, or RNG state.
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from draftfit.algorithms.common.providers import DraftConfigProvider
    from draftfit.config import Config

logger = logging.getLogger(__name__)

_CONFIG_FILE = "config.json"
_STATE_FILE = "training_state.pt"

# These are the architecture fields the legacy EAGLE generator copied from the
# target, plus RoPE/attention fields needed by newer Qwen-family targets.  A
# whitelist avoids feeding a target-specific nested config into the registered
# Llama/Qwen3 draft classes.
_TARGET_ARCHITECTURE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
)


@dataclass(frozen=True)
class WarmStartReport:
    """Observable result of loading draft weights without training state."""

    source: str
    checkpoint_format: Literal["specforge", "pretrained"]
    loaded_keys: int
    missing_keys: Tuple[str, ...]
    loaded_embedding: bool


def _without_file_uri(source: str) -> str:
    return source[len("file://") :] if source.startswith("file://") else source


def _looks_like_local_path(source: str) -> bool:
    return (
        source.startswith((".", "/", "~", "file://"))
        or source.endswith(".json")
        or os.path.exists(os.path.expanduser(_without_file_uri(source)))
    )


def _draft_config_from_dict(
    payload: Dict[str, Any],
    *,
    target_config: Any = None,
    draft_capture_layer_indexing: Optional[str] = None,
) -> "PretrainedConfig":
    # Importing the draft package registers every built-in architecture before
    # consulting the registry. Model modules keep accelerator-only dependencies
    # behind execution boundaries, so this remains safe in a CPU producer.
    import draftfit.modeling.draft  # noqa: F401
    from draftfit.modeling.draft.registry import DRAFT_REGISTRY, available_drafts

    payload = _normalize_published_dspark_config(
        payload,
        target_config=target_config,
        draft_capture_layer_indexing=draft_capture_layer_indexing,
    )
    architectures = payload.get("architectures") or []
    if not isinstance(architectures, (list, tuple)) or len(architectures) != 1:
        raise ValueError(
            "draft config must name exactly one architecture; " f"got {architectures!r}"
        )
    architecture = architectures[0]
    if architecture == "Qwen3DSparkModel":
        # Published DSpark checkpoints keep method fields at the JSON root.
        # Fold them into our registered architecture without touching weights.
        method = dict(payload.get("dflash_config") or {})
        for field in (
            "target_layer_ids",
            "mask_token_id",
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "confidence_head_alpha",
            "sample_from_anchor",
            "shift_label",
            "pure_draft_prefix_len",
            "projector_type",
            "attention_mode",
        ):
            if field in payload:
                if field in method and method[field] != payload[field]:
                    raise ValueError(f"conflicting DSpark config field {field}")
                method[field] = payload.pop(field)
        method.setdefault("sample_from_anchor", True)
        method.setdefault("projector_type", "dspark")
        if method["projector_type"] != "dspark":
            raise ValueError("published DSpark config requires projector_type=dspark")
        payload["dflash_config"] = method
        payload["architectures"] = ["DSparkDraftModel"]
        architecture = "DSparkDraftModel"
    if architecture not in DRAFT_REGISTRY:
        raise ValueError(
            f"draft architecture {architecture!r} is not registered; "
            f"available: {available_drafts()}"
        )

    payload["architectures"] = [architecture]
    payload["tie_word_embeddings"] = False
    payload.setdefault("use_cache", True)
    if payload.get("draft_vocab_size") is None:
        payload["draft_vocab_size"] = payload.get("vocab_size")
    config = DRAFT_REGISTRY[architecture].config_class.from_dict(payload)
    _validate_published_speculators_semantics(config)
    return config


def _published_capture_layer_ids(
    payload: Dict[str, Any],
    *,
    declaration: Optional[str] = None,
    source_ids: Any = None,
) -> list[int]:
    """Resolve declared source slots to physical blocks, retaining provenance."""
    stored = payload.get("draft_capture_layer_indexing")
    for value in (declaration, stored):
        if value is not None and value not in ("block_output", "embedding_inclusive"):
            raise ValueError(f"unsupported draft_capture_layer_indexing: {value!r}")
    if declaration is not None and stored is not None and declaration != stored:
        raise ValueError("conflicting draft_capture_layer_indexing in source config")
    indexing = declaration or stored
    if indexing is None:
        raise ValueError(
            "generic speculators requires explicit draft_capture_layer_indexing"
        )
    saved_ids = payload.get("draft_capture_source_layer_ids")
    if source_ids is None:
        source_ids = saved_ids
    for ids in (source_ids,) if saved_ids is None else (source_ids, saved_ids):
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(index) is not int or index < 0 for index in ids)
        ):
            raise ValueError(
                "speculators capture provenance requires nonnegative source layer IDs"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("speculators source layer IDs must not contain duplicates")
    if saved_ids is not None and saved_ids != source_ids:
        raise ValueError("conflicting speculators capture source layer IDs")
    if indexing == "embedding_inclusive" and 0 in source_ids:
        raise ValueError("embedding_inclusive capture slot 0 is unsupported")
    payload["draft_capture_source_layer_ids"] = list(source_ids)
    payload["draft_capture_layer_indexing"] = indexing
    return [index - int(indexing == "embedding_inclusive") for index in source_ids]


def _validate_published_capture_layers(
    payload: Dict[str, Any], *, declaration: Optional[str] = None
) -> None:
    physical = _published_capture_layer_ids(payload, declaration=declaration)
    configured = (payload.get("dflash_config") or {}).get("target_layer_ids")
    if (
        not isinstance(configured, list)
        or any(type(index) is not int for index in configured)
        or configured != physical
    ):
        raise ValueError("speculators target_layer_ids differs from capture provenance")
    depth = payload.get("target_num_hidden_layers") or payload.get("num_target_layers")
    if (
        type(depth) is not int
        or depth <= 0
        or any(index >= depth for index in physical)
    ):
        raise ValueError(
            "speculators capture layers require valid target depth metadata"
        )


def _normalize_published_dspark_config(
    payload: Dict[str, Any],
    *,
    target_config: Any = None,
    draft_capture_layer_indexing: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten registered speculators configs without guessing target geometry."""
    from draftfit.training.official_dspark import is_taichu_config

    payload = copy.deepcopy(payload)
    architecture = (payload.get("architectures") or [None])[0]
    taichu = is_taichu_config(payload)
    if taichu and any(
        value not in (None, "block_output")
        for value in (
            draft_capture_layer_indexing,
            payload.get("draft_capture_layer_indexing"),
        )
    ):
        raise ValueError("Taichu capture indexing must preserve block_output semantics")
    if (
        architecture in {"DSparkDraftModel", "DFlashDraftModel"}
        and "transformer_layer_config" in payload
    ):
        transformer = payload.pop("transformer_layer_config")
        if transformer.get("model_type") != "qwen3":
            raise ValueError("unsupported nested speculators transformer model_type")
        for field, value in transformer.items():
            if field in payload and payload[field] != value:
                raise ValueError(f"conflicting nested speculators field {field}")
            payload[field] = value
        if payload.get("draft_vocab_size") not in (None, payload["vocab_size"]):
            raise ValueError(
                "nested speculators reduced draft vocabulary is unsupported"
            )
        method = dict(payload.get("dflash_config") or {})

        def put(field, value):
            if field in method and method[field] != value:
                raise ValueError(f"conflicting speculators config field {field}")
            method[field] = value

        taps = payload.pop("aux_hidden_state_layer_ids")
        if "target_layer_ids" in payload and payload.pop("target_layer_ids") != taps:
            raise ValueError("conflicting speculators config field target_layer_ids")
        if not taichu:
            taps = _published_capture_layer_ids(
                payload, declaration=draft_capture_layer_indexing, source_ids=taps
            )
        put("target_layer_ids", taps)
        for field in (
            "mask_token_id",
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "confidence_head_alpha",
            "sample_from_anchor",
            "shift_label",
            "pure_draft_prefix_len",
            "projector_type",
            "sliding_window_non_causal",
            "domino_auf",
            "lambda_base_start",
            "lambda_base_decay_ratio",
            "emb_dim",
            "gru_hidden_dim",
        ):
            if field in payload:
                put(field, payload.pop(field))
        family = "dspark" if architecture == "DSparkDraftModel" else "dflash"
        method.setdefault("projector_type", family)
        method.setdefault("sample_from_anchor", family == "dspark")
        if family == "dspark":
            method.setdefault("attention_mode", "gqa")
        if method["projector_type"] != family or method.get("domino_auf", False):
            raise ValueError("unsupported nested speculators projector semantics")
        if family == "dflash" and (
            method.get("sample_from_anchor") or method.get("shift_label", False)
        ):
            raise ValueError("unsupported nested DFlash anchor/label semantics")
        depth = payload.get("num_target_layers")
        if target_config is not None:
            from draftfit.modeling.target.target_utils import target_text_config

            text = target_text_config(target_config)
            actual_depth = (
                text.get("num_hidden_layers")
                if isinstance(text, dict)
                else getattr(text, "num_hidden_layers", None)
            )
            if depth is not None and depth != actual_depth:
                raise ValueError(
                    "speculators num_target_layers differs from target metadata"
                )
            depth = actual_depth
            for field in ("hidden_size", "vocab_size"):
                actual = (
                    text.get(field)
                    if isinstance(text, dict)
                    else getattr(text, field, None)
                )
                if actual is not None and actual != payload[field]:
                    raise ValueError(
                        f"speculators {field} differs from target metadata"
                    )
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError(
                "nested speculators requires target depth metadata; last auxiliary tap is not target depth"
            )
        if not taps or any(
            not isinstance(tap, int) or isinstance(tap, bool) or tap < 0 or tap >= depth
            for tap in taps
        ):
            raise ValueError(
                "speculators auxiliary taps are outside target depth metadata"
            )
        payload["num_target_layers"] = depth
        payload["dflash_config"] = method
        payload["auto_map"] = {"AutoModel": f"{family}.{architecture}"}
        if taichu:
            payload["official_dspark_format"] = "taichu_speculators"
        else:
            payload["published_speculators_format"] = True
            if (
                payload.get("speculators_attention_semantics", "anchor_relative")
                != "anchor_relative"
            ):
                raise ValueError("conflicting speculators attention semantics")
            payload["speculators_attention_semantics"] = "anchor_relative"
        # Preserve verifier provenance; the loading path binds it to the actual target.
    elif architecture == "Lfm2DSparkDraftModel":
        # The published LFM speculator is a Qwen3-shaped draft with a
        # different rotary pairing, not the LFM target's conv/MoE network.
        payload["architectures"] = ["DSparkDraftModel"]
        method = dict(payload.get("dflash_config") or {})
        payload["num_target_layers"] = method.pop("num_target_layers")
        method.update(
            projector_type="dspark",
            markov_rank=payload.pop("markov_rank"),
            markov_head_type=payload.pop("markov_head_type"),
            enable_confidence_head=payload.pop("enable_confidence_head"),
            confidence_head_with_markov=True,
            sample_from_anchor=True,
        )
        payload["dflash_config"] = method
        payload["auto_map"] = {"AutoModel": "dspark.DSparkDraftModel"}
        payload["official_dspark_format"] = "lfm2"
    if payload.get("published_speculators_format") is True:
        _validate_published_capture_layers(
            payload, declaration=draft_capture_layer_indexing
        )
    return payload


def load_draft_config_source(
    source: str,
    *,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    target_config: Any = None,
    target_model_path: Optional[str] = None,
    target_revision: Optional[str] = None,
    draft_capture_layer_indexing: Optional[str] = None,
) -> "PretrainedConfig":
    """Load a registered draft config from JSON, a directory, or an HF repo."""

    if not source:
        raise ValueError("draft config source must be non-empty")
    local_source = os.path.abspath(os.path.expanduser(_without_file_uri(str(source))))
    if os.path.isfile(local_source):
        config_path = local_source
    elif os.path.isdir(local_source):
        config_path = os.path.join(local_source, _CONFIG_FILE)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"draft model directory has no {_CONFIG_FILE}: {local_source}"
            )
    else:
        if _looks_like_local_path(str(source)):
            raise FileNotFoundError(f"draft model config does not exist: {source}")
        if source.startswith(
            (
                "TaichuAI/ZDTaichu5.0-9B-DSpark",
                "LiquidAI/LFM2.5-",
                "wkk66d/Qwen3-0.6B-DSpark",
                "orestis-z/dflash-qwen3-0.6b-microcycle-dflash",
            )
        ):
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=source, filename=_CONFIG_FILE, cache_dir=cache_dir
            )
        else:
            from transformers import AutoConfig

            loaded = AutoConfig.from_pretrained(
                source, cache_dir=cache_dir, trust_remote_code=trust_remote_code
            )
            return _draft_config_from_dict(
                loaded.to_dict(),
                target_config=target_config,
                draft_capture_layer_indexing=draft_capture_layer_indexing,
            )

    try:
        with open(config_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid draft config JSON {config_path}: {exc}") from exc
    if "transformer_layer_config" in payload:
        from draftfit.training.official_dspark import is_taichu_config

        verifier = (payload.get("speculators_config") or {}).get("verifier") or {}
        target_model_path = target_model_path or verifier.get("name_or_path")
        if (
            not target_model_path
            and is_taichu_config(payload)
            and payload.get("num_target_layers") is None
        ):
            target_model_path = "TaichuAI/ZDTaichu5.0-9B"
        if target_config is None and target_model_path:
            from draftfit.modeling.target.target_utils import load_target_config

            target_config = load_target_config(
                target_model_path,
                cache_dir=cache_dir,
                trust_remote_code=False,
                revision=target_revision,
            )
    config = _draft_config_from_dict(
        payload,
        target_config=target_config,
        draft_capture_layer_indexing=draft_capture_layer_indexing,
    )
    if target_model_path:
        config.speculators_target_model_path = target_model_path
        config.speculators_target_revision = target_revision
    return config


def _validate_published_speculators_semantics(config) -> None:
    if getattr(config, "published_speculators_format", False) is not True:
        return
    if getattr(config, "speculators_attention_semantics", None) != "anchor_relative":
        raise ValueError(
            "published speculators requires anchor_relative attention metadata"
        )
    if config.architectures not in (
        ["DFlashDraftModel"],
        ["DSparkDraftModel"],
        ["DFlash2DraftModel"],
    ):
        raise ValueError(
            "published speculators attention requires native DFlash/DSpark/DFlash2"
        )
    _validate_published_capture_layers(config.to_dict())
    if config.architectures == ["DFlash2DraftModel"] and (
        "sliding_attention" in (getattr(config, "layer_types", None) or [])
        or getattr(config, "is_causal", False)
    ):
        # Native DFlash2 overrides both mask-option methods and currently uses
        # query-relative windows. Only its non-causal full mask matches here.
        raise ValueError(
            "native DFlash2 cannot preserve published anchor_relative sliding/causal masks"
        )


def _validate_warm_start_capture_provenance(source_config, draft_config) -> None:
    source_published = (
        getattr(source_config, "published_speculators_format", False) is True
    )
    draft_published = (
        getattr(draft_config, "published_speculators_format", False) is True
    )
    if not (source_published or draft_published):
        return
    if not (source_published and draft_published):
        raise ValueError("warm start requires matching published capture provenance")
    for config in (source_config, draft_config):
        _validate_published_speculators_semantics(config)
    for field in ("draft_capture_layer_indexing", "draft_capture_source_layer_ids"):
        if getattr(source_config, field) != getattr(draft_config, field):
            raise ValueError(f"warm start has incompatible capture provenance {field}")
    if (
        source_config.dflash_config["target_layer_ids"]
        != draft_config.dflash_config["target_layer_ids"]
    ):
        raise ValueError("warm start has incompatible capture target_layer_ids")


def _checkpoint_config_source(checkpoint_source: Optional[str]) -> Optional[str]:
    """Return a config-bearing warm-start source, if it can be identified."""

    if not checkpoint_source:
        return None
    raw_source = str(checkpoint_source)
    local_source = os.path.abspath(os.path.expanduser(_without_file_uri(raw_source)))
    if os.path.isfile(local_source):
        if os.path.basename(local_source) == _CONFIG_FILE:
            return local_source
        sibling = os.path.join(os.path.dirname(local_source), _CONFIG_FILE)
        return sibling if os.path.isfile(sibling) else None
    if os.path.isdir(local_source):
        direct = os.path.join(local_source, _CONFIG_FILE)
        if os.path.isfile(direct):
            return direct
        # A run root may point at a future checkpoint layout that also persists
        # config.json.  Do not resolve or read training state just to find it.
        candidates = []
        for pointer in glob.glob(os.path.join(local_source, "*-latest")):
            candidate = os.path.join(os.path.realpath(pointer), _CONFIG_FILE)
            if os.path.isfile(candidate):
                candidates.append(candidate)
        if len(set(candidates)) == 1:
            return candidates[0]
        return None
    if _looks_like_local_path(raw_source):
        return None
    # A non-local warm-start source is an HF repository. Its config is the best
    # architecture contract unless the run supplies an explicit config source.
    return raw_source


def _target_text_config(config: "PretrainedConfig") -> "PretrainedConfig":
    return getattr(config, "text_config", config)


def _serializable_config_value(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
    except ImportError:
        pass
    return value


def _generate_draft_config(
    cfg: "Config", provider: "DraftConfigProvider", *, target_config: Any = None
) -> "PretrainedConfig":
    defaults = provider.target_defaults
    if defaults is None:
        raise ValueError(
            f"training.strategy={cfg.training.strategy!r} requires "
            "model.draft_model_config "
            "or a pretrained model.draft_checkpoint_path containing config.json"
        )

    from draftfit.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
        target_vocab_size,
    )

    if target_config is None:
        target_config = load_target_config(
            cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
            revision=cfg.model.target_revision,
        )
    target_config = target_text_config(target_config)
    if isinstance(target_config, dict):
        from transformers import PretrainedConfig

        target_config = PretrainedConfig.from_dict(target_config)
    payload: Dict[str, Any] = {}
    for field in _TARGET_ARCHITECTURE_FIELDS:
        value = getattr(target_config, field, None)
        if value is not None:
            payload[field] = _serializable_config_value(value)
    payload["vocab_size"] = target_vocab_size(target_config)

    required = [name for name in ("vocab_size", "hidden_size") if name not in payload]
    if required:
        raise ValueError(
            f"target config {cfg.model.target_model_path!r} cannot generate a "
            f"draft config; missing {required}"
        )
    payload["architectures"] = [provider.architecture]
    payload["model_type"] = defaults.model_type
    payload["num_hidden_layers"] = defaults.num_hidden_layers
    payload["tie_word_embeddings"] = False
    payload["use_cache"] = True
    if payload.get("pad_token_id") is None:
        payload["pad_token_id"] = 0
    if defaults.draft_vocab_size is not None:
        payload["draft_vocab_size"] = defaults.draft_vocab_size
    if defaults.populate is not None:
        defaults.populate(payload, target_config, cfg)
    return _draft_config_from_dict(payload)


def _apply_draft_overrides(
    cfg: "Config",
    draft_config: "PretrainedConfig",
    provider: "DraftConfigProvider",
) -> None:
    published = getattr(draft_config, "published_speculators_format", False) is True
    if published:
        _validate_published_speculators_semantics(draft_config)
        _validate_published_capture_layers(
            draft_config.to_dict(),
            declaration=cfg.model.draft_capture_layer_indexing,
        )
        if cfg.model.target_layer_ids is not None and cfg.model.target_layer_ids != (
            draft_config.dflash_config["target_layer_ids"]
        ):
            raise ValueError(
                "model.target_layer_ids conflicts with source capture provenance"
            )
    num_layers = cfg.model.draft_num_hidden_layers
    if num_layers is not None:
        draft_config.num_hidden_layers = num_layers
    if cfg.model.draft_block_size is not None:
        draft_config.block_size = cfg.model.draft_block_size
    if provider.apply_overrides is not None:
        provider.apply_overrides(cfg, draft_config)
    if published:
        _validate_published_speculators_semantics(draft_config)


def convert_family_draft_config(
    draft_config: "PretrainedConfig",
    strategy: str,
    *,
    retrain_anchor_alignment: bool = False,
    mask_token_id: Optional[int] = None,
) -> "PretrainedConfig":
    """Keep shared DFlash weights/layout while changing the objective head."""
    draft_config = copy.deepcopy(draft_config)
    if mask_token_id is not None:
        method = dict(getattr(draft_config, "dflash_config", None) or {})
        if method.get("mask_token_id") not in (None, mask_token_id):
            raise ValueError("conflicting mask_token_id in DFlash-family source config")
        method["mask_token_id"] = mask_token_id
        draft_config.dflash_config = method
    architecture = draft_config.architectures[0]
    if architecture in {
        "DFlashDraftModel",
        "DSparkDraftModel",
        "DFlash2DraftModel",
        "TaichuDFlash2DraftModel",
    }:
        if getattr(draft_config, "target_hidden_size", None) is None:
            draft_config.target_hidden_size = draft_config.hidden_size
    if architecture in {
        "DFlashDraftModel",
        "DFlash2DraftModel",
        "TaichuDFlash2DraftModel",
    }:
        method = dict(getattr(draft_config, "dflash_config", None) or {})
        method.setdefault("sample_from_anchor", False)
        draft_config.dflash_config = method
    if strategy == "dflash2" and architecture == "DSparkDraftModel":
        return convert_family_draft_config(
            convert_family_draft_config(
                draft_config,
                "dflash",
                retrain_anchor_alignment=retrain_anchor_alignment,
            ),
            strategy,
        )
    if strategy == "dspark" and architecture in {
        "DFlash2DraftModel",
        "TaichuDFlash2DraftModel",
    }:
        return convert_family_draft_config(
            convert_family_draft_config(draft_config, "dflash"), strategy
        )
    if strategy == "dflash2" and architecture == "DFlashDraftModel":
        payload = draft_config.to_dict()
        if (
            payload.get("draft_vocab_size", payload["vocab_size"])
            != payload["vocab_size"]
        ):
            raise ValueError("DFlash2 requires draft_vocab_size equal to vocab_size")
        method = dict(payload.get("dflash_config") or {})
        if method.get("sample_from_anchor", False):
            raise ValueError("DFlash2 transfer requires sample_from_anchor=false")
        if method.get("mask_token_id") is None:
            raise ValueError(
                "DFlash2 transfer requires a mask_token_id in the source config"
            )
        taichu_text = (
            payload.get("official_dspark_format") == "taichu_speculators"
        ) or (
            "mrope_section" in (payload.get("rope_parameters") or {})
            and len(set(payload.get("layer_types") or [])) > 1
        )
        payload["architectures"] = [
            "TaichuDFlash2DraftModel" if taichu_text else "DFlash2DraftModel"
        ]
        payload["model_type"] = "qwen3_dflash2"
        payload["auto_map"] = {"AutoModel": "dflash2.DFlash2DraftModel"}
        payload["target_num_hidden_layers"] = payload.get(
            "target_num_hidden_layers"
        ) or payload.get("num_target_layers")
        method["sample_from_anchor"] = False
        payload["dflash_config"] = method
        return _draft_config_from_dict(payload)
    if strategy == "dflash" and architecture in {
        "DFlash2DraftModel",
        "TaichuDFlash2DraftModel",
    }:
        payload = draft_config.to_dict()
        payload["architectures"] = ["DFlashDraftModel"]
        payload["model_type"] = "qwen3"
        payload["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}
        payload["num_target_layers"] = payload.get(
            "target_num_hidden_layers"
        ) or payload.get("num_target_layers")
        method = dict(payload.get("dflash_config") or {})
        method["sample_from_anchor"] = False
        payload["dflash_config"] = method
        return _draft_config_from_dict(payload)
    if strategy == "dspark" and architecture == "DFlashDraftModel":
        payload = draft_config.to_dict()
        payload["architectures"] = ["DSparkDraftModel"]
        payload["auto_map"] = {"AutoModel": "dspark.DSparkDraftModel"}
        payload["dflash_config"] = {
            **dict(payload.get("dflash_config") or {}),
            "projector_type": "dspark",
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "confidence_head_alpha": 1.0,
            "sample_from_anchor": False,
        }
        return _draft_config_from_dict(payload)
    if strategy == "dflash" and architecture == "DSparkDraftModel":
        payload = draft_config.to_dict()
        if (payload.get("dflash_config") or {}).get(
            "sample_from_anchor", True
        ) and not retrain_anchor_alignment:
            raise ValueError(
                "DSpark-to-DFlash warm start requires sample_from_anchor=false"
            )
        payload["architectures"] = ["DFlashDraftModel"]
        payload["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}
        method = dict(payload.get("dflash_config") or {})
        for field in (
            "projector_type",
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "confidence_head_alpha",
        ):
            method.pop(field, None)
        method["sample_from_anchor"] = False
        payload["dflash_config"] = method
        return _draft_config_from_dict(payload)
    expected = {
        "dflash": {"DFlashDraftModel"},
        "dspark": {"DSparkDraftModel"},
        "dflash2": {"DFlash2DraftModel", "TaichuDFlash2DraftModel"},
    }.get(strategy, set())
    if architecture in expected:
        return draft_config
    raise ValueError(
        f"unsupported DFlash-family transfer: {architecture} -> {strategy}"
    )


def resolve_draft_config(
    cfg: "Config", *, provider: "DraftConfigProvider", target_config: Any = None
) -> "PretrainedConfig":
    """Resolve and validate the draft architecture for one typed run config."""

    source = cfg.model.draft_model_config
    if not source:
        source = _checkpoint_config_source(cfg.model.draft_checkpoint_path)
    if source:
        draft_config = load_draft_config_source(
            source,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
            target_config=target_config,
            target_model_path=cfg.model.target_model_path,
            target_revision=cfg.model.target_revision,
            draft_capture_layer_indexing=cfg.model.draft_capture_layer_indexing,
        )
        _validate_published_speculators_semantics(draft_config)
        if not cfg.model.draft_model_config and cfg.training.strategy in (
            "dflash",
            "dspark",
            "dflash2",
        ):
            draft_config = convert_family_draft_config(
                draft_config,
                cfg.training.strategy,
                retrain_anchor_alignment=cfg.training.retrain_anchor_alignment,
                mask_token_id=cfg.model.mask_token_id,
            )
    else:
        draft_config = _generate_draft_config(
            cfg, provider, target_config=target_config
        )

    expected = provider.architecture
    compatible = provider.compatible_architectures or frozenset({expected})
    architectures = list(getattr(draft_config, "architectures", None) or [])
    if len(architectures) != 1 or architectures[0] not in compatible:
        raise ValueError(
            f"training.strategy={cfg.training.strategy!r} requires draft "
            f"architecture in {sorted(compatible)!r}, got {architectures!r}"
        )
    _apply_draft_overrides(cfg, draft_config, provider)
    return draft_config


def draft_config_dict(
    cfg: "Config", *, provider: "DraftConfigProvider"
) -> Dict[str, Any]:
    """JSON-compatible resolved draft config for capture-only producer roles."""

    return resolve_draft_config(cfg, provider=provider).to_dict()


def _has_pretrained_weights(path: str) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.isfile(os.path.join(path, name)) for name in names)


def _runtime_state_file(source: str) -> Optional[str]:
    """Resolve SpecForge checkpoint storage without loading resume state."""

    path = os.path.abspath(os.path.expanduser(_without_file_uri(str(source))))
    if os.path.isfile(path):
        return path if os.path.basename(path) == _STATE_FILE else None
    if not os.path.isdir(path):
        return None

    direct = os.path.join(path, _STATE_FILE)
    if os.path.isfile(direct) and not _has_pretrained_weights(path):
        return direct
    has_run_pointers = bool(
        glob.glob(os.path.join(path, "*-latest"))
        or glob.glob(os.path.join(path, "*-step*"))
    )
    if not has_run_pointers:
        return None

    from draftfit.training.checkpoint import CheckpointManager

    checkpoint_dir = CheckpointManager.resolve_resume_dir(path)
    return os.path.join(checkpoint_dir, _STATE_FILE)


def _load_specforge_draft_state(
    state_path: str,
    *,
    expected_strategy: str,
    draft_config: "PretrainedConfig",
    retrain_anchor_alignment: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    import torch

    try:
        # The restricted unpickler accepts tensors and primitive metadata while
        # refusing arbitrary checkpoint objects. Only draft_state_dict is used.
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"cannot read weights-only SpecForge checkpoint {state_path}: {exc}"
        ) from exc
    if not isinstance(state, dict) or not isinstance(
        state.get("draft_state_dict"), dict
    ):
        raise ValueError(
            f"SpecForge warm-start checkpoint has no draft_state_dict: {state_path}"
        )
    saved_strategy = state.get("strategy")
    family = {"dflash", "dspark", "dflash2"}
    cross_family = (
        saved_strategy in family
        and expected_strategy in family
        and saved_strategy != expected_strategy
    )
    dflash_objective_change = {saved_strategy, expected_strategy} <= {
        "dflash",
        "alr_ira",
    } and getattr(draft_config, "architectures", None) == ["DFlashDraftModel"]
    if (
        saved_strategy is not None
        and saved_strategy != expected_strategy
        and not cross_family
        and not dflash_objective_change
    ):
        raise ValueError(
            f"warm-start checkpoint {state_path} was written by strategy "
            f"{saved_strategy!r}; this run trains {expected_strategy!r}"
        )
    if cross_family:
        method = dict(getattr(draft_config, "dflash_config", None) or {})
        if method.get("sample_from_anchor") is not False:
            raise ValueError("DFlash-family transfer requires sample_from_anchor=false")
        prefix = saved_strategy
        expected = {
            f"{prefix}_block_size": int(draft_config.block_size),
            f"{prefix}_target_layer_ids": tuple(method.get("target_layer_ids", ())),
            f"{prefix}_mask_token_id": method.get("mask_token_id"),
            f"{prefix}_draft_num_hidden_layers": int(draft_config.num_hidden_layers),
        }
        if saved_strategy == "dspark":
            expected["dspark_sample_from_anchor"] = (
                True if retrain_anchor_alignment else False
            )
        for field, value in expected.items():
            actual = state.get(field)
            if field == "dspark_sample_from_anchor" and actual is None:
                actual = True  # older DSpark checkpoints used the default
            if actual != value:
                raise ValueError(f"DFlash-family checkpoint has incompatible {field}")
    weights = state["draft_state_dict"]
    if cross_family:
        weights = {
            key: value
            for key, value in weights.items()
            if not _source_only_family_key(key, expected_strategy)
        }
    return weights, cross_family


def _source_only_family_key(key: str, strategy: str) -> bool:
    if strategy != "dspark" and key.startswith(("markov_head.", "confidence_head.")):
        return True
    return strategy != "dflash2" and (
        key.startswith("candidate_selector.")
        or ".attention_conv." in key
        or ".mlp_conv." in key
    )


def _load_pretrained_draft_state(
    source: str,
    *,
    draft_config: "PretrainedConfig",
    cache_dir: Optional[str],
    trust_remote_code: bool,
    strategy: str,
    retrain_anchor_alignment: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    from draftfit.modeling.auto import AutoDraftModel

    source_config = load_draft_config_source(
        source,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        target_model_path=getattr(draft_config, "speculators_target_model_path", None),
        target_revision=getattr(draft_config, "speculators_target_revision", None),
        draft_capture_layer_indexing=getattr(
            draft_config, "draft_capture_layer_indexing", None
        ),
    )
    _validate_published_speculators_semantics(source_config)
    _validate_published_speculators_semantics(draft_config)
    _validate_warm_start_capture_provenance(source_config, draft_config)
    family = {
        "DFlashDraftModel",
        "DSparkDraftModel",
        "DFlash2DraftModel",
        "TaichuDFlash2DraftModel",
    }
    cross_family = (
        source_config.architectures != draft_config.architectures
        and source_config.architectures[0] in family
        and draft_config.architectures[0] in family
    )
    if source_config.architectures != draft_config.architectures and not cross_family:
        raise ValueError(
            f"warm-start checkpoint {source!r} has draft architecture "
            f"{source_config.architectures!r}, expected {draft_config.architectures!r}"
        )
    if getattr(source_config, "published_speculators_format", False) is True or getattr(
        source_config, "official_dspark_format", None
    ) in ("taichu_speculators", "lfm2"):
        # A shape-compatible scratch config can still change the anchor,
        # attention or RoPE semantics of published weights.
        for field in (
            "block_size",
            "vocab_size",
            "draft_vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "layer_types",
            "sliding_window",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "rope_is_neox_style",
            "rope_parameters",
            "speculators_attention_semantics",
        ):
            if getattr(source_config, field, None) != getattr(
                draft_config, field, None
            ):
                raise ValueError(f"published warm start has incompatible {field}")
        source_depth = getattr(
            source_config, "target_num_hidden_layers", None
        ) or getattr(source_config, "num_target_layers", None)
        destination_depth = getattr(
            draft_config, "target_num_hidden_layers", None
        ) or getattr(draft_config, "num_target_layers", None)
        if source_depth != destination_depth:
            raise ValueError(
                "published warm start has incompatible target depth metadata"
            )
        source_method = dict(getattr(source_config, "dflash_config", None) or {})
        target_method = dict(getattr(draft_config, "dflash_config", None) or {})
        for field in (
            "target_layer_ids",
            "mask_token_id",
            "sliding_window_non_causal",
            "pure_draft_prefix_len",
            "shift_label",
        ):
            if source_method.get(field) != target_method.get(field):
                raise ValueError(f"official DSpark warm start has incompatible {field}")
        if (
            not cross_family
            and getattr(source_config, "published_speculators_format", False) is True
        ):
            for field in ("projector_type", "domino_auf"):
                if source_method.get(field) != target_method.get(field):
                    raise ValueError(
                        f"published speculators warm start has incompatible {field}"
                    )
        if not cross_family and source_method.get(
            "sample_from_anchor"
        ) != target_method.get("sample_from_anchor"):
            raise ValueError(
                "official DSpark warm start has incompatible sample_from_anchor"
            )
    if cross_family:
        source_method = dict(getattr(source_config, "dflash_config", None) or {})
        target_method = getattr(draft_config, "dflash_config", None)
        if not isinstance(target_method, dict):
            raise ValueError("DFlash-family draft architecture requires dflash_config")
        if target_method.get("sample_from_anchor", False) is not False:
            raise ValueError("DFlash-family transfer requires sample_from_anchor=false")
        source_anchor = source_method.get(
            "sample_from_anchor", source_config.architectures == ["DSparkDraftModel"]
        )
        if source_anchor and not (
            retrain_anchor_alignment
            and source_config.architectures == ["DSparkDraftModel"]
        ):
            raise ValueError("DFlash-family source must use sample_from_anchor=false")
        for field in (
            "block_size",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "sliding_window",
        ):
            if getattr(source_config, field, None) != getattr(
                draft_config, field, None
            ):
                raise ValueError(f"DFlash-family warm start has incompatible {field}")
        if (
            strategy == "dflash2"
            and getattr(source_config, "draft_vocab_size", source_config.vocab_size)
            != source_config.vocab_size
        ):
            raise ValueError("DFlash2 transfer requires full draft vocabulary")
        from draftfit.modeling.draft.dflash import resolve_dflash_attention_layout

        if resolve_dflash_attention_layout(
            source_config
        ) != resolve_dflash_attention_layout(draft_config):
            raise ValueError(
                "DFlash-family warm start has incompatible attention layout"
            )
        for field in (
            "target_layer_ids",
            "mask_token_id",
            "shift_label",
            "pure_draft_prefix_len",
            "sliding_window_non_causal",
        ):
            if source_method.get(field) != target_method.get(field):
                raise ValueError(f"DFlash-family warm start has incompatible {field}")
    loaded, loading_info = AutoDraftModel.from_pretrained(
        source,
        config=source_config if cross_family else draft_config,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
    )
    # HF may discard extra source tensors when constructing the requested model.
    # Do not turn that lossy load into an apparently complete warm start.
    # Published DSpark exports sometimes carry copies of target embedding/head
    # tensors. The draft wrapper owns neither; all other extra keys still fail.
    extra_keys = set(loading_info.get("unexpected_keys") or [])
    allowed_extra = (
        {"embed_tokens.weight", "lm_head.weight"}
        if getattr(source_config, "published_speculators_format", False) is True
        or getattr(source_config, "official_dspark_format", None)
        == "taichu_speculators"
        else set()
    )
    if extra_keys - allowed_extra:
        raise ValueError(
            f"warm-start checkpoint {source!r} has incompatible unexpected_keys: "
            f"{sorted(extra_keys)}"
        )
    if (
        extra_keys
        and getattr(source_config, "published_speculators_format", False) is True
    ):
        import torch

        from draftfit.modeling.target.quantized_parts import load_target_parts
        from draftfit.training.official_dspark import verify_target_copies

        target_path = getattr(draft_config, "speculators_target_model_path", None)
        if not target_path:
            raise ValueError(
                "published target copies require the actual target model path"
            )
        target_parts = load_target_parts(
            target_path,
            cache_dir=cache_dir,
            device="cpu",
            dtype=torch.float32,
            trust_remote_code=False,
            revision=getattr(draft_config, "speculators_target_revision", None),
        )
        verify_target_copies(source, target_parts, cache_dir=cache_dir)
        del target_parts
    for field in ("mismatched_keys", "error_msgs"):
        if loading_info.get(field):
            raise ValueError(
                f"warm-start checkpoint {source!r} has incompatible {field}: "
                f"{loading_info[field]}"
            )
    missing_from_source = set(loading_info.get("missing_keys") or [])
    state = {
        key: value.detach().cpu()
        for key, value in loaded.state_dict().items()
        if key not in missing_from_source
        and not (cross_family and _source_only_family_key(key, strategy))
    }
    del loaded
    return state, cross_family


def warm_start_draft_model(
    model: Any,
    source: str,
    *,
    draft_config: "PretrainedConfig",
    strategy: str,
    allow_missing_embedding: bool = False,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    retrain_anchor_alignment: bool = False,
) -> WarmStartReport:
    """Load only draft weights, never optimizer/counters/RNG training state."""

    runtime_state = _runtime_state_file(source)
    if runtime_state is not None:
        _validate_published_speculators_semantics(draft_config)
        config_source = _checkpoint_config_source(source) or _checkpoint_config_source(
            runtime_state
        )
        source_payload = {}
        if config_source is not None:
            with open(config_source, encoding="utf-8") as stream:
                source_payload = json.load(stream)
        if (
            getattr(draft_config, "published_speculators_format", False) is True
            or source_payload.get("published_speculators_format") is True
        ):
            if not config_source:
                raise ValueError(
                    "published warm start requires a source config with capture provenance"
                )
            source_config = _draft_config_from_dict(
                source_payload,
                draft_capture_layer_indexing=getattr(
                    draft_config, "draft_capture_layer_indexing", None
                ),
            )
            _validate_warm_start_capture_provenance(source_config, draft_config)
        checkpoint_format: Literal["specforge", "pretrained"] = "specforge"
        state, cross_family = _load_specforge_draft_state(
            runtime_state,
            expected_strategy=strategy,
            draft_config=draft_config,
            retrain_anchor_alignment=retrain_anchor_alignment,
        )
    else:
        checkpoint_format = "pretrained"
        state, cross_family = _load_pretrained_draft_state(
            source,
            draft_config=draft_config,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            strategy=strategy,
            retrain_anchor_alignment=retrain_anchor_alignment,
        )

    if not state:
        raise ValueError(f"warm-start checkpoint {source!r} contains no draft weights")
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"warm-start checkpoint {source!r} has incompatible draft tensor "
            f"shapes: {exc}"
        ) from exc
    loaded_keys = len(state) - len(result.unexpected_keys)
    if result.unexpected_keys or loaded_keys == 0:
        raise ValueError(
            f"warm-start checkpoint {source!r} does not match this draft model: "
            f"loaded={loaded_keys}/{len(state)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    allowed_missing = set()
    if allow_missing_embedding:
        allowed_missing = {
            key for key in result.missing_keys if key == "embed_tokens.weight"
        }
    if cross_family:
        allowed_missing.update(
            key
            for key in result.missing_keys
            if (
                strategy == "dspark"
                and key.startswith(("markov_head.", "confidence_head."))
            )
            or (
                strategy == "dflash2"
                and (
                    key.startswith("candidate_selector.")
                    or ".attention_conv." in key
                    or ".mlp_conv." in key
                )
            )
        )
    required_missing = sorted(set(result.missing_keys) - allowed_missing)
    if required_missing:
        raise ValueError(
            f"warm-start checkpoint {source!r} is missing draft weights required "
            f"by this architecture: {required_missing}"
        )

    report = WarmStartReport(
        source=str(source),
        checkpoint_format=checkpoint_format,
        loaded_keys=loaded_keys,
        missing_keys=tuple(sorted(result.missing_keys)),
        loaded_embedding=any("embed" in key.lower() for key in state),
    )
    logger.info(
        "Warm-started %d draft tensors from %s (%s); missing=%s",
        report.loaded_keys,
        report.source,
        report.checkpoint_format,
        list(report.missing_keys),
    )
    return report


__all__ = [
    "WarmStartReport",
    "draft_config_dict",
    "load_draft_config_source",
    "resolve_draft_config",
    "warm_start_draft_model",
]
