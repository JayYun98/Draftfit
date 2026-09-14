"""Weight-free target spec normalization and run-config scaffolding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_LANES = {
    "dense",
    "moe",
    "moe_hybrid",
    "hybrid_stateful",
    "attention_hybrid",
    "linear_recurrent",
}
_STATES = {
    "kv",
    "recurrent",
    "hybrid_kv",
    "linear_plus_kv",
    "conv_plus_kv",
    "moe_plus_kv",
    "moe_plus_linear",
    "moe_plus_conv",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "target"


def _as_ints(values: Any) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class TargetSpec:
    """Normalized architecture facts plus bounded, testable recommendations."""

    model_id: str
    revision: str
    architecture_lane: str
    state_kind: str
    num_hidden_layers: int | None
    layer_types: tuple[str, ...]
    chat_template_present: bool
    recommendations: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    unknown: tuple[str, ...]

    @classmethod
    def from_inspection(cls, inspection: Mapping[str, Any]) -> "TargetSpec":
        if inspection.get("schema_version") != "target_inspect_v1":
            raise ValueError("expected target_inspect_v1 inspection")
        source = inspection.get("source")
        revision = inspection.get("revision")
        facts = inspection.get("facts")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("inspection source must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("inspection revision must be a non-empty string")
        if not isinstance(facts, Mapping):
            raise ValueError("inspection facts are required")
        lane = facts.get("architecture_lane")
        state = facts.get("state_kind")
        if lane not in _LANES:
            raise ValueError(f"unsupported architecture lane: {lane!r}")
        if state not in _STATES:
            raise ValueError(f"unsupported state kind: {state!r}")
        depth = facts.get("num_hidden_layers")
        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0
        ):
            raise ValueError("facts.num_hidden_layers must be positive or null")
        recommendations = inspection.get("recommendations") or {}
        if not isinstance(recommendations, Mapping):
            raise ValueError("inspection recommendations must be a mapping")
        return cls(
            model_id=source.strip(),
            revision=revision.strip(),
            architecture_lane=lane,
            state_kind=state,
            num_hidden_layers=depth,
            layer_types=tuple(str(value) for value in facts.get("layer_types", [])),
            chat_template_present=bool(facts.get("chat_template_present")),
            recommendations=dict(recommendations),
            evidence=tuple(inspection.get("evidence") or ()),
            unknown=tuple(str(value) for value in inspection.get("unknown") or ()),
        )

    @property
    def tap_candidates(self) -> tuple[int, ...]:
        return _as_ints(self.recommendations.get("target_tap_candidates", ()))

    @property
    def default_taps(self) -> tuple[int, ...]:
        candidates = self.tap_candidates
        if candidates:
            return candidates
        if self.num_hidden_layers:
            return (self.num_hidden_layers // 2,)
        return ()

    @property
    def support(self) -> dict[str, str]:
        if self.state_kind == "kv":
            return {
                "capture": "candidate",
                "train": "candidate",
                "serve": "backend_probe_required",
            }
        return {
            "capture": "state_probe_required",
            "train": "offline_candidate",
            "serve": "state_replay_required",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "target_spec_v1",
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture_lane": self.architecture_lane,
            "state_kind": self.state_kind,
            "num_hidden_layers": self.num_hidden_layers,
            "layer_types": list(self.layer_types),
            "chat_template_present": self.chat_template_present,
            "recommendations": dict(self.recommendations),
            "support": self.support,
            "evidence": list(self.evidence),
            "unknown": list(self.unknown),
        }

    def scaffold_config(
        self,
        *,
        strategy: str = "dspark",
        data_root: str = "./cache/hidden_states",
        output_root: str = "./outputs",
    ) -> dict[str, Any]:
        """Return a structurally valid offline config, without claiming data exists."""
        from specforge.algorithms.builtin import builtin_algorithm_registry

        try:
            algorithm = builtin_algorithm_registry().resolve(strategy)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        slug = _slug(self.model_id)
        depths = [
            int(value)
            for value in self.recommendations.get("draft_depth_candidates", (3,))
        ]
        depth = next((value for value in depths if value > 0), 3)
        blocks = [
            int(value)
            for value in self.recommendations.get("train_block_candidates", (7,))
        ]
        block = next((value for value in blocks if value > 1), 7)
        taps = list(self.default_taps)
        config: dict[str, Any] = {
            "model": {
                "target_model_path": self.model_id,
                "target_backend": "sglang",
                "trust_remote_code": False,
                "sglang_disable_radix_cache": self.state_kind != "kv",
            },
            "data": {
                "hidden_states_path": f"{data_root.rstrip('/')}/{slug}",
                "chat_template": (
                    "hf" if self.chat_template_present else "explicit-required"
                ),
                "cache_dir": "./cache",
            },
            "training": {
                "strategy": strategy,
                "anchor_sampling": "random",
                "fsdp_version": "v2",
                "fsdp_sharding": "FULL_SHARD",
            },
            "run_id": f"{slug}-{strategy}",
            "output_dir": f"{output_root.rstrip('/')}/{slug}-{strategy}",
        }
        requirement = algorithm.spec.draft
        fixed = dict(requirement.fixed_override_values)
        if "num_hidden_layers" in requirement.supported_overrides:
            config["model"]["draft_num_hidden_layers"] = fixed.get(
                "num_hidden_layers", depth
            )
        if "block_size" in requirement.supported_overrides:
            config["model"]["draft_block_size"] = block
            config["model"]["target_layer_ids"] = taps or None
        if algorithm.spec.capabilities.allows_aux_layer_override and len(taps) >= 3:
            config["model"]["aux_hidden_state_layer_ids"] = [
                taps[0],
                taps[len(taps) // 2],
                taps[-1],
            ]
        return config

    def write_scaffold(self, output: str, **kwargs: Any) -> tuple[str, str]:
        output_path = Path(output)
        config = self.scaffold_config(**kwargs)
        spec_path = output_path.with_suffix(output_path.suffix + ".target-spec.json")
        for path in (output_path, spec_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing to overwrite {path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for path, payload in ((output_path, config), (spec_path, self.to_dict())):
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return str(output_path), str(spec_path)


__all__ = ["TargetSpec"]
