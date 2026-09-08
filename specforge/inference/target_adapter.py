"""Small target-adapter boundary shared by rendering and validation.

The adapter deliberately owns only metadata and token rendering.  Heavy model
loading remains in the existing SGLang/transformers providers; this keeps the
same contract usable for dense, MoE and stateful targets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

from .state import TargetStateAdapter, TargetStateSnapshot


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    renderer: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("RenderedPrompt.text must be a string")
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError("input_ids and loss_mask must have equal lengths")
        if any(int(value) not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask must contain only 0/1 values")


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError(f"tokenizer input must be text, got {type(text).__name__}")
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise
        encoded = encode(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output mapping has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and encoded and isinstance(
        encoded[0], (list, tuple)
    ):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned multiple rows for one text")
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)):
        raise TypeError(
            "tokenizer input_ids must be a sequence, "
            f"got {type(encoded).__name__}"
        )
    return [int(token) for token in encoded]


def _normal_role(message: Mapping[str, Any], index: int) -> str:
    if not isinstance(message, Mapping):
        raise TypeError(f"message {index} must be a mapping")
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"message {index} requires a non-empty role")
    return {"human": "user", "gpt": "assistant"}.get(role.strip().lower(), role.strip().lower())


def _message_content(message: Mapping[str, Any], index: int) -> str:
    content = message.get("content", "")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError(
            f"message {index} content must be a string for Ling text rendering, "
            f"got {type(content).__name__}"
        )
    return content


class LingRenderer:
    """Renderer for Ling checkpoints with a deterministic fallback.

    When the checkpoint exposes a tokenizer chat template it is authoritative.
    The fallback mirrors the registered Ling text protocol and is used only
    when a tokenizer omits the template (for example in a local fixture).
    """

    name = "ling-3.0"

    def render(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> RenderedPrompt:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of role/content mappings")
        messages = list(messages)
        if not messages:
            raise ValueError("messages must not be empty")
        roles = [_normal_role(message, i) for i, message in enumerate(messages)]
        contents = [_message_content(message, i) for i, message in enumerate(messages)]
        normalized_messages = [
            {**dict(message), "role": role, "content": content}
            for message, role, content in zip(messages, roles, contents)
        ]

        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            try:
                text = template(normalized_messages, **kwargs)
            except TypeError:
                kwargs.pop("add_generation_prompt")
                text = template(normalized_messages, **kwargs)
            if not isinstance(text, str):
                raise TypeError(
                    "tokenizer.apply_chat_template(tokenize=False) must return "
                    f"str, got {type(text).__name__}"
                )
        else:
            parts: list[str] = []
            # Match the Ling template registry's fallback path (system prompt
            # is plain text, user is HUMAN, and turns terminate explicitly).
            if roles[0] != "system":
                parts.append("You are a helpful assistant.")
            for role, content in zip(roles, contents):
                if role == "system":
                    parts.append(content)
                    continue
                parts.append(f"<role>{'HUMAN' if role == 'user' else role.upper()}</role>{content}")
                if role in {"assistant", "tool"}:
                    parts.append("<|role_end|>")
            if add_generation_prompt:
                parts.append("<role>ASSISTANT</role>")
            text = "".join(parts)
        ids = _tokenize(tokenizer, text)
        # Mark assistant content using a rendered prefix first.  Searching the
        # whole sequence is only the fallback because identical user/assistant
        # text would otherwise mark the wrong occurrence.
        mask = [0 for _ in ids]
        cursor = 0
        for index, (role, content) in enumerate(zip(roles, contents)):
            if role != "assistant":
                continue
            content_ids = _tokenize(tokenizer, content)
            if not content_ids:
                continue
            found = -1
            prefix_ids: list[int] = []
            if callable(template):
                prefix_messages = normalized_messages[:index] + [{"role": "assistant", "content": ""}]
                try:
                    prefix_text = template(prefix_messages, tokenize=False, add_generation_prompt=False)
                    prefix_ids = _tokenize(tokenizer, prefix_text)
                except (TypeError, KeyError, ValueError):
                    prefix_ids = []
            if prefix_ids and ids[: len(prefix_ids)] == prefix_ids:
                candidate = len(prefix_ids)
                if ids[candidate : candidate + len(content_ids)] == content_ids:
                    found = candidate
            for start in range(cursor, len(ids) - len(content_ids) + 1):
                if found >= 0:
                    break
                if ids[start : start + len(content_ids)] == content_ids:
                    found = start
                    break
            if found < 0:
                continue
            mask[found : found + len(content_ids)] = [1] * len(content_ids)
            cursor = found + len(content_ids)
        return RenderedPrompt(str(text), tuple(ids), tuple(mask), self.name)


class ChatTemplateRenderer(LingRenderer):
    """Strict renderer for targets whose tokenizer owns the chat protocol.

    The inherited implementation contains the assistant-span matching logic;
    only the Ling fallback is disabled here.  A generic target must never
    silently receive another model family's delimiters.
    """

    name = "hf-chat-template"

    def render(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> RenderedPrompt:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ValueError(
                "generic target requires tokenizer.apply_chat_template; "
                "register an explicit renderer for targets without one"
            )
        rendered = super().render(
            tokenizer, messages, add_generation_prompt=add_generation_prompt
        )
        return RenderedPrompt(
            rendered.text,
            rendered.input_ids,
            rendered.loss_mask,
            self.name,
        )


@dataclass(frozen=True)
class TargetAdapter:
    """Model-specific metadata used by capture and validation paths."""

    model_id: str
    architecture_lane: str
    state_kind: str
    capture_layers: tuple[int, ...]
    renderer: LingRenderer
    capabilities: frozenset[str]
    revision: str = "unknown"

    _SUPPORTED_LANES: ClassVar[frozenset[str]] = frozenset(
        {
            "dense",
            "moe",
            "moe_hybrid",
            "hybrid_stateful",
            "attention_hybrid",
            "linear_recurrent",
        }
    )
    _SUPPORTED_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "kv",
            "recurrent",
            "hybrid_kv",
            "linear_plus_kv",
            "conv_plus_kv",
            "moe_plus_kv",
            "moe_plus_linear",
            "moe_plus_conv",
        }
    )
    _LANE_STATES: ClassVar[dict[str, frozenset[str]]] = {
        "dense": frozenset({"kv"}),
        "moe": frozenset({"kv"}),
        "moe_hybrid": frozenset(
            {"moe_plus_kv", "moe_plus_linear", "moe_plus_conv"}
        ),
        "hybrid_stateful": frozenset(
            {"hybrid_kv", "linear_plus_kv", "conv_plus_kv"}
        ),
        "attention_hybrid": frozenset({"kv", "hybrid_kv"}),
        "linear_recurrent": frozenset({"recurrent"}),
    }

    @classmethod
    def ling(cls, *, capture_layers: Sequence[int] = ()) -> "TargetAdapter":
        layers = tuple(int(layer) for layer in capture_layers)
        if any(layer < 0 for layer in layers):
            raise ValueError("capture layer IDs must be non-negative")
        if len(set(layers)) != len(layers):
            raise ValueError("capture layer IDs must be unique")
        return cls(
            model_id="inclusionAI/Ling-3.0-tiny",
            architecture_lane="moe_hybrid",
            state_kind="moe_plus_conv",
            capture_layers=layers,
            renderer=LingRenderer(),
            capabilities=frozenset(
                {"capture", "train", "serve", "rollback_replay", "exact_token_parity"}
            ),
        )

    @classmethod
    def from_inspection(
        cls,
        inspection: Mapping[str, Any],
        *,
        capture_layers: Sequence[int] | None = None,
        renderer: Any | None = None,
    ) -> "TargetAdapter":
        """Build an adapter from ``target_inspect_v1`` metadata only.

        This is intentionally weight-free: model execution remains owned by
        the configured backend.  Unknown architecture/state lanes are refused
        instead of being treated as a dense model by accident.
        """
        if not isinstance(inspection, Mapping):
            raise TypeError("target inspection must be a mapping")
        if inspection.get("schema_version") != "target_inspect_v1":
            raise ValueError(
                "unsupported target inspection schema; expected target_inspect_v1"
            )
        source = inspection.get("source")
        revision = inspection.get("revision")
        facts = inspection.get("facts")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("target inspection source must be non-empty")
        if (
            not isinstance(revision, str)
            or not revision.strip()
            or revision.strip().lower() in {"main", "master", "unknown"}
        ):
            raise ValueError(
                "generic target adapter requires a pinned model revision, not a branch"
            )
        if not isinstance(facts, Mapping):
            raise ValueError("target inspection facts are required")
        lane = facts.get("architecture_lane")
        state_kind = facts.get("state_kind")
        if lane not in cls._SUPPORTED_LANES:
            raise ValueError(f"unsupported target architecture lane: {lane!r}")
        if state_kind not in cls._SUPPORTED_STATES:
            raise ValueError(f"unsupported target state kind: {state_kind!r}")
        if state_kind not in cls._LANE_STATES[lane]:
            raise ValueError(
                f"state kind {state_kind!r} is incompatible with architecture lane {lane!r}"
            )
        n_layers = facts.get("num_hidden_layers")
        if n_layers is not None:
            if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers <= 0:
                raise ValueError("target inspection num_hidden_layers must be positive")
        if capture_layers is None:
            recommendations = inspection.get("recommendations")
            candidates = recommendations.get("target_tap_candidates", []) if isinstance(recommendations, Mapping) else []
            capture_layers = tuple(candidates)
        adapter = cls(
            model_id=source,
            architecture_lane=lane,
            state_kind=state_kind,
            capture_layers=tuple(capture_layers),
            renderer=renderer or ChatTemplateRenderer(),
            capabilities=frozenset({"capture", "train", "serve", "exact_token_parity"}),
            revision=revision.strip(),
        )
        if n_layers is not None and adapter.capture_layers:
            adapter.validate_capture_layers(adapter.capture_layers, num_hidden_layers=n_layers)
        return adapter

    def render(self, tokenizer: Any, messages: Sequence[Mapping[str, Any]], **kwargs) -> RenderedPrompt:
        return self.renderer.render(tokenizer, messages, **kwargs)

    def validate_capture_layers(self, layers: Sequence[int], *, num_hidden_layers: int | None = None) -> tuple[int, ...]:
        raw = tuple(layers)
        if not raw or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw):
            raise ValueError(f"capture layers must be distinct non-negative integers, got {list(layers)!r}")
        resolved = tuple(raw)
        if any(layer < 0 for layer in resolved) or len(set(resolved)) != len(resolved):
            raise ValueError(f"capture layers must be distinct non-negative integers, got {list(layers)!r}")
        if num_hidden_layers is not None and any(layer >= int(num_hidden_layers) for layer in resolved):
            raise ValueError(f"capture layer exceeds target depth {num_hidden_layers}: {list(resolved)!r}")
        return resolved

    def contract_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture_lane": self.architecture_lane,
            "state_kind": self.state_kind,
            "capture_layers": self.capture_layers,
            "renderer": self.renderer.name,
            "capabilities": sorted(self.capabilities),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]


@dataclass(frozen=True, init=False)
class LingTargetAdapter(TargetAdapter):
    """Named Ling adapter for callers that prefer a concrete type."""

    def __init__(self, *, capture_layers: Sequence[int] = ()) -> None:
        base = TargetAdapter.ling(capture_layers=capture_layers)
        for name in ("model_id", "architecture_lane", "state_kind", "capture_layers", "renderer", "capabilities", "revision"):
            object.__setattr__(self, name, getattr(base, name))


__all__ = [
    "ChatTemplateRenderer",
    "LingRenderer",
    "LingTargetAdapter",
    "RenderedPrompt",
    "TargetAdapter",
    "TargetStateAdapter",
    "TargetStateSnapshot",
]
