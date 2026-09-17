"""Preflight checks for editable projects; never construct models or write output."""

import json
from pathlib import Path


def validate_conversation_file(path):
    """Stream public JSONL input; report locations, never private row contents."""
    rows = 0
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        while line := handle.readline(2 * 1024 * 1024 + 1):
            rows += 1
            if len(line) > 2 * 1024 * 1024:
                raise ValueError(f"data row {rows}: exceeds 2 MiB character limit")
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(
                    f"data row {rows}: expected a JSON object on each line"
                ) from None
            messages = row.get("conversations") if isinstance(row, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"data row {rows}: requires a nonempty conversations array"
                )
            assistant = False
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"data row {rows}: conversation messages must be objects"
                    )
                role = message.get("role", message.get("from"))
                if role not in (
                    "system",
                    "developer",
                    "user",
                    "human",
                    "assistant",
                    "gpt",
                    "tool",
                    "function",
                ):
                    raise ValueError(
                        f"data row {rows}: unsupported or missing message role"
                    )
                content = message.get("content", message.get("value"))
                if content is not None and not isinstance(content, str):
                    raise ValueError(
                        f"data row {rows}: this text recipe requires string content"
                    )
                assistant |= role in ("assistant", "gpt") and bool(
                    content or message.get("tool_calls")
                )
            if not assistant:
                raise ValueError(
                    f"data row {rows}: requires a nonempty assistant training response"
                )
    if not rows:
        raise ValueError("training data is empty")
    return rows


def validate_project_data(data, inspection):
    """Validate the conversation recipe without executing the tokenizer."""
    if data.chat_template == "explicit-required":
        raise ValueError(
            "target has no chat template; select an explicit renderer with --set data.chat_template=NAME"
        )
    from speculative_train_platform.data.template import TEMPLATE_REGISTRY

    TEMPLATE_REGISTRY.get(data.chat_template)
    if data.is_preformatted:
        raise ValueError(
            "target prepare requires conversation JSONL; use an explicit run config for preformatted data"
        )
    if data.chat_template == "hf":
        if data.train_only_last_turn:
            raise ValueError("hf rendering requires all-turn training")
        if inspection["facts"].get("chat_template_has_generation") is False:
            raise ValueError(
                "hf rendering requires native {% generation %} spans; select a registered renderer"
            )
    return validate_conversation_file(data.train_data_path)


def validate_draft_metadata(
    *, strategy, target_depth, target_metadata, draft, layers, mask_token_id
):
    """Check provider-resolved draft dimensions and taps using metadata alone."""
    if (
        not target_depth
        or not layers
        or len(set(layers)) != len(layers)
        or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= target_depth
            for i in layers
        )
    ):
        raise ValueError(
            f"capture layers must be unique indices within target depth {target_depth}: {layers}"
        )
    for name in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
    ):
        value = getattr(draft, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"draft requires positive {name}; supply an explicit compatible --draft-config"
            )
    if strategy in ("dspark", "dflash", "dflash2", "domino"):
        for name, expected in (
            ("hidden_size", target_metadata.get("hidden_size")),
            (
                "vocab_size",
                target_metadata.get("padded_vocab_size")
                or target_metadata.get("vocab_size"),
            ),
        ):
            if getattr(draft, name) != expected:
                raise ValueError(
                    f"draft {name} must match target {expected}, got {getattr(draft, name)}"
                )
    if mask_token_id is not None and not 0 <= mask_token_id < draft.vocab_size:
        raise ValueError("model.mask_token_id must be inside the target vocabulary")
