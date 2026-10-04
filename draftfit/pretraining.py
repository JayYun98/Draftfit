"""Single-device raw-text warm start for Qwen3-shaped DFlash-family drafts."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import torch

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDFlashModel,
    OnlineDSparkModel,
)
from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    target_text_config,
)
from draftfit.training.model_loading import load_draft_config_source
from draftfit.utils import load_tokenizer

_FAMILY = {
    "dflash": ("DFlashDraftModel", OnlineDFlashModel),
    "dspark": ("DSparkDraftModel", OnlineDSparkModel),
    "dflash2": ("DFlash2DraftModel", OnlineDFlash2Model),
}


def _set_taps(config, taps: list[int]) -> None:
    config.target_layer_ids = taps
    config.dflash_config = {**(getattr(config, "dflash_config", None) or {}), "target_layer_ids": taps}
    if config.architectures == ["DFlash2DraftModel"]:
        config.num_target_layers = len(taps)


def _packed_text(path: str, tokenizer, sequence_length: int) -> Iterator[torch.Tensor]:
    eos = tokenizer.eos_token_id
    if isinstance(eos, (tuple, list)):
        eos = eos[0] if eos else None
    if eos is None:
        raise ValueError("raw-text pretraining requires a tokenizer EOS token")
    buffer: list[int] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number} requires a string 'text' field")
            if not row["text"].strip():
                continue
            buffer.extend(tokenizer.encode(row["text"], add_special_tokens=False))
            buffer.append(eos)
            while len(buffer) >= sequence_length:
                yield torch.tensor(buffer[:sequence_length], dtype=torch.long)
                del buffer[:sequence_length]


def _make_wrapper(strategy: str, draft, parts, mask_token_id: int, anchors: int):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=parts.lm_head,
        target_embed_tokens=parts.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=draft.block_size,
        attention_backend="sdpa",
        num_anchors=anchors,
    )
    if strategy == "dspark":
        kwargs.update(hard_targets=True,
                      sample_from_anchor=bool(draft.config.dflash_config.get("sample_from_anchor", True)))
    return _FAMILY[strategy][1](**kwargs)


def _position_metrics(model_metrics: dict) -> dict:
    ratios = model_metrics.get("ratio_metrics", {})
    result = {}
    for source, name in (("accuracy_position", "position_accuracy"),
                         ("ce_position", "position_ce")):
        if source in ratios:
            numerator, denominator = ratios[source]
            result[name] = (numerator / denominator.clamp_min(1)).detach().cpu().tolist()
    return result


def pretrain(
    *, target: str, draft_config: str, strategy: str, corpus: str,
    output: str, steps: int, sequence_length: int, batch_size: int = 1,
    learning_rate: float = 1e-4, validation: str | None = None,
    validation_sequences: int = 32,
    target_revision: str | None = None, seed: int = 42,
) -> dict:
    """Train a draft using only target embedding/head weights and hard text tokens."""
    if strategy not in _FAMILY:
        raise ValueError(f"strategy must be one of {sorted(_FAMILY)}")
    if min(steps, sequence_length, batch_size, validation_sequences) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps, sequence_length, batch_size, validation_sequences and learning_rate must be positive")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists: {output}")
    if validation is not None and Path(validation).resolve() == Path(corpus).resolve():
        raise ValueError("validation must be a separate held-out corpus file")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = load_draft_config_source(draft_config)
    if config.architectures != [_FAMILY[strategy][0]]:
        raise ValueError(f"{strategy} requires {_FAMILY[strategy][0]} draft config")
    _set_taps(config, [0])
    if sequence_length <= int(config.block_size):
        raise ValueError("sequence_length must exceed draft block_size")
    tokenizer = load_tokenizer(target, revision=target_revision)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        target, revision=target_revision, device=device.type, dtype=dtype,
    )
    target_config = target_text_config(parts.config)
    if int(config.hidden_size) != int(target_config.hidden_size) or int(config.vocab_size) != parts.embed_tokens.num_embeddings:
        raise ValueError("draft hidden size and vocabulary must match the frozen target")
    if float(getattr(config, "input_embedding_scale", 1.0)) != 1.0:
        raise ValueError("draft input_embedding_scale requires explicit scaling support")
    if float(getattr(target_config, "embedding_multiplier", 1.0)) != 1.0:
        raise ValueError("target embedding_multiplier requires explicit scaling support")
    if "gemma" in str(getattr(target_config, "model_type", "")):
        raise ValueError("Gemma embedding scaling is not supported by this pretrainer")
    mask_token_id = (getattr(config, "dflash_config", None) or {}).get(
        "mask_token_id", getattr(config, "mask_token_id", None)
    )
    if not isinstance(mask_token_id, int) or not 0 <= mask_token_id < int(config.vocab_size):
        raise ValueError("draft config requires a valid mask_token_id")
    draft = AutoDraftModel.from_config(config, torch_dtype=dtype).to(device)
    initial_projection = draft.fc.weight.detach().clone()
    wrapper = _make_wrapper(strategy, draft, parts, mask_token_id, anchors=min(16, sequence_length - 1)).to(device)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=learning_rate)
    source = _packed_text(corpus, tokenizer, sequence_length)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".draftfit-pretrain-", dir=destination.parent))
    try:
        with (scratch / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(1, steps + 1):
                samples = [next(source, None) for _ in range(batch_size)]
                if any(sample is None for sample in samples):
                    raise ValueError(f"corpus exhausted before step {step}; add text or reduce steps")
                ids = torch.stack(samples).to(device)
                with torch.no_grad():
                    hidden = parts.embed_tokens(ids)
                wrapper.train()
                optimizer.zero_grad(set_to_none=True)
                loss, accuracy, model_metrics = wrapper(
                    ids, hidden, torch.ones_like(ids, dtype=torch.float32)
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite pretraining loss at step {step}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise ValueError(f"non-finite pretraining gradient at step {step}")
                optimizer.step()
                last_row = {
                    "step": step, "loss": float(loss.detach()),
                    "accuracy": float(accuracy.detach()),
                    "grad_norm": float(gradient), **_position_metrics(model_metrics),
                }
                stream.write(json.dumps(last_row) + "\n")
        result = {
            "strategy": strategy, "target": target,
            "target_revision": target_revision, "draft_config": draft_config,
            "corpus": corpus, "validation_corpus": validation,
            "seed": seed, "sequence_length": sequence_length,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "steps": steps, "tokens": steps * batch_size * sequence_length,
            "train": last_row,
            "projection_update_norm": float(torch.linalg.vector_norm(
                draft.fc.weight.detach().float() - initial_projection.float()
            )),
        }
        if validation is not None:
            result["validation"] = _validate_pretrain(
                wrapper, parts, tokenizer, validation, sequence_length,
                validation_sequences, device,
            )
        draft.save_pretrained(scratch, safe_serialization=True)
        (scratch / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"output already exists: {output}")
        scratch.rename(destination)
        return result
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _validate_pretrain(wrapper, parts, tokenizer, validation, sequence_length,
                       validation_sequences, device):
    totals = {"loss": 0.0, "accuracy": 0.0}
    token_count = 0.0
    count = 0
    position_num = None
    position_den = None
    for sample in _packed_text(validation, tokenizer, sequence_length):
        if count >= validation_sequences:
            break
        ids = sample.unsqueeze(0).to(device)
        wrapper.eval()
        with torch.no_grad():
            loss, accuracy, model_metrics = wrapper(
                ids, parts.embed_tokens(ids),
                torch.ones_like(ids, dtype=torch.float32),
            )
        if not torch.isfinite(loss):
            raise ValueError("non-finite validation loss")
        weight = float(model_metrics["accuracy_denom"])
        totals["loss"] += float(loss) * weight
        totals["accuracy"] += float(accuracy) * weight
        token_count += weight
        if "accuracy_position" in model_metrics["ratio_metrics"]:
            numerator, denominator = model_metrics["ratio_metrics"]["accuracy_position"]
            position_num = numerator.detach().clone() if position_num is None else position_num + numerator
            position_den = denominator.detach().clone() if position_den is None else position_den + denominator
        count += 1
    if not count:
        raise ValueError("validation corpus has no full sequence")
    result = {name: value / token_count for name, value in totals.items()}
    result["sequences"] = count
    result["supervised_tokens"] = int(token_count)
    if position_num is not None:
        result["position_accuracy"] = (
            position_num / position_den.clamp_min(1)
        ).cpu().tolist()
    return result


def expand_aux_layers(checkpoint: str, output: str, taps: list[int]) -> dict:
    """Widen a one-tap projection without changing its embedding-only output."""
    if not taps or len(set(taps)) != len(taps) or 0 not in taps or any(t < 0 for t in taps):
        raise ValueError("destination taps must be unique nonnegative IDs including 0")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = load_draft_config_source(checkpoint)
    if config.architectures[0] not in {entry[0] for entry in _FAMILY.values()}:
        raise ValueError("auxiliary expansion supports only DFlash-family drafts")
    if list(config.dflash_config.get("target_layer_ids", [])) != [0]:
        raise ValueError("source checkpoint must have exactly one auxiliary tap: 0")
    if max(taps) >= int(getattr(config, "target_num_hidden_layers", config.num_target_layers)):
        raise ValueError("destination tap exceeds target depth")
    source = AutoDraftModel.from_pretrained(checkpoint, config=config)
    old = source.fc.weight.detach()
    width = old.shape[1]
    _set_taps(config, taps)
    expanded = AutoDraftModel.from_config(config, torch_dtype=old.dtype)
    state = source.state_dict()
    projection = old.new_zeros((old.shape[0], width * len(taps)))
    slot = taps.index(0)
    projection[:, slot * width:(slot + 1) * width] = old
    state["fc.weight"] = projection
    expanded.load_state_dict(state, strict=True)
    destination.mkdir(parents=True)
    expanded.save_pretrained(destination, safe_serialization=True)
    return {"source_taps": [0], "target_taps": taps, "embedding_slot": slot,
            "output": str(destination)}
