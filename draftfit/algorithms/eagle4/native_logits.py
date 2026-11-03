"""Ingest externally captured, unshifted full-vocabulary EAGLE4 logits.

Each saved row contains input_ids[N], loss_mask[N], aux_hidden_state[N,3H],
native_logits[N-start,V], and scalar int64 native_logit_start, with 1 <= start < N.
Logits at original position j supervise output j-1. The loss mask already names
output positions. This module persists producer artifacts; it does not capture
or certify a native inference backend.
"""

from dataclasses import dataclass
from pathlib import Path

from draftfit.algorithms.contracts import OfflineStorageContract

NORMALIZER_ID = "eagle4_offline_v1"
NATIVE_KEYS = frozenset(
    {
        "input_ids",
        "loss_mask",
        "aux_hidden_state",
        "native_logits",
        "native_logit_start",
    }
)
DENSE_KEYS = frozenset({"input_ids", "loss_mask", "aux_hidden_state", "hidden_state"})


@dataclass(frozen=True)
class Eagle4OfflineStorageContract(OfflineStorageContract):
    """Keep the default capture layout and describe the external logits layout."""

    def required_tensors_for(self, target_repr):
        if target_repr == "native_logits":
            return NATIVE_KEYS
        if target_repr == "hidden_state":
            return DENSE_KEYS
        raise ValueError(
            f"unsupported EAGLE4 offline target representation: {target_repr!r}"
        )


STORAGE_CONTRACT = Eagle4OfflineStorageContract(
    format="specforge_hidden_states_v1",
    required_tensors=DENSE_KEYS,
    normalizer=NORMALIZER_ID,
)


def load_offline_manifest(path):
    from draftfit.inference.capture_manifest import load_capture_manifest

    root = Path(path).expanduser()
    manifest = (root.parent if root.is_file() else root) / "capture_manifest.json"
    return load_capture_manifest(str(manifest)) if manifest.is_file() else None


def offline_representation(path):
    manifest = load_offline_manifest(path)
    return manifest["contract"].get("target_repr") if manifest else "hidden_state"


def _validate_native_logits(
    input_ids, target, loss_mask, start, *, vocab_size, attention_mask=None
):
    import torch

    if (
        input_ids.ndim != 2
        or input_ids.dtype not in (torch.int32, torch.int64)
        or not all(input_ids.shape)
    ):
        raise ValueError("native input_ids must be nonempty integer [batch, sequence]")
    batch, sequence = input_ids.shape
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("native logits require a positive target vocabulary size")
    if (
        target.ndim != 3
        or target.shape[0] != batch
        or target.shape[2] != vocab_size
        or not target.is_floating_point()
        or target.requires_grad
        or not torch.isfinite(target).all()
    ):
        raise ValueError(
            "native logits must be finite detached floating [batch, suffix, vocab]"
        )
    if (
        not isinstance(start, torch.Tensor)
        or start.dtype != torch.int64
        or start.shape != (batch,)
    ):
        raise ValueError("native_logit_start must be an int64 [batch] tensor")
    if loss_mask.shape != input_ids.shape or not torch.all(
        (loss_mask == 0) | (loss_mask == 1)
    ):
        raise ValueError("native loss_mask must be binary [batch, sequence]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape or not torch.all(
        (attention_mask == 0) | (attention_mask == 1)
    ):
        raise ValueError("native attention_mask must be binary [batch, sequence]")
    lengths = attention_mask.long().sum(-1).tolist()
    starts = start.tolist()
    if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
        raise ValueError("native input_ids are outside the target vocabulary")
    suffix_lengths = []
    for row, (length, first) in enumerate(zip(lengths, starts)):
        expected = torch.arange(sequence, device=attention_mask.device) < length
        if not torch.equal(attention_mask[row].bool(), expected):
            raise ValueError(
                "native attention must describe right-padded contiguous tokens"
            )
        if not 1 <= first < length:
            raise ValueError("native_logit_start must be in [1, sequence length)")
        count = length - first
        suffix_lengths.append(count)
        if (
            loss_mask[row, : first - 1].any()
            or loss_mask[row, length - 1 :].any()
            or not loss_mask[row].any()
        ):
            raise ValueError(
                "native logits do not cover every supervised output position"
            )
        if target[row, count:].count_nonzero():
            raise ValueError("native logits suffix padding must be zero")
    if target.shape[1] != max(suffix_lengths):
        raise ValueError(
            "native logits must contain the complete original-position suffix"
        )
    return starts, suffix_lengths


def prepare_native_logits_target(
    input_ids, target, loss_mask, start, device, *, vocab_size, attention_mask=None
):
    """Validate padded suffixes, shift IDs once, and scatter j to j-1."""
    import torch

    starts, suffix_lengths = _validate_native_logits(
        input_ids,
        target,
        loss_mask,
        start,
        vocab_size=vocab_size,
        attention_mask=attention_mask,
    )
    batch, sequence = input_ids.shape
    ids = input_ids.to(device, non_blocking=True)
    shifted_ids = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, -1:])), dim=1)
    dense = target.new_zeros((batch, sequence, vocab_size), device=device)
    for row, (first, count) in enumerate(zip(starts, suffix_lengths)):
        dense[row, first - 1 : first - 1 + count] = target[row, :count].to(device)
    return shifted_ids, dense, loss_mask.to(device, non_blocking=True)[..., None]


def normalize_native_sample(raw, max_len):
    import torch

    ids, mask = raw["input_ids"], raw["loss_mask"]
    aux, logits, start = (
        raw["aux_hidden_state"],
        raw["native_logits"],
        raw["native_logit_start"],
    )
    if ids.ndim != 1 or mask.shape != ids.shape:
        raise ValueError("native stored IDs and mask must have shape [sequence]")
    if aux.ndim == 3 and aux.shape[0] == 1:
        aux = aux[0]
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if (
        aux.ndim != 2
        or aux.shape[0] != ids.numel()
        or not aux.shape[1]
        or aux.shape[1] % 3
        or not aux.is_floating_point()
        or aux.requires_grad
        or not torch.isfinite(aux).all()
    ):
        raise ValueError(
            "native auxiliary states must be finite detached [sequence, 3*hidden]"
        )
    if (
        not isinstance(start, torch.Tensor)
        or start.ndim != 0
        or start.dtype != torch.int64
    ):
        raise ValueError("stored native_logit_start must be a scalar int64 tensor")
    if logits.ndim != 2:
        raise ValueError("stored native logits must have shape [suffix, vocab]")
    _validate_native_logits(
        ids[None],
        logits[None],
        mask[None],
        start.reshape(1),
        vocab_size=logits.shape[-1],
    )
    if type(max_len) is not int or max_len < 2:
        raise ValueError("native max_len must be an integer >= 2")
    length = min(ids.numel(), max_len)
    first = start.item()
    if first >= length:
        raise ValueError("native truncation removes the captured suffix")
    mask = mask[:length].clone()
    mask[-1] = 0
    if not mask.any():
        raise ValueError("native truncation removes all supervision")
    return {
        "input_ids": ids[:length][None],
        "loss_mask": mask[None],
        "attention_mask": torch.ones_like(mask[None], dtype=torch.long),
        "hidden_state": aux[:length][None],
        "target": logits[: length - first][None],
        "native_logit_start": start.reshape(1),
    }


def build_offline_normalizer(max_len, *, ttt_length=1, use_usp_preprocess=False):
    from draftfit.algorithms.eagle3.data import (
        build_offline_normalizer as dense_builder,
    )

    dense = dense_builder(
        max_len, ttt_length=ttt_length, use_usp_preprocess=use_usp_preprocess
    )

    def normalize(raw):
        if "native_logits" in raw:
            if use_usp_preprocess:
                raise ValueError("native logits do not support USP preprocessing")
            return normalize_native_sample(raw, max_len)
        return dense(raw)

    return normalize


def build_offline_collator():
    import torch

    from draftfit.algorithms.eagle3.data import build_offline_collator as dense_builder

    dense = dense_builder()

    def collate(samples):
        native = ["native_logit_start" in sample for sample in samples]
        if any(native) and not all(native):
            raise ValueError("cannot mix native logits and hidden-state samples")
        if all(native) and dense.sp_degree != 1:
            raise ValueError("native logits do not support sequence parallelism")
        batch = dense(samples)
        if all(native):
            batch["native_logit_start"] = torch.cat(
                [s["native_logit_start"] for s in samples]
            )
            width = max(s["target"].shape[1] for s in samples)
            batch["target"] = batch["target"][:, :width]
        return batch

    return collate


def save_native_logits_features(
    output_path,
    samples,
    *,
    target_model,
    target_revision,
    target_hidden_size,
    target_vocab_size,
    capture_layers,
    capture_backend,
    capture_settings=None,
    target_quantization=None,
):
    """Save external native-logit artifacts and their existing capture manifest.

    The producer must supply exact token order, three post-block taps, native
    full-vocabulary logits, and runtime identity in capture_settings. No backend
    execution or native numerical correctness is established by this writer.
    The output directory must be new. Failed writes have no published manifest.
    """
    import torch

    from draftfit.inference.capture_manifest import (
        build_capture_manifest,
        write_capture_manifest,
    )

    if any(
        type(size) is not int or size <= 0
        for size in (target_hidden_size, target_vocab_size)
    ):
        raise ValueError(
            "native capture hidden and vocabulary sizes must be positive integers"
        )
    layers = list(capture_layers)
    if len(layers) != 3 or len(set(layers)) != 3 or layers != sorted(layers):
        raise ValueError("native EAGLE4 requires three distinct ordered capture layers")
    manifest = build_capture_manifest(
        strategy="eagle4",
        capture_method="eagle3",
        capture_layers=layers,
        feature_names=sorted(NATIVE_KEYS),
        target_model=target_model,
        target_revision=target_revision,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=target_vocab_size,
        target_repr="native_logits",
        target_feature="native_logits",
        capture_backend=capture_backend,
        capture_settings=capture_settings,
        target_quantization=target_quantization,
    )
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    for index, raw in enumerate(samples):
        if set(raw) != NATIVE_KEYS:
            raise ValueError(f"native feature keys must be {sorted(NATIVE_KEYS)}")
        sample = normalize_native_sample(raw, raw["input_ids"].numel())
        if (
            sample["target"].shape[-1] != target_vocab_size
            or sample["hidden_state"].shape[-1] != 3 * target_hidden_size
        ):
            raise ValueError(
                "native feature dimensions differ from the capture manifest"
            )
        torch.save(
            {key: value.detach().cpu() for key, value in raw.items()},
            output / f"{index:08d}.ckpt",
        )
        count += 1
    if not count:
        raise ValueError("native feature samples must be nonempty")
    write_capture_manifest(str(output / "capture_manifest.json"), manifest)
    return manifest
