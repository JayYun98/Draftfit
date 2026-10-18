"""Bounded, synchronous token-to-feature source for joint family training.

Token rows remain recoverable; prefetched features live only in the bounded RAM
queue until their joint update succeeds. Features are captured once per requested
microbatch, copied to CPU, and normalized with the offline family adapters.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from draftfit.algorithms.common.dflash_family_data import (
    build_collator,
    build_dspark_collator,
    normalize_dspark_offline_sample,
    normalize_offline_sample,
)
from draftfit.data.loss_mask import has_consecutive_supervised_tokens
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.runtime.contracts import TrainBatch
from draftfit.utils import get_local_device as get_device


@contextmanager
def _preserve_rng(device):
    """Teacher loading/capture must not advance the draft's random streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    module = getattr(torch, device.type, None)
    accelerator_state = None
    if device.type in {"cuda", "npu"}:
        accelerator_state = module.get_rng_state_all()
    elif device.type == "mps":
        accelerator_state = module.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if accelerator_state is not None:
            if device.type == "mps":
                module.set_rng_state(accelerator_state)
            else:
                module.set_rng_state_all(accelerator_state)


def _capture_layers(stage):
    override = stage.config.model.target_layer_ids
    return tuple(
        override
        if override is not None
        else stage.draft_config.dflash_config["target_layer_ids"]
    )


def _token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _read_rows(path, max_length):
    if not path:
        raise ValueError("joint online requires train and eval tokenized JSONL paths")
    path = Path(path).expanduser().resolve()
    rows, hashes, identifiers = [], set(), set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"{where}: invalid tokenized JSONL") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{where}: expected a tokenized JSON object")
            if {"messages", "conversations", "text", "prompt", "response"} & raw.keys():
                raise ValueError(f"{where}: raw chat/text is not supported")
            ids = raw.get("input_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(
                    type(value) is not int or not 0 <= value < 2**63 for value in ids
                )
            ):
                raise ValueError(
                    f"{where}: input_ids must be nonempty nonnegative integers"
                )
            if len(ids) > max_length:
                raise ValueError(
                    f"{where}: input_ids exceeds max_length; truncation is forbidden"
                )
            loss = raw.get("loss_mask")
            attention = raw.get("attention_mask", [1] * len(ids))
            for name, mask in (("loss_mask", loss), ("attention_mask", attention)):
                if (
                    not isinstance(mask, list)
                    or len(mask) != len(ids)
                    or any(
                        type(value) not in (int, bool) or value not in (0, 1)
                        for value in mask
                    )
                ):
                    raise ValueError(
                        f"{where}: {name} must be binary and match input_ids"
                    )
            length = sum(attention)
            if not length or attention != [1] * length + [0] * (len(ids) - length):
                raise ValueError(
                    f"{where}: attention_mask must be contiguous right padding"
                )
            if any(loss[length:]):
                raise ValueError(f"{where}: loss_mask cannot supervise padding")
            if not any(loss):
                raise ValueError(f"{where}: loss_mask needs positive supervised tokens")
            if not has_consecutive_supervised_tokens(loss):
                raise ValueError(
                    f"{where}: family samples require two consecutive supervised tokens"
                )
            # Strip only explicit padding, never truncate attended tokens. Hash
            # token content independently of row IDs, loss masks and padding.
            hashes.add(_token_hash(ids))
            ids, loss = ids[:length], loss[:length]
            content_hash = _token_hash(ids)
            hashes.add(content_hash)
            for key in ("id", "sample_id"):
                if key in raw:
                    value = raw[key]
                    if type(value) not in (int, str) or value == "":
                        raise ValueError(
                            f"{where}: {key} must be a nonempty string or integer"
                        )
                    identifiers.add(str(value))
            domain = raw.get("domain", "unlabeled")
            if not isinstance(domain, str) or not domain:
                raise ValueError(f"{where}: domain must be a nonempty string")
            rows.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long, device="cpu"),
                    "loss_mask": torch.tensor(loss, dtype=torch.long, device="cpu"),
                    "domain": domain,
                }
            )
    if not rows:
        raise ValueError(f"{path}: tokenized JSONL must be nonempty")
    return (
        rows,
        hashes,
        identifiers,
        {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "num_samples": len(rows),
        },
    )


class JointOnlineSource:
    """One lazy frozen teacher shared by training and held-out loaders.

    This source is synchronous and intended for the single-process joint loop.
    ``capture_calls`` and ``captured_samples`` count successful captures only.
    """

    def __init__(self, stage, needs_dspark: bool):
        cfg = stage.config
        self.needs_dspark = needs_dspark
        self._max_length = cfg.data.max_length
        self._layers = _capture_layers(stage)
        model = cfg.model
        self._model_path = model.target_model_path
        self._target_device = (
            torch.device(cfg.training.joint_target_device)
            if cfg.training.joint_target_device
            else None
        )
        self._model_kwargs = {
            "revision": model.target_revision,
            "trust_remote_code": model.trust_remote_code,
            "torch_dtype": getattr(torch, model.torch_dtype),
            "cache_dir": model.cache_dir,
            "attn_implementation": model.hf_attn_implementation,
        }
        train, train_hashes, train_ids, train_info = _read_rows(
            cfg.data.prompts_path,
            self._max_length,
        )
        evaluation, eval_hashes, eval_ids, eval_info = _read_rows(
            cfg.data.eval_data_path,
            self._max_length,
        )
        if train_hashes & eval_hashes:
            raise ValueError(
                "joint online train/eval overlap: duplicate input_ids content"
            )
        if train_ids & eval_ids:
            raise ValueError("joint online train/eval overlap: duplicate sample ID")
        self._rows = {False: train, True: evaluation}
        self._sources = {"train": train_info, "eval": eval_info}
        self._teacher = None
        self._queue_bytes = getattr(cfg.training, "joint_feature_queue_bytes", 0)
        self._queue_batches = getattr(cfg.training, "joint_feature_queue_batches", 2)
        self._feature_width = getattr(
            stage.draft_config, "target_hidden_size", None
        ) or getattr(stage.draft_config, "hidden_size", 0)
        self._feature_itemsize = torch.tensor(
            [], dtype=getattr(torch, model.torch_dtype)
        ).element_size()
        self._leases = {}
        self._queues = []
        self._queue = None
        self.consumed_samples = 0
        self.capture_calls = 0
        self.captured_samples = 0
        self.capture_seconds = 0.0
        self.capture_peak_allocated_bytes = None
        self.capture_peak_reserved_bytes = None

    def queue_metrics(self):
        keys = (
            "resident_bytes",
            "peak_bytes",
            "resident_items",
            "produced",
            "consumed",
            "cancelled",
            "backpressure",
        )
        return {
            key: (
                max((q.metrics()[key] for q in self._queues), default=0)
                if key == "peak_bytes"
                else sum(q.metrics()[key] for q in self._queues)
            )
            for key in keys
        }

    def consume(self, batches):
        """Called only after every family backward and the optimizer update."""
        for batch in batches:
            owned = self._leases.pop(id(batch), None)
            if owned:
                queue, lease = owned
                queue.release(lease, consumed=True)
            self.consumed_samples += len(batch.sample_ids)

    def close_queues(self):
        for queue in self._queues:
            queue.clear()
        self._leases.clear()

    def _batch_bytes(self, rows, indices):
        length = max(rows[index]["input_ids"].numel() for index in indices)
        features = len(self._layers) + int(self.needs_dspark)
        # IDs, supervision mask and collated attention mask are int64.
        return (
            len(indices)
            * length
            * (features * self._feature_width * self._feature_itemsize + 3 * 8)
        )

    def fingerprint_sources(self):
        """Identity of the bytes validated at construction, safe for JSON."""
        return {split: dict(info) for split, info in self._sources.items()}

    def loader(self, stage, epoch, shuffle, drop_last, evaluation=False):
        if (
            stage.config.data.max_length != self._max_length
            or _capture_layers(stage) != self._layers
        ):
            raise ValueError(
                "joint online stages must share max_length and capture layers"
            )
        rows = self._rows[evaluation]
        indices = list(range(len(rows)))
        if shuffle:
            # Same single-process order as the offline DistributedSampler.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stage.config.training.seed + epoch)
            indices = torch.randperm(len(rows), generator=generator).tolist()
        return _TokenLoader(
            self, rows, indices, stage.config.training.batch_size, drop_last, evaluation
        ), len(rows)

    def _capture_batch(self, rows, indices, evaluation):
        selected = [rows[index] for index in indices]
        length = max(row["input_ids"].numel() for row in selected)
        ids = torch.zeros((len(selected), length), dtype=torch.long, device="cpu")
        loss, attention = torch.zeros_like(ids), torch.zeros_like(ids)
        for index, row in enumerate(selected):
            size = row["input_ids"].numel()
            ids[index, :size] = row["input_ids"]
            loss[index, :size] = row["loss_mask"]
            attention[index, :size] = 1
        device = self._target_device or get_device()
        with _preserve_rng(device):
            if self._teacher is None:
                teacher = OfflineTransformersCapture.from_pretrained(
                    self._model_path,
                    device=device,
                    **self._model_kwargs,
                )
                teacher.set_capture_layers(
                    list(self._layers),
                    capture_method="dspark" if self.needs_dspark else "dflash",
                )
                self._teacher = teacher
            started = time.perf_counter()
            captured = self._teacher.capture(
                input_ids=ids,
                attention_mask=attention,
                loss_mask=loss,
            )
            hidden = captured.hidden_states.detach().to("cpu")
            last = (
                captured.last_hidden_states.detach().to("cpu")
                if self.needs_dspark
                else None
            )
            # CPU copies synchronize capture before recording its elapsed time.
            del captured
            if device.type == "cuda":
                # The parent owns peak resets. Preserve earlier eval peaks even
                # when the training window has since reset allocator statistics.
                self.capture_peak_allocated_bytes = max(
                    self.capture_peak_allocated_bytes or 0,
                    torch.cuda.max_memory_allocated(device),
                )
                self.capture_peak_reserved_bytes = max(
                    self.capture_peak_reserved_bytes or 0,
                    torch.cuda.max_memory_reserved(device),
                )
            self.capture_seconds += time.perf_counter() - started
            self.capture_calls += 1
            self.captured_samples += len(selected)
        normalizer = (
            normalize_dspark_offline_sample
            if self.needs_dspark
            else normalize_offline_sample
        )
        features = []
        for index, row in enumerate(selected):
            size = row["input_ids"].numel()
            raw = {**row, "hidden_states": hidden[index, :size]}
            if last is not None:
                raw["target_last_hidden_states"] = last[index, :size]
            features.append(normalizer(raw, self._max_length))
        collator = build_dspark_collator() if self.needs_dspark else build_collator()
        split = "eval" if evaluation else "train"
        return TrainBatch(
            sample_ids=[f"joint-online:{split}:{index:08d}" for index in indices],
            strategy="dspark" if self.needs_dspark else "dflash",
            tensors=collator(features),
            metadata={"target_repr": "hidden_state" if self.needs_dspark else None},
        )


class _TokenLoader:
    def __init__(self, source, rows, indices, batch_size, drop_last, evaluation):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.source, self.rows, self.indices = source, rows, indices
        self.batch_size, self.drop_last, self.evaluation = (
            batch_size,
            drop_last,
            evaluation,
        )
        self._seek_batches = 0

    def __len__(self):
        count = len(self.indices)
        return (
            count // self.batch_size if self.drop_last else -(-count // self.batch_size)
        )

    def seek(self, batch_offset):
        """Skip batches on the next iteration only, without touching the teacher."""
        if type(batch_offset) is not int or not 0 <= batch_offset <= len(self):
            raise ValueError(f"seek offset must be between 0 and {len(self)} batches")
        self._seek_batches = batch_offset

    def __iter__(self):
        start, self._seek_batches = self._seek_batches, 0
        if not self.source._queue_bytes or self.evaluation:
            for batch in range(start, len(self)):
                offset = batch * self.batch_size
                yield self.source._capture_batch(
                    self.rows,
                    self.indices[offset : offset + self.batch_size],
                    self.evaluation,
                )
            return
        from draftfit.training.feature_queue import BoundedFeatureQueue

        if self.source._queue is None:
            self.source._queue = BoundedFeatureQueue(
                self.source._queue_bytes, self.source._queue_batches
            )
            self.source._queues.append(self.source._queue)
        queue = self.source._queue
        next_batch = start
        try:
            for _ in range(start, len(self)):
                while next_batch < len(self) and queue.resident_items < queue.max_items:
                    offset = next_batch * self.batch_size
                    indices = self.indices[offset : offset + self.batch_size]
                    expected = self.source._batch_bytes(self.rows, indices)
                    if expected > queue.max_bytes:
                        raise ValueError(
                            "feature batch exceeds joint queue byte budget"
                        )
                    if not queue.can_fit(expected):
                        queue.backpressure += 1
                        break
                    captured = self.source._capture_batch(self.rows, indices, False)
                    queue.publish(captured)
                    del captured
                    next_batch += 1
                if not queue.ready:
                    raise ValueError(
                        "joint queue byte budget cannot hold accumulation window; increase bytes or reduce batch/accumulation"
                    )
                lease = queue.acquire()
                self.source._leases[id(lease.value)] = (queue, lease)
                yield lease.value
        finally:
            # Unconsumed prefetch is reproducible from token rows and cursor.
            # Leases may still belong to an accumulation window crossing the
            # epoch boundary. They keep source-wide credit until optimizer ACK.
            queue.cancel_ready()
