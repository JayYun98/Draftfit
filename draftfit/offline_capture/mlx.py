"""Bounded local MLX Qwen hybrid teacher; no GPU/backend parity claim."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import numpy as np
import torch

from draftfit.inference.capture import TeacherCaptureBatch


class OfflineMLXCapture:
    """Post-block, pre-final-norm taps with independent unpadded sequences.

    Padding is removed before each forward to avoid contaminating DeltaNet state.
    Each row gets fresh recurrent/KV state. This is offline capture, not a
    speculative decoder or an online service.
    """

    def __init__(self, model, *, native_sink=False):
        self._model = getattr(model, "language_model", model)
        args = self._model.args
        if args.model_type not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError(f"unsupported MLX text decoder: {args.model_type}")
        self.native_sink = native_sink
        self._lock = threading.RLock()
        self.hidden_size, self.vocab_size = args.hidden_size, args.vocab_size
        if any(
            type(value) is not int or value <= 0
            for value in (self.hidden_size, self.vocab_size)
        ):
            raise ValueError("MLX target geometry must use positive integer dimensions")
        body = self._model.model
        if not all(hasattr(body, key) for key in ("layers", "norm", "embed_tokens")):
            raise ValueError("MLX target must expose layers, norm and embeddings")
        self.num_hidden_layers = len(body.layers)
        if (
            not self.num_hidden_layers
            or self.num_hidden_layers != args.num_hidden_layers
        ):
            raise ValueError("MLX target layer count differs from config")
        self._model.eval()
        self.capture_layers = None

    @classmethod
    def from_pretrained(cls, path, *, revision=None):
        """Load local weights only; never execute a repository's Python loader."""
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                "MLX capture requires a local model directory, not GGUF or a Hub ID"
            )
        if revision and path.parent.name == "snapshots" and path.name != revision:
            raise ValueError("target revision differs from local snapshot")
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict):
            raise ValueError("MLX config must be an object")
        if config.get("model_file") or config.get("auto_map"):
            raise ValueError("custom model code is not accepted by MLX capture")
        if config.get("model_type") == "prism_hadamard_qwen35":
            from .mlx_hadamard import load_bonsai_text

            return cls(load_bonsai_text(path), native_sink=True)
        if config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError(
                "MLX capture supports Qwen hybrid and reviewed Bonsai packs only"
            )
        from mlx_lm.utils import load_model

        if config["model_type"] == "qwen3_5_text":
            from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

            model, _ = load_model(
                path,
                strict=True,
                get_model_classes=lambda config: (TextModel, TextModelArgs),
            )
        else:
            model, _ = load_model(path, strict=True)
        return cls(model)

    def set_capture_layers(self, layer_ids, *, capture_method="dspark"):
        layers = list(layer_ids)
        depth = len(self._model.model.layers)
        if capture_method not in {"dspark", "dflash"}:
            raise ValueError("MLX capture_method must be dspark or dflash")
        if (
            not layers
            or any(type(i) is not int or not 0 <= i < depth for i in layers)
            or layers != sorted(set(layers))
        ):
            raise ValueError(
                "capture layers must be unique ascending indices within target depth"
            )
        with self._lock:
            self.capture_layers = layers

    def forward_features(self, ids):
        with self._lock:
            return self._forward_features(ids)

    def _forward_features(self, ids):
        """MLX arrays in/out; the native readout remains available for parity checks."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        if self.capture_layers is None:
            raise ValueError("set_capture_layers must be called before capture")
        body = self._model.model
        cache = self._model.make_cache()
        if len(cache) != self.num_hidden_layers:
            raise ValueError("MLX target cache count differs from layer count")
        if self.native_sink:
            states = []
            # Run the runtime's own forward. A temporary callable proxy observes
            # post-block outputs without requiring a patched MLX-VLM sink API.
            original = list(body.layers)

            class Tap:
                def __init__(self, layer):
                    self.layer = layer

                def __getattr__(self, name):
                    return getattr(self.layer, name)

                def __call__(self, *args, **kwargs):
                    h = self.layer(*args, **kwargs)
                    states.append(h)
                    return h

            try:
                for i in self.capture_layers:
                    body.layers[i] = Tap(original[i])
                last = body(ids, cache=cache)
            finally:
                body.layers[:] = original
        else:
            h = body.embed_tokens(ids)
            fa_mask = create_attention_mask(h, cache[body.fa_idx])
            ssm_mask = create_ssm_mask(h, cache[body.ssm_idx])
            states = []
            for i, (layer, c) in enumerate(zip(body.layers, cache)):
                h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
                if i in self.capture_layers:
                    states.append(h)
            last = body.norm(h)
        if len(states) != len(self.capture_layers):
            raise ValueError("MLX target did not return every requested tap")
        aux = mx.concatenate(states, axis=-1)
        mx.eval(aux, last)
        return aux, last

    @torch.no_grad()
    def capture(self, *, input_ids, attention_mask, loss_mask):
        with self._lock:
            return self._capture(
                input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
            )

    def _capture(self, *, input_ids, attention_mask, loss_mask):
        if self.capture_layers is None:
            raise ValueError("set_capture_layers must be called before capture")
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("input_ids must be nonempty [batch, sequence]")
        ids = input_ids.detach().cpu()
        if (
            ids.dtype not in (torch.int32, torch.int64)
            or torch.any(ids < 0)
            or torch.any(ids >= self.vocab_size)
        ):
            raise ValueError("input_ids must be integer IDs inside target vocabulary")
        attention, loss = attention_mask.detach().cpu(), loss_mask.detach().cpu()
        for name, mask in (("attention_mask", attention), ("loss_mask", loss)):
            if mask.shape != ids.shape or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{name} must be binary and match input_ids")
        if torch.any(loss.bool() & ~attention.bool()):
            raise ValueError("loss_mask cannot select padding")
        positions_per_row = []
        for mask in attention:
            positions = mask.nonzero().flatten()
            if not len(positions) or int(positions[-1] - positions[0] + 1) != len(
                positions
            ):
                raise ValueError(
                    "attention_mask must contain nonempty contiguous tokens"
                )
            positions_per_row.append(positions)
        import mlx.core as mx

        aux_rows, last_rows = [], []
        for row, positions in zip(ids, positions_per_row):
            aux, last = self.forward_features(
                mx.array(row[positions].numpy()[None], dtype=mx.int32)
            )
            # Explicit FP32 bridge also supports MLX BF16, which NumPy cannot expose.
            a = torch.from_numpy(np.array(aux.astype(mx.float32), copy=True))[0]
            h = torch.from_numpy(np.array(last.astype(mx.float32), copy=True))[0]
            if a.shape != (
                len(positions),
                len(self.capture_layers) * self.hidden_size,
            ) or h.shape != (len(positions), self.hidden_size):
                raise ValueError("MLX target returned incompatible feature shape")
            if not torch.isfinite(a).all() or not torch.isfinite(h).all():
                raise ValueError("MLX target returned non-finite features")
            padded_aux = torch.zeros(len(row), a.shape[-1])
            padded_last = torch.zeros(len(row), h.shape[-1])
            padded_aux[positions], padded_last[positions] = a, h
            aux_rows.append(padded_aux)
            last_rows.append(padded_last)
        return TeacherCaptureBatch(
            torch.stack(aux_rows),
            torch.stack(last_rows),
            ids.clone(),
            attention.clone(),
            loss.clone(),
        )


def text_config(path):
    config = json.loads((Path(path) / "config.json").read_text())
    return copy.deepcopy(config.get("text_config", config))
