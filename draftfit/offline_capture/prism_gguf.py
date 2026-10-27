"""Experimental exact GGUF capture through a separately built, pinned C bridge.

The native target remains resident. Each row starts with empty hybrid state;
this is a full-prefix reference capture path, not a KV-cached serving engine.
CPU mode selects no model GPU and disables context offload. Global backend
discovery still runs and may probe installed GPU backends.
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
from pathlib import Path

import numpy as np
import torch

from draftfit.inference.capture import TeacherCaptureBatch

PRISM_REVISION = "e3de715222b2667c54a267441159a5729f11155e"


class OfflinePrismGGUFCapture:
    def __init__(
        self,
        path,
        *,
        bridge_path,
        expected_sha256,
        device="cpu",
        torch_dtype=torch.float32,
        context_size=1024,
        chunk_size=32,
        expected_bridge_sha256=None,
    ):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file() or self.path.suffix.lower() != ".gguf":
            raise ValueError("Prism capture requires a local GGUF file")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)
        ):
            raise ValueError("Prism capture requires the exact GGUF SHA256")
        self.expected_sha256 = expected_sha256
        if self._sha256(self.path) != expected_sha256:
            raise ValueError("GGUF SHA256 mismatch")
        # Reject wrong profile/geometry before loading a native library or weights.
        from draftfit.modeling.target.gguf_parts import (
            _validate_bonsai,
            read_gguf_metadata,
        )

        header = read_gguf_metadata(self.path)
        _validate_bonsai(header)
        self.hidden_size = header.metadata["qwen35.embedding_length"]
        self.vocab_size = header.metadata["tokenizer.ggml.tokens"]
        self.num_hidden_layers = header.metadata["qwen35.block_count"]
        if torch_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unsupported capture tensor dtype")
        if (
            type(context_size) is not int
            or not 64 <= context_size <= 8192
            or type(chunk_size) is not int
            or not 1 <= chunk_size <= 512
        ):
            raise ValueError("invalid native context/chunk size")
        self.dtype, self.context_size, self.chunk_size = (
            torch_dtype,
            context_size,
            chunk_size,
        )
        self.native_device = str(device)
        if self.native_device not in ("cpu", "cuda", "cuda:0", "mps"):
            raise ValueError("Prism capture supports CPU or the first local GPU only")
        self.bridge_path = Path(bridge_path).expanduser().resolve(strict=True)
        if not self.bridge_path.is_file():
            raise ValueError("Prism capture requires a local bridge library file")
        self.bridge_sha256 = self._sha256(self.bridge_path)
        if (
            expected_bridge_sha256 is not None
            and self.bridge_sha256 != expected_bridge_sha256
        ):
            raise ValueError("Prism bridge SHA256 mismatch")
        self.lib = ctypes.CDLL(str(self.bridge_path))
        self.lib.draftfit_prism_revision.restype = ctypes.c_char_p
        if self.lib.draftfit_prism_revision().decode() != PRISM_REVISION:
            raise ValueError("Prism bridge revision mismatch")
        self.lib.draftfit_prism_error.restype = ctypes.c_char_p
        self.lib.draftfit_prism_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.draftfit_prism_open.restype = ctypes.c_void_p
        self.lib.draftfit_prism_close.argtypes = [ctypes.c_void_p]
        self.lib.draftfit_prism_close.restype = None
        self.lib.draftfit_prism_geometry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.draftfit_prism_geometry.restype = ctypes.c_int
        self.lib.draftfit_prism_capture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.draftfit_prism_capture.restype = ctypes.c_int
        self.capture_layers = None
        self._handle = None
        self._lock = threading.Lock()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _verify_identity(self):
        if self._sha256(self.path) != self.expected_sha256:
            raise ValueError(
                "GGUF SHA256 mismatch before native capture initialization"
            )
        if self._sha256(self.bridge_path) != self.bridge_sha256:
            raise ValueError("Prism bridge changed before capture initialization")

    def _native_error(self):
        message = self.lib.draftfit_prism_error()
        return (
            message.decode("utf-8", errors="replace")
            if message
            else "unknown Prism capture error"
        )

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(path, **kwargs)

    def set_capture_layers(self, layer_ids, *, capture_method="dspark"):
        layers = list(layer_ids)
        if (
            capture_method not in {"dspark", "dflash"}
            or not layers
            or any(
                type(i) is not int or not 0 <= i < self.num_hidden_layers
                for i in layers
            )
            or layers != sorted(set(layers))
        ):
            raise ValueError(
                "Prism taps must be unique ascending post-block IDs in [0,64)"
            )
        with self._lock:
            if self._handle and layers != self.capture_layers:
                raise ValueError(
                    "cannot change native taps after context initialization"
                )
            self.capture_layers = layers

    def _open(self):
        if self._handle:
            return
        if self.capture_layers is None:
            raise ValueError("set_capture_layers must precede capture")
        self._verify_identity()
        if self.native_device.startswith("cuda"):
            # ggml has its own allocator; release unused Torch cache before it
            # reserves the resident target beside the already-built drafts.
            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()
        taps = np.array(self.capture_layers, dtype=np.int32)
        handle = self.lib.draftfit_prism_open(
            str(self.path).encode(),
            taps.ctypes.data,
            len(taps),
            self.context_size,
            self.chunk_size,
            int(self.native_device != "cpu"),
        )
        if not handle:
            raise RuntimeError(self._native_error())
        try:
            dims = np.empty(3, dtype=np.int32)
            code = self.lib.draftfit_prism_geometry(handle, dims.ctypes.data)
            if code or tuple(dims) != (
                self.hidden_size,
                self.vocab_size,
                self.num_hidden_layers,
            ):
                raise ValueError(
                    "Prism target geometry differs from Bonsai Qwen35 contract"
                )
            self._verify_identity()
        except BaseException:
            self.lib.draftfit_prism_close(handle)
            raise
        self._handle = handle

    @torch.no_grad()
    def capture(self, *, input_ids, attention_mask, loss_mask, return_logits=False):
        ids = input_ids.detach().cpu()
        if (
            ids.ndim != 2
            or not all(ids.shape)
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "input_ids must be a nonempty integer [batch,sequence] tensor"
            )
        if torch.any(ids < 0) or torch.any(ids >= self.vocab_size):
            raise ValueError("input_ids outside target vocabulary")
        attention, loss = attention_mask.detach().cpu(), loss_mask.detach().cpu()
        for name, mask in (("attention_mask", attention), ("loss_mask", loss)):
            if mask.shape != ids.shape or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{name} must be binary and match input_ids")
        if torch.any(loss.bool() & ~attention.bool()):
            raise ValueError("loss_mask cannot supervise padding")
        if type(return_logits) is not bool:
            raise ValueError("return_logits must be boolean")
        positions = []
        for mask in attention:
            pos = mask.nonzero().flatten()
            if (
                not len(pos)
                or int(pos[-1] - pos[0] + 1) != len(pos)
                or len(pos) > self.context_size
            ):
                raise ValueError(
                    "attention must select a nonempty contiguous sequence within native context"
                )
            positions.append(pos)
        with self._lock:
            if self.capture_layers is None:
                raise ValueError("set_capture_layers must precede capture")
            shape = (*ids.shape, self.hidden_size)
            aux = torch.zeros(
                *ids.shape,
                len(self.capture_layers) * self.hidden_size,
                dtype=self.dtype,
            )
            hidden = torch.zeros(shape, dtype=self.dtype)
            logits = (
                torch.zeros(*ids.shape, self.vocab_size, dtype=torch.float32)
                if return_logits
                else None
            )
            self._open()
            for row, pos in enumerate(positions):
                tokens = ids[row, pos].numpy().astype(np.int32, copy=False)
                a = np.empty(
                    (len(pos), len(self.capture_layers) * self.hidden_size),
                    dtype=np.float32,
                )
                h = np.empty((len(pos), self.hidden_size), dtype=np.float32)
                l = (
                    np.empty((len(pos), self.vocab_size), dtype=np.float32)
                    if return_logits
                    else None
                )
                result = self.lib.draftfit_prism_capture(
                    self._handle,
                    tokens.ctypes.data,
                    len(tokens),
                    a.ctypes.data,
                    h.ctypes.data,
                    l.ctypes.data if l is not None else None,
                )
                if result:
                    raise RuntimeError(self._native_error())
                if not all(
                    np.isfinite(value).all() for value in (a, h) if value is not None
                ) or (l is not None and not np.isfinite(l).all()):
                    raise ValueError("native capture produced non-finite values")
                converted_aux = torch.from_numpy(a).to(self.dtype)
                converted_hidden = torch.from_numpy(h).to(self.dtype)
                if (
                    not torch.isfinite(converted_aux).all()
                    or not torch.isfinite(converted_hidden).all()
                ):
                    raise ValueError("native features overflow capture tensor dtype")
                aux[row, pos] = converted_aux
                hidden[row, pos] = converted_hidden
                if l is not None:
                    logits[row, pos] = torch.from_numpy(l)
        batch = TeacherCaptureBatch(
            aux, hidden, ids.clone(), attention.clone(), loss.clone()
        )
        batch.native_logits = logits
        return batch

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if self._handle:
                    self.lib.draftfit_prism_close(self._handle)
                    self._handle = None

    def __del__(self):
        self.close()
