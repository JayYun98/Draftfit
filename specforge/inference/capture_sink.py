# Copyright 2024 SGLang Team
# Copyright 2026 The SpecForge contributors
# Licensed under the Apache License, Version 2.0.
"""Owned synchronous form of SpecCaptureSink in our SGLang v0.5.14 patch.

Preserves raw {store_id}/{sample_id}/g{gen}/{feature} objects, metadata-only
responses and explicit retry replacement. MooncakeFeatureStore owns raw-buffer
registration and status handling; this sink does not create local SampleRefs.
"""

import math
import re

import torch

from specforge.runtime.data_plane.mooncake_store import _TORCH_DTYPES


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value):
        raise ValueError("capture namespace and feature names must be 1–256 safe characters")
    return value


class CaptureSink:
    def __init__(self, transport, *, aux_layer_ids):
        if getattr(transport._put_config, "with_hard_pin", False) is not True:
            raise ValueError("native teacher capture requires Mooncake hard pin support")
        self.transport = transport
        self.aux_layer_ids = list(aux_layer_ids)

    def validate(self, spec, length):
        """Validate all metadata before inference or a store mutation."""
        if not isinstance(spec, dict):
            raise ValueError("spec_capture entries must be objects")
        _name(spec.get("store_id"))
        if spec["store_id"] != self.transport.store_id:
            raise ValueError("capture store_id differs from configured store namespace")
        _name(spec.get("sample_id"))
        if type(spec.get("gen")) is not int or spec["gen"] != 1:
            raise ValueError("native teacher requires stable generation gen=1")
        if type(spec.get("replace", False)) is not bool:
            raise ValueError("replace must be boolean")
        features = spec.get("features")
        if not isinstance(features, dict) or not features or set(features) - {"aux", "last_hidden"}:
            raise ValueError("features must request aux and/or last_hidden")
        names = [_name(name) for name in features.values()]
        passthrough = spec.get("passthrough", [])
        if not isinstance(passthrough, list) or len(passthrough) > 16:
            raise ValueError("passthrough must be a list of at most 16 tensors")
        for item in passthrough:
            if not isinstance(item, dict):
                raise ValueError("passthrough tensor must be an object")
            names.append(_name(item.get("name")))
            shape = item.get("shape")
            if (not isinstance(shape, list) or not 2 <= len(shape) <= 4
                    or any(type(d) is not int or d <= 0 for d in shape)
                    or shape[:2] != [1, length] or math.prod(shape) > length * 64):
                raise ValueError("passthrough shape must start [1, prompt_length] and have <=64 values per token")
            dtype = _TORCH_DTYPES.get(item.get("dtype"))
            if dtype is None or not isinstance(item.get("data"), list):
                raise ValueError("unsupported passthrough dtype or data")
            pending = list(item["data"])
            while pending:
                value = pending.pop()
                if isinstance(value, list):
                    pending.extend(value)
                elif dtype == torch.bool:
                    if value not in (0, 1) or type(value) not in (int, bool):
                        raise ValueError("boolean passthrough requires boolean or 0/1 values")
                elif not dtype.is_floating_point and type(value) is not int:
                    raise ValueError("integer passthrough requires integer values")
            try:
                tensor = torch.tensor(item["data"], dtype=dtype)
            except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
                raise ValueError("invalid passthrough data") from exc
            if tensor.numel() != math.prod(shape) or not torch.isfinite(tensor).all():
                raise ValueError("passthrough data does not match shape or contains nonfinite values")
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")

    def _remove(self, keys):
        pending = list(keys)
        for _ in range(3):
            failed = []
            for key in pending:
                try:
                    exists = self.transport._store.is_exist(key)
                    if exists != 0 and (exists != 1 or not self.transport._store_remove(key)):
                        failed.append(key)
                except Exception:
                    failed.append(key)
            pending = failed
            if not pending:
                return
        raise RuntimeError(f"capture cleanup failed for {len(pending)} keys; client provisional cleanup must retry")

    def put_sample(self, spec, *, aux, last_hidden):
        length = aux.shape[0] if aux is not None else last_hidden.shape[0]
        self.validate(spec, length)
        tensors = {}
        for artifact, tensor in (("aux", aux), ("last_hidden", last_hidden)):
            if artifact in spec["features"]:
                if tensor is None or tensor.ndim != 2 or tensor.shape[0] != length or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                    raise ValueError("invalid captured tensor")
                tensors[spec["features"][artifact]] = tensor.detach().cpu().unsqueeze(0).contiguous()
        for item in spec.get("passthrough", []):
            tensors[item["name"]] = torch.tensor(item["data"], dtype=_TORCH_DTYPES[item["dtype"]]).reshape(item["shape"]).contiguous()
        keys = {name: f"{spec['store_id']}/{spec['sample_id']}/g{spec['gen']}/{name}" for name in tensors}
        if spec.get("replace", False):
            self._remove(list(keys.values()))
        else:
            for key in keys.values():
                exists = self.transport._store.is_exist(key)
                if exists == 1:
                    raise ValueError("capture keys already exist; retry with replace=true")
                if exists != 0:
                    raise RuntimeError("capture key existence check failed")
        try:
            for name, tensor in tensors.items():
                self.transport._store_put_tensor(keys[name], tensor)
        except BaseException:
            self._remove(list(keys.values()))
            raise
        return {
            "store_id": spec["store_id"], "sample_id": spec["sample_id"], "gen": spec["gen"],
            "aux_layer_ids": self.aux_layer_ids,
            "features": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
                         for name, tensor in tensors.items()},
        }
