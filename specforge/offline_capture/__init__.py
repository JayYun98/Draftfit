"""Dependency-lazy offline teachers; online serving is a separate boundary."""

from importlib import import_module


def load_offline_capture(model_path, *, backend="sglang", **kwargs):
    """Select a real capture implementation without importing other engines."""
    if backend == "sglang":
        from .sglang import load_offline_capture as load

        return load(model_path, **kwargs)
    if backend == "transformers":
        from .transformers import OfflineTransformersCapture

        return OfflineTransformersCapture.from_pretrained(model_path, **kwargs)
    if backend == "vllm":
        from .vllm import OfflineVLLMCapture

        return OfflineVLLMCapture.from_pretrained(model_path, **kwargs)
    raise ValueError(f"unsupported offline teacher backend: {backend!r}")

__all__ = [
    "OfflineCaptureBatch",
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "OfflineSGLangCapture",
    "load_offline_capture",
    "load_offline_eagle3_capture",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.sglang"), name)
    globals()[name] = value
    return value
