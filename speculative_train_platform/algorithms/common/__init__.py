"""Shared provider ports and small data adapters for built-in algorithms."""

from speculative_train_platform.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerInputAdapter,
    ServerStreamingProvider,
    StepProvider,
    StepRuntimeConfig,
    TargetDerivedDraftDefaults,
    make_registration,
)

__all__ = [
    "AlgorithmProviders",
    "DraftConfigProvider",
    "ModelProvider",
    "OfflineCaptureLayout",
    "OfflineDataProvider",
    "ServerCaptureLayout",
    "ServerInputAdapter",
    "ServerStreamingProvider",
    "StepProvider",
    "StepRuntimeConfig",
    "TargetDerivedDraftDefaults",
    "make_registration",
]
