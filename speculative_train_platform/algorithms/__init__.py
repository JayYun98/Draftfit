"""Topology-free algorithm contracts and explicit registrations."""

from speculative_train_platform.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from speculative_train_platform.algorithms.registry import (
    AlgorithmRegistration,
    AlgorithmRegistry,
)

__all__ = [
    "AlgorithmCapabilities",
    "AlgorithmRegistration",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "DraftRequirement",
    "FeatureContract",
    "FeatureMode",
    "OfflineStorageContract",
]
