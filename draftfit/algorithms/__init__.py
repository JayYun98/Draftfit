"""Topology-free algorithm contracts and explicit registrations."""

from draftfit.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from draftfit.algorithms.registry import AlgorithmRegistration, AlgorithmRegistry

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
