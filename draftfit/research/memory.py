"""Compatibility namespace for the paper-owned memory policies."""

import sys
from types import ModuleType

from . import proposals
from .papers.ngramrl import method as ngramrl
from .papers.specmem import method as specmem


def _binding(module, name):
    # A patch on the old path must update, and restore, the implementation global.
    return property(
        lambda self: getattr(module, name),
        lambda self, value: setattr(module, name, value),
        lambda self: delattr(module, name),
    )


class _MemoryAPI(ModuleType):
    NgramMemory = _binding(ngramrl, "NgramMemory")
    NgramSnapshot = _binding(ngramrl, "NgramSnapshot")
    SpecMemMemory = _binding(specmem, "SpecMemMemory")
    normalized_embedding = _binding(specmem, "normalized_embedding")
    Proposal = _binding(proposals, "Proposal")
    token_tuple = _binding(proposals, "token_tuple")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(__all__))


__all__ = [
    "NgramMemory",
    "NgramSnapshot",
    "Proposal",
    "SpecMemMemory",
    "normalized_embedding",
    "token_tuple",
]

sys.modules[__name__].__class__ = _MemoryAPI
