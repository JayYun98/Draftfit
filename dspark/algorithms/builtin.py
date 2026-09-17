"""Explicit immutable catalog of DSpark built-in algorithms."""

from __future__ import annotations

from dspark.algorithms.dflash2.providers import create_registration as dflash2
from dspark.algorithms.dflash.providers import create_registration as dflash
from dspark.algorithms.domino.providers import create_registration as domino
from dspark.algorithms.dspark.providers import create_registration as dspark
from dspark.algorithms.eagle3.providers import create_registration as eagle3
from dspark.algorithms.peagle.providers import create_registration as peagle
from dspark.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry(
        (eagle3(), peagle(), dflash(), dflash2(), domino(), dspark())
    )


__all__ = ["builtin_algorithm_registry"]
