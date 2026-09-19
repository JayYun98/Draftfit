"""Explicit immutable catalog of Draftfit algorithms."""

from __future__ import annotations

from draftfit.algorithms.dflash2.providers import create_registration as dflash2
from draftfit.algorithms.dflash.providers import create_registration as dflash
from draftfit.algorithms.domino.providers import create_registration as domino
from draftfit.algorithms.dspark.providers import create_registration as dspark
from draftfit.algorithms.eagle3.providers import create_registration as eagle3
from draftfit.algorithms.peagle.providers import create_registration as peagle
from draftfit.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry(
        (eagle3(), peagle(), dflash(), dflash2(), domino(), dspark())
    )


__all__ = ["builtin_algorithm_registry"]
