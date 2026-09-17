"""Explicit immutable catalog of Speculative Train Platform algorithms."""

from __future__ import annotations

from speculative_train_platform.algorithms.dflash2.providers import (
    create_registration as dflash2,
)
from speculative_train_platform.algorithms.dflash.providers import (
    create_registration as dflash,
)
from speculative_train_platform.algorithms.domino.providers import (
    create_registration as domino,
)
from speculative_train_platform.algorithms.dspark.providers import (
    create_registration as dspark,
)
from speculative_train_platform.algorithms.eagle3.providers import (
    create_registration as eagle3,
)
from speculative_train_platform.algorithms.peagle.providers import (
    create_registration as peagle,
)
from speculative_train_platform.algorithms.registry import AlgorithmRegistry


def builtin_algorithm_registry() -> AlgorithmRegistry:
    """Return a fresh immutable catalog without module-level mutation."""

    return AlgorithmRegistry(
        (eagle3(), peagle(), dflash(), dflash2(), domino(), dspark())
    )


__all__ = ["builtin_algorithm_registry"]
