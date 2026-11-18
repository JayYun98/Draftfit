"""Token validation and proposal distributions shared by reference policies."""

from __future__ import annotations

import math
from dataclasses import dataclass


def token_tuple(tokens):
    values = tuple(tokens)
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError("token IDs must be nonnegative integers")
    return values


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[int, ...] = ()
    probabilities: tuple[dict[int, float], ...] = ()

    def __post_init__(self):
        token_tuple(self.tokens)
        if len(self.tokens) != len(self.probabilities):
            raise ValueError("each proposal token needs its actual distribution")
        for token, distribution in zip(self.tokens, self.probabilities):
            token_tuple(distribution)
            if (
                not distribution
                or any(not math.isfinite(p) or p <= 0 for p in distribution.values())
                or not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-7)
                or token not in distribution
            ):
                raise ValueError(
                    "proposal probabilities must define normalized support"
                )
