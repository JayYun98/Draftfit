# SPDX-License-Identifier: MIT
# Adapted from js-lee-AI/Redline d78ede9. Copyright (c) 2026 Jungseob Lee.
"""Select a configuration using paired reference-relative correctness risk.

A violation is a prompt the reference gets right and the candidate gets wrong;
risk divides violations by all calibration prompts, not only reference successes.
The binomial test and Holm family include every supplied configuration, including
the reference. Guarantees assume a fixed configuration family and independent,
representative calibration prompts; this numerical kernel cannot verify either.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _log_pvalues(k, n, alpha):
    """Log P[Binomial(n, alpha) <= k], without probability underflow."""
    values = []
    for count in k:
        if count == n:
            values.append(0.0)
            continue
        terms = [
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(alpha)
            + (n - i) * math.log1p(-alpha)
            for i in range(int(count) + 1)
        ]
        largest = max(terms)
        values.append(
            min(
                0.0, largest + math.log(math.fsum(math.exp(x - largest) for x in terms))
            )
        )
    return np.asarray(values)


def _holm(log_p, delta):
    valid = np.zeros(len(log_p), dtype=bool)
    for rank, index in enumerate(np.argsort(log_p, kind="stable")):
        if log_p[index] > math.log(delta) - math.log(len(log_p) - rank):
            break
        valid[index] = True
    return valid


@dataclass
class Selection:
    """Selection arrays follow input order; displayed pvalues may underflow to 0."""

    names: list[str]
    cost: np.ndarray
    accuracy: np.ndarray
    violations: np.ndarray
    n: int
    pvalues: np.ndarray
    valid: np.ndarray
    deployed: int
    reference: int
    alpha: float
    delta: float
    lower_is_better: bool = False

    @property
    def risk(self):
        return self.violations / self.n

    @property
    def deployed_name(self):
        return self.names[self.deployed]


def select(
    correct, cost, names=None, alpha=0.1, delta=0.1, reference=0, lower_is_better=False
):
    """Return the best-cost Holm-valid configuration, or the reference fallback.

    ``correct`` is a nonempty binary [configuration, prompt] matrix of paired
    outcomes. Costs are positive and higher is better unless lower_is_better.
    Input order breaks cost ties. Fallback does not make the reference valid.
    Decisions use log probabilities even when displayed pvalues round to zero.
    """
    raw = np.asarray(correct)
    if raw.ndim != 2 or not all(raw.shape) or not np.isin(raw, [0, 1]).all():
        raise ValueError(
            "correct must be a nonempty binary [configuration, prompt] matrix"
        )
    for name, value in (("alpha", alpha), ("delta", delta)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must be within (0,1)")
    if (
        isinstance(reference, (bool, np.bool_))
        or not isinstance(reference, Integral)
        or not 0 <= reference < raw.shape[0]
    ):
        raise ValueError("reference configuration index is out of range")
    if not isinstance(lower_is_better, (bool, np.bool_)):
        raise ValueError("lower_is_better must be a boolean")
    correct = raw.astype(bool)
    cost = np.array(cost, dtype=float, copy=True)
    m, n = correct.shape
    if cost.shape != (m,):
        raise ValueError(
            f"cost needs one value per configuration, got shape {cost.shape} for m={m}"
        )
    if not np.isfinite(cost).all() or not (cost > 0).all():
        raise ValueError("configuration costs must be finite and positive")
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be distinct and match the configuration count")
    names = [str(i) for i in range(m)] if names is None else [str(s) for s in names]
    if (
        len(names) != m
        or len(set(names)) != m
        or any(not name.strip() for name in names)
    ):
        raise ValueError(
            "names must be nonempty, distinct and match the configuration count"
        )
    reference = int(reference)
    k = (correct[reference][None, :] & ~correct).sum(1)
    log_p = _log_pvalues(k, n, float(alpha))
    valid = _holm(log_p, float(delta))
    best = None
    for index in np.flatnonzero(valid):
        if best is None or (
            cost[index] < cost[best] if lower_is_better else cost[index] > cost[best]
        ):
            best = index
    deployed = reference if best is None else int(best)
    return Selection(
        names,
        cost,
        correct.mean(1),
        k,
        n,
        np.exp(log_p),
        valid,
        deployed,
        reference,
        float(alpha),
        float(delta),
        bool(lower_is_better),
    )
