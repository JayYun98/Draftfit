"""CPU best-mode prefix selection from supplied full-vocabulary probabilities.

Independent implementation of MFD v1 Section 4.3. The reviewed author reference
is inference/hf/ddtree.py::build_mixture_tree at the revision in catalog.py.
This reference uses binary64 log scores; near ties depend on that arithmetic.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real

MAX_MODES = 8
MAX_DEPTH = 16
MAX_VOCAB = 4096
MAX_CELLS = 262144
MAX_BUDGET = 256
MAX_SAMPLES = 32
MAX_LINE_BYTES = 8 * 1024 * 1024
NORMALIZATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class PrefixNode:
    node_id: int
    parent: int
    prefix: tuple[int, ...]
    log_score: float


@dataclass(frozen=True)
class PrefixTree:
    """Ordered non-root nodes; root 0 and its empty prefix are implicit."""

    nodes: tuple[PrefixNode, ...]


@dataclass(order=True, frozen=True)
class _Candidate:
    key: tuple[float, int, tuple[int, ...], int]
    parent: int = field(compare=False)
    rank: int = field(compare=False)
    parent_log_terms: tuple[float, ...] = field(compare=False)
    choices: tuple[tuple[float, int, float], ...] = field(compare=False)


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _distribution(values, name):
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{name} must contain finite numeric probabilities in [0, 1]"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"{name} must contain finite numeric probabilities in [0, 1]"
            )
        result.append(number)
    if abs(math.fsum(result) - 1.0) > NORMALIZATION_TOLERANCE:
        raise ValueError(
            f"{name} must sum to 1 within 1e-8; no renormalization is applied"
        )
    return tuple(result)


def _validate(probabilities, priors, budget):
    if type(budget) is not int or not 0 <= budget <= MAX_BUDGET:
        raise ValueError("budget must be an integer in [0, 256]")
    modes = _sequence(probabilities, "probabilities")
    if not 1 <= len(modes) <= MAX_MODES:
        raise ValueError("probabilities must have 1..8 modes")
    first = _sequence(modes[0], "probabilities[0]")
    if not 1 <= len(first) <= MAX_DEPTH:
        raise ValueError("probabilities must have depth in [1, 16]")
    first_row = _sequence(first[0], "probabilities[0][0]")
    if not 1 <= len(first_row) <= MAX_VOCAB:
        raise ValueError("probabilities must have vocabulary size in [1, 4096]")
    depth, vocab = len(first), len(first_row)
    if len(modes) * depth * vocab > MAX_CELLS:
        raise ValueError("probabilities may contain at most 262144 cells")
    validated = []
    for k, mode in enumerate(modes):
        if len(_sequence(mode, f"probabilities[{k}]")) != depth:
            raise ValueError("probabilities must have the same depth for every mode")
        rows = []
        for position, row in enumerate(mode):
            name = f"probabilities[{k}][{position}]"
            if len(_sequence(row, name)) != vocab:
                raise ValueError(
                    "probabilities must have the same full vocabulary at every position"
                )
            rows.append(_distribution(row, name))
        validated.append(tuple(rows))
    if priors is None:
        weights = (1.0 / len(modes),) * len(modes)
    else:
        if len(_sequence(priors, "priors")) != len(modes):
            raise ValueError("priors must have one probability per mode")
        weights = _distribution(priors, "priors")
    return tuple(validated), weights


def select_prefixes(
    probabilities: Sequence[Sequence[Sequence[float]]],
    *,
    priors: Sequence[float] | None = None,
    budget: int,
) -> PrefixTree:
    """Select distinct prefixes by max_k(log prior_k + sum log q_k).

    Probabilities are dense [mode][position][token] distributions. They are
    validated even for budget zero and are never renormalized. The budget counts
    non-root nodes. Equal log scores prefer depth, token tuple, then mode.
    """
    modes, weights = _validate(probabilities, priors, budget)
    if budget == 0:
        return PrefixTree(())
    log_rows = tuple(
        tuple(
            tuple(
                (token, math.log(value)) for token, value in enumerate(row) if value > 0
            )
            for row in mode
        )
        for mode in modes
    )
    frontier: list[_Candidate] = []

    def offer(mode, parent, prefix, parent_log_terms, rank=0, choices=None):
        if choices is None:
            choices = tuple(
                heapq.nsmallest(
                    budget,
                    (
                        (
                            -math.fsum((*parent_log_terms, log_probability)),
                            token,
                            log_probability,
                        )
                        for token, log_probability in log_rows[mode][len(prefix)]
                    ),
                )
            )
        negative_score, token, _ = choices[rank]
        child = prefix + (token,)
        heapq.heappush(
            frontier,
            _Candidate(
                (negative_score, len(child), child, mode),
                parent,
                rank,
                parent_log_terms,
                choices,
            ),
        )

    for mode, prior in enumerate(weights):
        if prior > 0:
            offer(mode, 0, (), (math.log(prior),))
    edges: dict[tuple[int, int], int] = {}
    selected: list[PrefixNode] = []
    while frontier and len(selected) < budget:
        entry = heapq.heappop(frontier)
        negative_score, depth, prefix, mode = entry.key
        score = -negative_score
        edge = (entry.parent, prefix[-1])
        node_id = edges.get(edge)
        if node_id is None:
            node_id = len(selected) + 1
            edges[edge] = node_id
            selected.append(PrefixNode(node_id, entry.parent, prefix, score))
        if entry.rank + 1 < len(entry.choices):
            offer(
                mode,
                entry.parent,
                prefix[:-1],
                entry.parent_log_terms,
                entry.rank + 1,
                entry.choices,
            )
        if depth < len(modes[mode]):
            mode_log_terms = (
                *entry.parent_log_terms,
                entry.choices[entry.rank][2],
            )
            offer(mode, node_id, prefix, mode_log_terms)
    return PrefixTree(tuple(selected))


def run_mfd(config):
    """Read bounded JSONL rows and return JSON-safe CPU reference results."""
    from .catalog import MFD_PAPER_REVISION, MFD_SOURCE_REVISION

    answers = []
    if (
        type(config.data.max_samples) is not int
        or not 1 <= config.data.max_samples <= MAX_SAMPLES
    ):
        raise ValueError("max_samples must be an integer in [1, 32]")
    with open(config.data.input_path, "rb") as handle:
        for index in range(config.data.max_samples):
            line = handle.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"MFD input row {index + 1} exceeds 8 MiB")
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ValueError(
                    f"malformed MFD input row {index + 1}: {error}"
                ) from error
            if not isinstance(row, dict) or "probabilities" not in row:
                raise ValueError(
                    f"MFD input row {index + 1} must be an object with probabilities"
                )
            identifier = row.get("id", index)
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                raise ValueError(
                    f"MFD input row {index + 1} id must be a string or integer"
                )
            try:
                tree = select_prefixes(
                    row["probabilities"],
                    priors=row.get("priors"),
                    budget=config.tree.budget,
                )
            except ValueError as error:
                raise ValueError(f"MFD input row {index + 1}: {error}") from error
            answers.append({"id": identifier, **asdict(tree)})
        if handle.read(1):
            raise ValueError(f"MFD input exceeds max_samples={config.data.max_samples}")
    if not answers:
        raise ValueError("MFD input is empty")
    return {
        "answers": answers,
        "implementation": "cpu_reference",
        "source_revision": MFD_SOURCE_REVISION,
        "paper_revision": MFD_PAPER_REVISION,
        "score_semantics": "max_over_weighted_modes",
        "model_execution_performed": False,
        "serving_validation": "not-run",
    }
