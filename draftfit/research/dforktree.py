"""CPU probability reference for DForkTree v1 equations 2 and 3.

Evidence is supplied by the caller. This module does not estimate n-grams,
construct a tree, train a model, or execute the author's GPU implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

__all__ = ["CandidateScores", "score_candidates"]


@dataclass(frozen=True)
class CandidateScores:
    """Corrected probabilities in the caller's candidate order, plus the tail."""

    candidate_ids: tuple[int, ...]
    candidate_probs: tuple[float, ...]
    tail_probability: float
    clipped_pmi: tuple[float, ...]
    weight: float


def _sequence(values, name):
    if isinstance(
        values, (str, bytes, bytearray, memoryview, Mapping)
    ) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a nonempty sequence")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must be a nonempty sequence")
    return result


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite Python int or float")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite in binary64") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite in binary64")
    return result


def _probabilities(values, name, size, *, positive):
    result = tuple(_number(value, name) for value in _sequence(values, name))
    if len(result) != size:
        raise ValueError("candidate IDs and probability vectors must be aligned")
    if any(not 0 <= value <= 1 or (positive and value == 0) for value in result):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} values must lie in {interval}")
    if math.fsum(result) > 1:
        raise ValueError(f"{name} subset mass must not exceed one")
    return result


def _log_ratio(numerator, denominator):
    if numerator * 2 >= denominator and numerator <= denominator * 2:
        return math.log1p((numerator - denominator) / denominator)
    numerator_mantissa, numerator_exponent = math.frexp(numerator)
    denominator_mantissa, denominator_exponent = math.frexp(denominator)
    if numerator_mantissa < denominator_mantissa:
        numerator_mantissa *= 2
        numerator_exponent -= 1
    return math.log(numerator_mantissa / denominator_mantissa) + (
        numerator_exponent - denominator_exponent
    ) * math.log(2)


def score_candidates(
    candidate_ids: Sequence[int],
    *,
    marginals: Sequence[int | float],
    conditional: Sequence[int | float],
    unigram: Sequence[int | float],
    position: int,
    parent_path: Literal["spine", "forked"],
    spine_weight: int | float = 0.3,
    fork_weight: int | float = 0.6,
    pmi_clip: int | float = 10.0,
) -> CandidateScores:
    """Score aligned subset probabilities, retaining the unlisted tail class.

    Position is one-based. Position one uses zero correction; later positions
    select the weight from the parent's path classification. All inputs are
    validated even for neutral corrections and unsupported candidates.
    Each probability vector has binary64 ``math.fsum`` at most one.
    Conditional and unigram evidence must be positive for every candidate.
    """
    ids = _sequence(candidate_ids, "candidate_ids")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("candidate IDs must be nonnegative Python integers")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    q = _probabilities(marginals, "marginals", len(ids), positive=False)
    branch = _probabilities(conditional, "conditional", len(ids), positive=True)
    base = _probabilities(unigram, "unigram", len(ids), positive=True)
    if type(position) is not int or position < 1:
        raise ValueError("position must be a positive Python integer")
    if type(parent_path) is not str or parent_path not in ("spine", "forked"):
        raise ValueError("parent_path must be spine or forked")
    spine = _number(spine_weight, "spine_weight")
    fork = _number(fork_weight, "fork_weight")
    clip = _number(pmi_clip, "pmi_clip")
    if min(spine, fork, clip) < 0:
        raise ValueError("weights and pmi_clip must be nonnegative")
    weight = 0.0 if position == 1 else (spine if parent_path == "spine" else fork)
    pmi = tuple(max(-clip, min(clip, _log_ratio(c, u))) for c, u in zip(branch, base))
    rho = 1.0 - math.fsum(q)
    if weight == 0 or not any(pmi):
        return CandidateScores(ids, q, rho, pmi, weight)
    masses, evidence = q + (rho,), pmi + (0.0,)
    center = max(t for mass, t in zip(masses, evidence) if mass > 0)
    logits = tuple(
        math.log(mass) + weight * (t - center) if mass > 0 else -math.inf
        for mass, t in zip(masses, evidence)
    )
    peak = max(logits)
    scaled = tuple(math.exp(logit - peak) for logit in logits)
    total = math.fsum(scaled)
    corrected = tuple(mass / total for mass in scaled)
    return CandidateScores(ids, corrected[:-1], corrected[-1], pmi, weight)


def run_dforktree_scoring(config):
    from .catalog import DFORKTREE_SOURCE

    fields = {
        "id",
        "candidate_ids",
        "marginals",
        "conditional",
        "unigram",
        "position",
        "parent_path",
    }
    digest = hashlib.sha256()
    answers = []
    with open(config.data.input_path, "rb") as stream:
        line_number = 0
        while len(answers) < config.data.max_samples:
            line = stream.readline()
            if not line:
                break
            line_number += 1
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if type(row) is not dict or set(row) != fields:
                    raise ValueError("row must contain exactly the scoring fields")
                if type(row["id"]) is not str or not row["id"]:
                    raise ValueError("row id must be a nonempty string")
                scores = score_candidates(
                    row["candidate_ids"],
                    marginals=row["marginals"],
                    conditional=row["conditional"],
                    unigram=row["unigram"],
                    position=row["position"],
                    parent_path=row["parent_path"],
                    **config.scoring.model_dump(),
                )
            except ValueError as error:
                raise ValueError(f"input line {line_number}: {error}") from error
            answers.append(
                {
                    "id": row["id"],
                    "position": row["position"],
                    "parent_path": row["parent_path"],
                    "candidate_ids": list(scores.candidate_ids),
                    "candidate_probs": list(scores.candidate_probs),
                    "tail_probability": scores.tail_probability,
                    "clipped_pmi": list(scores.clipped_pmi),
                    "weight": scores.weight,
                }
            )
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if not answers:
        raise ValueError("DForkTree scoring input has no selected nonblank rows")
    return {
        "answers": answers,
        "implementation_scope": "cpu_probability_scoring_reference",
        "input_kind": config.input_kind,
        "processed_count": len(answers),
        "input_sha256": digest.hexdigest(),
        "input_digest_scope": "full_file",
        "source": deepcopy(DFORKTREE_SOURCE),
        "arithmetic": "python_binary64",
        "tail_convention": "true_zero_no_floor",
        "author_gpu_numerical_parity": "not_validated",
        "ngram_counting": "not_implemented",
        "corpus_smoothing": "not_implemented",
        "tree_construction": "not_implemented",
        "verifier": "not_implemented",
        "serving_integration": "not_implemented",
        "training": "not_implemented",
        "production_engine_validated": False,
        "throughput_claim": False,
    }
