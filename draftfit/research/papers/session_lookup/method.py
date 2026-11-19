"""Propose a full continuation from already observed session tokens."""

from typing import Literal

from pydantic import NonNegativeInt, PositiveInt, validate_call


@validate_call(config={"strict": True})
def propose(
    context_ids: list[NonNegativeInt],
    *,
    k: PositiveInt,
    min_ngram: PositiveInt,
    max_ngram: PositiveInt,
    match: Literal["earliest", "recent"],
) -> list[int]:
    if min_ngram > max_ngram:
        raise ValueError("min_ngram cannot exceed max_ngram")
    size = len(context_ids)
    # A valid continuation begins strictly before the query suffix and fits k.
    for n in range(min(max_ngram, (size - 1) // 2), min_ngram - 1, -1):
        ends = range(n, min(size - n - 1, size - k) + 1)
        suffix = context_ids[-n:]
        for end in ends if match == "earliest" else reversed(ends):
            if context_ids[end - n : end] == suffix:
                return context_ids[end : end + k]
    return []
