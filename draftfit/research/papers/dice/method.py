"""Replay DICE token scores for one prefix and one supplied position draw."""

import math
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from draftfit.config.schema import StrictConfigModel

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=1)]


class DiceRankingConfig(StrictConfigModel):
    method: Literal["dice_ranking"]
    draft_logits: Vector
    position_gumbels: Vector
    alpha: Finite = Field(gt=0)
    sigma: Finite = Field(gt=0)
    temperature: Finite = Field(ge=0)
    top_k: int = Field(strict=True, gt=0)
    output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_vocabulary(self):
        if len(self.position_gumbels) != len(self.draft_logits):
            raise ValueError("supply one position draw per vocabulary token")
        if self.top_k > len(self.draft_logits):
            raise ValueError("top_k exceeds vocabulary size")
        return self

    def execute(self):
        # Keep the supplied binary floats exact until the common offset is gone.
        # Otherwise a large shared logit can round away the entire position draw.
        alpha, temperature, sigma = map(
            Fraction, (self.alpha, self.temperature, self.sigma)
        )
        scores = [
            alpha * Fraction(logit) + temperature * Fraction(noise)
            for logit, noise in zip(self.draft_logits, self.position_gumbels)
        ]
        peak = max(scores)
        gaps = ((peak - score) / sigma for score in scores)
        # exp(-746) already rounds to binary64 zero; larger gaps need no conversion.
        weights = [0.0 if gap >= 746 else math.exp(-float(gap)) for gap in gaps]
        total = math.fsum(weights)
        # Rank before exponentiation so underflow cannot introduce a false tie.
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return {
            "method": self.method,
            "candidate_ids": ranked[: self.top_k],
            "token_probabilities": [w / total for w in weights],
        }
