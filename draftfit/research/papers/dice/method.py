"""Replay DICE token scores for one prefix and one supplied position draw."""

import math
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
        scores = [
            self.alpha * logit + self.temperature * noise
            for logit, noise in zip(self.draft_logits, self.position_gumbels)
        ]
        if not all(map(math.isfinite, scores)):
            raise ValueError("weighted scores exceed floating-point range")
        peak = max(scores)

        def centered(value):
            delta = value - peak
            if math.isfinite(delta):
                return delta / self.sigma
            return value / self.sigma - peak / self.sigma

        weights = [math.exp(centered(value)) for value in scores]
        total = math.fsum(weights)
        # Rank before exponentiation so underflow cannot introduce a false tie.
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return {
            "method": self.method,
            "candidate_ids": ranked[: self.top_k],
            "token_probabilities": [w / total for w in weights],
        }
