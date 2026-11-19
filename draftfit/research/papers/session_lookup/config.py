"""Recipe for one session-local token proposal."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from draftfit.config.schema import StrictConfigModel

Token = Annotated[int, Field(strict=True, ge=0)]
Count = Annotated[int, Field(strict=True, gt=0)]


class SessionLookupConfig(StrictConfigModel):
    method: Literal["session_lookup"]
    context_ids: list[Token] = Field(strict=True)
    k: Count
    min_ngram: Count = 3
    max_ngram: Count = 3
    match: Literal["earliest", "recent"] = "recent"
    output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_range(self):
        if self.min_ngram > self.max_ngram:
            raise ValueError("min_ngram cannot exceed max_ngram")
        return self

    def execute(self):
        from .method import propose

        return {
            "method": self.method,
            "proposed_ids": propose(
                self.context_ids,
                k=self.k,
                min_ngram=self.min_ngram,
                max_ngram=self.max_ngram,
                match=self.match,
            ),
        }
