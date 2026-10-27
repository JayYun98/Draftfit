"""Strict recipes for the four selectively integrated native research methods."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter, model_validator

from draftfit.config.schema import StrictConfigModel


class Target(StrictConfigModel):
    target_model_path: str = Field(min_length=1)
    target_revision: str | None = None
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    trust_remote_code: bool = False


class TargetDraft(Target):
    draft_checkpoint_path: str = Field(min_length=1)


class InputData(StrictConfigModel):
    input_path: str
    max_samples: int = Field(default=32, gt=0)


class Runtime(StrictConfigModel):
    device: str = "cpu"
    seed: int = 42


class Generation(StrictConfigModel):
    max_new_tokens: int = Field(default=128, ge=0)
    stop_token_ids: list[int] = Field(default_factory=list)
    temperature: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class CastOptions(StrictConfigModel):
    budget: int = Field(default=63, gt=0)
    branch_k: int = Field(default=8, gt=0)


class CastConfig(StrictConfigModel):
    method: Literal["cast"]
    model: TargetDraft
    data: InputData
    generation: Generation = Field(default_factory=Generation)
    speculation: CastOptions = Field(default_factory=CastOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class RolloutOptions(StrictConfigModel):
    block_size: int = Field(default=16, ge=4)
    target_layer_ids: list[int]
    anchors_per_sample: int = Field(default=2, gt=0)
    in_rollout_anchors: bool = True
    stop_token_ids: list[int] = Field(default_factory=list)
    max_length: int = Field(default=2048, gt=0)

    @model_validator(mode="after")
    def check(self):
        if not self.target_layer_ids or self.target_layer_ids != sorted(set(self.target_layer_ids)) or min(self.target_layer_ids) < 0:
            raise ValueError("target_layer_ids must be distinct ascending nonnegative indices")
        if self.max_length <= self.block_size:
            raise ValueError("max_length must exceed block_size")
        return self


class ALRConfig(StrictConfigModel):
    method: Literal["alr_ira"]
    model: Target
    data: InputData
    training: RolloutOptions
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class OspreyModel(StrictConfigModel):
    source_checkpoint: str
    num_layers: int = Field(default=2, gt=0)
    target_hidden_size: int = Field(gt=0)
    target_layer_ids: list[int]
    target_tokenizer: str | None = None
    target_revision: str | None = None

    @model_validator(mode="after")
    def check_taps(self):
        taps = self.target_layer_ids
        if len(taps) != 3 or taps != sorted(set(taps)) or min(taps) < 0:
            raise ValueError("Osprey requires three distinct ascending nonnegative target layer taps")
        return self


class OspreyConfig(StrictConfigModel):
    method: Literal["osprey"]
    model: OspreyModel
    output_dir: str


class RiskOptions(StrictConfigModel):
    alpha: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)
    delta: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)
    reference: int = Field(default=0, ge=0)
    lower_is_better: bool = False


class RedlineConfig(StrictConfigModel):
    method: Literal["redline"]
    data: InputData
    calibration: RiskOptions = Field(default_factory=RiskOptions)
    output_dir: str


Recipe = Annotated[Union[CastConfig, ALRConfig, OspreyConfig, RedlineConfig], Field(discriminator="method")]
ADAPTER = TypeAdapter(Recipe)


def load_recipe(path: str, overrides=()):
    with open(path, encoding="utf-8") as handle:
        if Path(path).suffix in {".yml", ".yaml"}:
            import yaml
            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    config = ADAPTER.validate_python(raw)
    raw = config.model_dump()
    for value in overrides:
        if "=" not in value:
            raise ValueError("override must be section.field=value")
        field, text = value.split("=", 1)
        keys = field.split("."); node = raw
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"unknown override {field}")
            node = node[key]
        if keys[-1] not in node:
            raise ValueError(f"unknown override {field}")
        import yaml
        node[keys[-1]] = yaml.safe_load(text)
    return ADAPTER.validate_python(raw)
