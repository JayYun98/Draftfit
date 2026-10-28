"""Strict recipes for selectively integrated native research methods."""

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


class DecodeDiagnostics(StrictConfigModel):
    compare_ar: bool = False
    record_acceptance: bool = False
    position_bin_size: int = Field(default=256, gt=0)
    warmup: int = Field(default=0, ge=0, le=10)
    repetitions: int = Field(default=1, gt=0, le=30)


class ReferenceDecodeOptions(StrictConfigModel):
    generation: Generation = Field(default_factory=Generation)
    diagnostics: DecodeDiagnostics = Field(default_factory=DecodeDiagnostics)

    @model_validator(mode="after")
    def paired_greedy(self):
        if self.diagnostics.compare_ar and self.generation.temperature != 0:
            raise ValueError(
                "paired token-workload AR comparison requires greedy generation"
            )
        return self


class CastOptions(StrictConfigModel):
    budget: int = Field(default=63, gt=0)
    branch_k: int = Field(default=8, gt=0)


class ValuationOptions(StrictConfigModel):
    scorer: Target
    calibration: InputData
    delta: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)


class CastConfig(StrictConfigModel):
    method: Literal["cast"]
    model: TargetDraft
    data: InputData
    generation: Generation = Field(default_factory=Generation)
    speculation: CastOptions = Field(default_factory=CastOptions)
    valuation: ValuationOptions | None = None
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str

    @model_validator(mode="after")
    def greedy_valuation(self):
        if self.valuation is not None and self.generation.temperature != 0:
            raise ValueError("ValuSpec filtering requires greedy generation")
        return self


class RolloutOptions(StrictConfigModel):
    block_size: int = Field(default=16, ge=4)
    target_layer_ids: list[int]
    anchors_per_sample: int = Field(default=2, gt=0)
    in_rollout_anchors: bool = True
    stop_token_ids: list[int] = Field(default_factory=list)
    max_length: int = Field(default=2048, gt=0)

    @model_validator(mode="after")
    def check(self):
        if (
            not self.target_layer_ids
            or self.target_layer_ids != sorted(set(self.target_layer_ids))
            or min(self.target_layer_ids) < 0
        ):
            raise ValueError(
                "target_layer_ids must be distinct ascending nonnegative indices"
            )
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
            raise ValueError(
                "Osprey requires three distinct ascending nonnegative target layer taps"
            )
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


class NgramOptions(StrictConfigModel):
    n_min: int = Field(default=2, ge=1, le=64)
    n_max: int = Field(default=5, ge=1, le=64)
    window_steps: int = Field(default=8, gt=0)
    max_prompts: int = Field(default=128, gt=0)
    max_rollouts_per_prompt: int = Field(default=16, gt=0)
    max_tokens_per_rollout: int = Field(default=4096, gt=0)
    draft_tokens: int = Field(default=8, gt=0)
    state_path: str | None = None

    @model_validator(mode="after")
    def check_orders(self):
        if self.n_min > self.n_max:
            raise ValueError("n_min must not exceed n_max")
        return self


class SpecMemOptions(StrictConfigModel):
    capacity: int = Field(default=48, gt=0)
    max_users: int = Field(default=128, gt=0)
    max_tokens_per_entry: int = Field(default=4096, gt=0)
    max_embedding_dimensions: int = Field(default=4096, gt=0)
    writeback_mode: Literal["eos", "verified_sequence"] = "eos"
    threshold: float = Field(default=0.35, ge=-1, le=1, allow_inf_nan=False)
    draft_tokens: int = Field(default=32, gt=0)
    shared_seed_path: str | None = None
    state_path: str | None = None


class NgramConfig(ReferenceDecodeOptions):
    method: Literal["ngramrl"]
    model: Target
    data: InputData
    memory: NgramOptions = Field(default_factory=NgramOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class SpecMemConfig(ReferenceDecodeOptions):
    method: Literal["specmem"]
    model: Target
    data: InputData
    memory: SpecMemOptions = Field(default_factory=SpecMemOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class ARTargetDraft(Target):
    draft_model_path: str = Field(min_length=1)
    draft_revision: str | None = None


class RIFTOptions(StrictConfigModel):
    context_budget: int = Field(default=128, gt=0)
    recent_fraction: float = Field(default=0.25, gt=0, le=1, allow_inf_nan=False)
    observation_window: int = Field(default=64, gt=0)
    draft_tokens: int = Field(default=8, gt=0)


class RIFTConfig(ReferenceDecodeOptions):
    method: Literal["rift"]
    model: ARTargetDraft
    data: InputData
    cache: RIFTOptions = Field(default_factory=RIFTOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class TreeWYOptions(StrictConfigModel):
    max_nodes: int = Field(default=64, gt=0, le=512)
    use_qk_l2norm: bool = True
    scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class TreeWYConfig(StrictConfigModel):
    method: Literal["treewy"]
    data: InputData
    tree: TreeWYOptions = Field(default_factory=TreeWYOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


class ReplayOptions(StrictConfigModel):
    active_roots: int = Field(default=4, gt=0)
    max_roots: int = Field(default=32, gt=0)
    replay_width: int = Field(default=32, gt=0)
    max_processed_positions: int = Field(default=100000, gt=0)
    root_temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    root_top_p: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    revision_template: str = (
        "Question:\n{question}\nPrevious solution:\n{solution}\nWrite a corrected complete solution:\n"
    )
    answer_extractor: Literal["token_sequence", "last_line"] = "last_line"

    @model_validator(mode="after")
    def stationary(self):
        from string import Formatter

        fields = set()
        for _, field, spec, conversion in Formatter().parse(self.revision_template):
            if field is not None:
                if spec or conversion:
                    raise ValueError(
                        "revision_template cannot use format specs or conversions"
                    )
                fields.add(field)
        if fields != {"question", "solution"}:
            raise ValueError(
                "stationary revision_template must use only {question} and {solution}"
            )
        return self


class ARRConfig(StrictConfigModel):
    method: Literal["arr"]
    model: Target
    data: InputData
    refinement: ReplayOptions = Field(default_factory=ReplayOptions)
    generation: Generation = Field(default_factory=Generation)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str

    @model_validator(mode="after")
    def check_revision(self):
        if self.generation.temperature != 0:
            raise ValueError(
                "ARR revisions require greedy generation; set refinement.root_temperature for sampling"
            )
        if self.generation.max_new_tokens < 1 or not self.generation.stop_token_ids:
            raise ValueError(
                "ARR requires a positive per-call cap and explicit EOS stop_token_ids"
            )
        return self


class MoEPolicyOptions(StrictConfigModel):
    gamma: int = Field(default=8, gt=0)
    max_utility: int = Field(default=4, gt=0)
    smoothing: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    target_layers: int = Field(gt=0)
    expert_bytes: int = Field(gt=0)
    cpu_unit_ms: float = Field(gt=0, allow_inf_nan=False)
    gpu_unit_ms: float = Field(gt=0, allow_inf_nan=False)
    io_unit_ms: float = Field(ge=0, allow_inf_nan=False)
    draft_unit_ms: float = Field(ge=0, allow_inf_nan=False)


class MoEPolicyConfig(StrictConfigModel):
    method: Literal["moe_spac"]
    data: InputData
    policy: MoEPolicyOptions
    output_dir: str


class QueryLayoutOptions(StrictConfigModel):
    phase_mask: int = Field(default=3, ge=0, le=15)
    atol: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    rtol: float = Field(default=1e-4, ge=0, allow_inf_nan=False)


class QueryLayoutConfig(StrictConfigModel):
    method: Literal["query_layout"]
    data: InputData
    layout: QueryLayoutOptions = Field(default_factory=QueryLayoutOptions)
    runtime: Runtime = Field(default_factory=Runtime)
    output_dir: str


Recipe = Annotated[
    Union[
        CastConfig,
        ALRConfig,
        OspreyConfig,
        RedlineConfig,
        NgramConfig,
        SpecMemConfig,
        RIFTConfig,
        TreeWYConfig,
        ARRConfig,
        MoEPolicyConfig,
        QueryLayoutConfig,
    ],
    Field(discriminator="method"),
]
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
        keys = field.split(".")
        node = raw
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"unknown override {field}")
            node = node[key]
        if keys[-1] not in node:
            raise ValueError(f"unknown override {field}")
        import yaml

        node[keys[-1]] = yaml.safe_load(text)
    return ADAPTER.validate_python(raw)
