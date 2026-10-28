"""EAGLE-4 registration; reuse the EAGLE feature and recurrent-step contracts."""

from dataclasses import replace

from draftfit.algorithms.common.providers import make_registration
from draftfit.algorithms.eagle3 import providers as eagle3

ARCHITECTURE = "LlamaForCausalLMEagle4"


def populate_defaults(payload, target_config, config):
    depth = target_config.num_hidden_layers
    payload.update(
        draft_vocab_size=payload["vocab_size"],
        fc_norm=True,
        norm_output=True,
        sliding_window=4096,
        use_sliding_window=True,
        efficient_lm_head_config={
            "type": "low_rank",
            "rank": min(128, payload["hidden_size"], payload["vocab_size"]),
            "num_candidates": min(128, payload["vocab_size"]),
        },
        eagle_config={"eagle_aux_hidden_state_layer_ids": (
            config.model.aux_hidden_state_layer_ids or [1, depth // 2 - 1, depth - 4]
        )},
    )


def build_draft(config, draft_config):
    if config.training.eagle4_stage == "candidates" and not (
        config.model.draft_checkpoint_path or config.training.resume_from
    ):
        raise ValueError("EAGLE4 candidate training requires a trained dense draft checkpoint")
    if config.model.vocab_mapping_path:
        raise ValueError("EAGLE4 does not support vocabulary pruning")
    draft = eagle3.build_draft(config, draft_config)
    if config.training.eagle4_initialize_candidates:
        if config.training.eagle4_stage != "candidates" or config.training.resume_from:
            raise ValueError("EAGLE4 candidate initialization requires candidate-stage warm start, not full resume")
        draft.initialize_candidate_head(randomized=True)
    draft.set_training_stage(config.training.eagle4_stage)
    return draft


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from draftfit.algorithms.eagle4.model import Eagle4TrainingModel
    from draftfit.algorithms.model_providers import AlgorithmModelParts, _device, _torch_dtype
    from draftfit.modeling.target.target_head import TargetHead

    model = Eagle4TrainingModel(
        draft_model=draft_model,
        length=config.training.ttt_length,
        attention_backend=config.training.attention_backend,
        lk_loss_type=config.training.lk_loss_type,
        kl_scale=config.training.kl_scale,
        kl_decay=config.training.kl_decay,
    ).to(device=_device(), dtype=_torch_dtype(config))
    needs_head = config.mode == "offline" or (
        config.deployment.mode == "disaggregated" and config.training.role == "consumer"
    )
    head = TargetHead.from_pretrained(
        config.model.target_model_path,
        lm_head_key=config.model.lm_head_key,
        revision=config.model.target_revision,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    ) if needs_head else None
    return AlgorithmModelParts(model=model, target_head=head)


def build_step(wrapped_model, *, target_head=None, **options):
    from draftfit.training.strategies.base import Eagle3TrainStrategy

    class Eagle4TrainStrategy(Eagle3TrainStrategy):
        name = "eagle4"

    return Eagle4TrainStrategy(wrapped_model, target_head=target_head, **options)


def resume_contract(config, draft_model, training_model):
    return {
        **{key.replace("eagle3_", "eagle4_", 1): value for key, value in
           eagle3.resume_contract(config, draft_model, training_model).items()},
        "eagle4_stage": config.training.eagle4_stage,
        "eagle4_sliding_window": draft_model.sliding_window,
        "eagle4_candidate_head": dict(draft_model.config.efficient_lm_head_config),
        "eagle4_norm_output": draft_model.norm_output,
    }


def build_offline_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(hidden_states_path, run_id=run_id, strategy="eagle4",
                                 ttt_length=ttt_length, max_len=max_len,
                                 target_repr="hidden_state")


def resolve_capture_layers(config, draft_config, target_config):
    layers = eagle3.resolve_capture_layers(config, draft_config, target_config)
    if len(layers) != 3 or len(set(layers)) != 3:
        raise ValueError("EAGLE4 requires three distinct target feature layers")
    return layers


def create_registration():
    base = eagle3.create_registration()
    spec = replace(
        base.spec,
        name="eagle4",
        draft=replace(base.spec.draft, compatible_architectures={ARCHITECTURE},
                      default_architecture=ARCHITECTURE),
        capabilities=replace(base.spec.capabilities, attention_backends={"sdpa"},
                             supports_vocab_mapping=False),
    )
    providers = replace(
        base.providers,
        algorithm_name="eagle4",
        step=replace(base.providers.step, build=build_step, resume_contract=resume_contract),
        model=replace(
            base.providers.model,
            draft_config=replace(
                base.providers.model.draft_config,
                architecture=ARCHITECTURE,
                compatible_architectures=frozenset({ARCHITECTURE}),
                target_defaults=replace(base.providers.model.draft_config.target_defaults,
                                        draft_vocab_size=None, populate=populate_defaults),
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
        ),
        vocab_mapping_modes=frozenset(),
        offline=tuple(replace(provider, build_reader=build_offline_reader)
                      for provider in base.providers.offline),
    )
    return make_registration(spec, providers)
