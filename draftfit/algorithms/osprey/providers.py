"""Osprey uses EAGLE3 feature/step contracts while preserving its own embedding."""
from dataclasses import replace
from draftfit.algorithms.eagle3.providers import create_registration as eagle3
from draftfit.algorithms.common.defaults import no_missing_checkpoint_keys
from draftfit.algorithms.common.providers import DraftConfigProvider, make_registration
from draftfit.algorithms.contracts import DraftRequirement


def build_reader(path, *, run_id, ttt_length, max_len):
    from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader
    return OfflineManifestReader(path, run_id=run_id, strategy="osprey",
        ttt_length=ttt_length, max_len=max_len, target_repr="hidden_state")


def build_draft(config, draft_config):
    if config.model.load_target_embedding:
        raise ValueError("Osprey requires model.load_target_embedding=false to preserve its aligned embedding")
    if not (config.model.draft_checkpoint_path or config.training.resume_from):
        raise ValueError("Osprey requires a converted draft checkpoint or a full resume")
    if config.model.vocab_mapping_path:
        raise ValueError("Osprey preserves its converted full vocabulary; align the tokenizer before training")
    from draftfit.algorithms.model_providers import build_eagle3_draft
    return build_eagle3_draft(config, draft_config)


def build_step(wrapped_model, *, target_head=None, **options):
    from draftfit.training.strategies.base import Eagle3TrainStrategy
    class OspreyTrainStrategy(Eagle3TrainStrategy):
        name = "osprey"

        def checkpoint_state_filter(self, state_dict):
            return {key.replace("draft_model.", ""): value
                    for key, value in state_dict.items() if key.startswith("draft_model.")}
    return OspreyTrainStrategy(wrapped_model, target_head=target_head, **options)


def resume_contract(config, draft_model, training_model):
    from draftfit.algorithms.eagle3.providers import resume_contract as eagle_contract
    return {key.replace("eagle3_", "osprey_"): value
            for key, value in eagle_contract(config, draft_model, training_model).items()}


def create_registration():
    base = eagle3()
    spec = replace(base.spec, name="osprey", draft=DraftRequirement(
        compatible_architectures={"OspreyQwen3DraftModel", "OspreyLlamaDraftModel"},
        default_architecture="OspreyQwen3DraftModel",
        supported_overrides={"attention_layout"},
    ), capabilities=replace(base.spec.capabilities, attention_backends={"sdpa", "flex_attention", "fa"},
                            supports_vocab_mapping=False))
    providers = replace(base.providers, algorithm_name="osprey", vocab_mapping_modes=frozenset(),
        offline=tuple(replace(provider, build_reader=build_reader) for provider in base.providers.offline),
        step=replace(base.providers.step, build=build_step,
                     resume_contract=resume_contract,
                     allowed_missing_checkpoint_keys=no_missing_checkpoint_keys),
        model=replace(base.providers.model, build_draft=build_draft,
            draft_config=DraftConfigProvider(architecture="OspreyQwen3DraftModel",
                compatible_architectures=frozenset({"OspreyQwen3DraftModel", "OspreyLlamaDraftModel"})),
            allow_missing_warm_start_embedding=False))
    return make_registration(spec, providers)
