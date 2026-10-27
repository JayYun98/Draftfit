"""ALR/IRA registration reuses the platform trainer, model, and feature reader."""
from dataclasses import replace
from functools import partial
from draftfit.algorithms.dflash.providers import create_registration as dflash
from draftfit.algorithms.common.dflash_family_data import _normalize_hidden_states, build_dspark_collator
from draftfit.algorithms.common.providers import make_registration

KEYS = ("input_ids", "loss_mask", "hidden_states", "target_last_hidden_states", "alr_anchor")
NORMALIZER = "alr_ira_offline_v1"



def build_step(model, **_options):
    from .step import ALRTrainStrategy
    return ALRTrainStrategy(model)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    if config.training.loss_type != "dflash":
        raise ValueError("ALR/IRA uses sparse target KD; training.loss_type must be dflash")
    validate_prepared_cache(config, draft_model)
    from draftfit.algorithms.model_providers import _build_dflash_family_model
    from .model import ALRTrainingModel
    return _build_dflash_family_model(config, draft_model, tokenizer, lambda common: ALRTrainingModel(
        **common, topk=config.training.alr_topk,
        hard_label_mix=config.training.alr_hard_label_mix, tail_bucket=config.training.alr_tail_bucket,
    ))


def validate_prepared_cache(config, draft_model):
    from pathlib import Path
    from draftfit.inference.capture_manifest import load_capture_manifest
    for path in (config.data.hidden_states_path, config.data.eval_hidden_states_path):
        if not path:
            continue
        manifest = load_capture_manifest(str(Path(path) / "capture_manifest.json"))
        contract = manifest["contract"]
        settings = manifest.get("producer", {}).get("settings", {})
        if contract["strategy"] != "alr_ira" or settings.get("label_source") != "target_rollout":
            raise ValueError("ALR training requires a prepared target-rollout cache")
        if contract["capture_layers"] != list(draft_model.target_layer_ids):
            raise ValueError("ALR cache target taps differ from the draft config")
        if settings.get("block_size") != int(draft_model.block_size):
            raise ValueError("ALR cache block size differs from the draft config")
        if manifest["target"]["model"] != config.model.target_model_path or (
            config.model.target_revision and manifest["target"]["revision"] != config.model.target_revision
        ):
            raise ValueError("ALR cache target provenance differs from the run config")


def resume_contract(config, draft_model, model):
    return {"alr_ira_block_size": int(model.block_size), "alr_ira_topk": int(model.topk),
            "alr_ira_hard_label_mix": model.hard_label_mix, "alr_ira_tail_bucket": model.tail_bucket,
            "alr_ira_target_layer_ids": tuple(draft_model.target_layer_ids),
            "alr_ira_loss_decay_gamma": model.loss_decay_gamma, "alr_ira_mask_token_id": int(model.mask_token_id),
            "alr_ira_attention_backend": model.attention_backend}


def build_reader(path, **kwargs):
    from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader
    from pathlib import Path
    from draftfit.inference.capture_manifest import load_capture_manifest

    manifest = load_capture_manifest(str(Path(path) / "capture_manifest.json"))
    if manifest.get("producer", {}).get("settings", {}).get("label_source") != "target_rollout":
        raise ValueError("ALR training requires a prepared target-rollout cache")
    return OfflineManifestReader(path, strategy="alr_ira", feature_keys=KEYS, target_repr=None, **kwargs)


def normalize(raw, max_len):
    result = {"input_ids": raw["input_ids"][:max_len].unsqueeze(0),
              "loss_mask": raw["loss_mask"][:max_len].unsqueeze(0)}
    for key in ("hidden_states", "target_last_hidden_states"):
        result[key] = _normalize_hidden_states(raw, key, max_len, description=key)
    if len({value.shape[1] for value in result.values()}) != 1:
        raise ValueError("ALR features have mismatched sequence lengths")
    anchor = raw["alr_anchor"].reshape(1, 1).long()
    if not 0 <= int(anchor[0, 0]) < result["input_ids"].shape[1] - 1:
        raise ValueError("truncation removed the ALR anchor/rollout")
    if result["loss_mask"][0, int(anchor[0, 0]) + 1:].sum() < 1:
        raise ValueError("ALR sample needs a supervised rollout slot")
    return {**result, "alr_anchor": anchor}


def build_normalizer(max_len, **_options):
    return partial(normalize, max_len=max_len)


def build_collator():
    import torch
    base = build_dspark_collator()
    def collate(features):
        return {**base(features), "alr_anchor": torch.cat([row["alr_anchor"] for row in features])}
    return collate


def valid_loss_mask(mask):
    return bool((mask > 0).any())


def one_loss_token(_config, _draft_config):
    return 1


def create_registration():
    base = dflash()
    contract = base.spec.feature_contracts[0]
    contract = replace(contract, required_tensors=set(KEYS),
                       storage=replace(contract.storage, required_tensors=set(KEYS), normalizer=NORMALIZER))
    spec = replace(base.spec, name="alr_ira", feature_contracts=(contract,),
                   capabilities=replace(base.spec.capabilities, attention_backends={"eager", "sdpa"}))
    offline = replace(base.providers.offline[0], normalizer_id=NORMALIZER,
        capture_layout=None,
        build_reader=build_reader, build_normalizer=build_normalizer, build_collator=build_collator)
    providers = replace(base.providers, algorithm_name="alr_ira", server_streaming=(), offline=(offline,),
        step=replace(base.providers.step, build=build_step, resume_contract=resume_contract),
        model=replace(base.providers.model, build_training_model=build_training_model,
                      minimum_loss_tokens=one_loss_token, loss_mask_filter=valid_loss_mask))
    return make_registration(spec, providers)
