"""Run CAST reference decoding on a local target and draft."""

import json


def run_cast(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.training.model_loading import load_draft_config_source

    from .decode import cast_generate

    model_cfg = config.model
    target = (
        AutoModelForCausalLM.from_pretrained(
            model_cfg.target_model_path,
            revision=model_cfg.target_revision,
            trust_remote_code=model_cfg.trust_remote_code,
            dtype=getattr(torch, model_cfg.torch_dtype),
            attn_implementation="eager",
        )
        .to(config.runtime.device)
        .eval()
    )
    if target.config.model_type not in {"llama", "qwen2", "qwen3"} or any(
        kind != "full_attention"
        for kind in getattr(target.config, "layer_types", ["full_attention"])
    ):
        raise ValueError(
            "native CAST currently supports dense full-attention Llama/Qwen text targets"
        )
    draft_cfg = load_draft_config_source(model_cfg.draft_checkpoint_path)
    if draft_cfg.architectures[0] not in {"DFlashDraftModel", "DFlash2DraftModel"}:
        raise ValueError("native CAST requires a standard DFlash/DFlash2 draft")
    if getattr(draft_cfg, "rope_is_neox_style", True) is False:
        raise ValueError("native CAST target/draft RoPE pairing must match")
    draft_cfg._attn_implementation = "sdpa"
    draft = (
        AutoDraftModel.from_pretrained(
            model_cfg.draft_checkpoint_path,
            config=draft_cfg,
            dtype=getattr(torch, model_cfg.torch_dtype),
        )
        .to(config.runtime.device)
        .eval()
    )
    tokenizer = None
    answers = []
    torch.manual_seed(config.runtime.seed)
    branch_filter, calibration = None, None
    if config.valuation is not None:
        from ..valuspec.recipe import prepare_filter

        branch_filter, calibration, samples = prepare_filter(
            config, target, draft.block_size - 1
        )
    else:
        samples = []
        with open(config.data.input_path) as handle:
            for index, line in enumerate(handle):
                if index >= config.data.max_samples:
                    break
                row = json.loads(line)
                if "input_ids" in row:
                    ids = row["input_ids"]
                else:
                    if tokenizer is None:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_cfg.target_model_path,
                            revision=model_cfg.target_revision,
                            trust_remote_code=model_cfg.trust_remote_code,
                        )
                    ids = tokenizer(row["prompt"]).input_ids
                samples.append((row.get("id", index), ids))
    for identifier, tokens in samples:
        ids = torch.tensor([tokens], dtype=torch.long, device=config.runtime.device)
        output, stats = cast_generate(
            draft,
            target,
            ids,
            config.generation.max_new_tokens,
            config.generation.stop_token_ids,
            budget=config.speculation.budget,
            branch_k=config.speculation.branch_k,
            temperature=config.generation.temperature,
            return_stats=True,
            branch_filter=branch_filter,
        )
        generated = output[0, ids.shape[1] :].tolist()
        answers.append({"id": identifier, "token_ids": generated, "stats": stats})
    return {
        "answers": answers,
        "model": model_cfg.model_dump(),
        "valuation": calibration,
    }
