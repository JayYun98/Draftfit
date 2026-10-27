"""Run native research preparation, reference decoding, and calibration."""

import json
import shutil
import tempfile
from pathlib import Path


def plan(config, recipe_path=None):
    """Describe the selected work without loading models or writing output."""
    return {
        "project": config.method,
        "implementation": "native_reference" if config.method == "cast" else "native",
        "config": config.model_dump(),
        "execution": "not-run",
        "serving_compatibility": "not_validated",
    }


def _cast(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.training.model_loading import load_draft_config_source

    from .cast.decode import cast_generate

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
    with open(config.data.input_path) as handle:
        for index, line in enumerate(handle):
            if index >= config.data.max_samples:
                break
            row = json.loads(line)
            if "input_ids" in row:
                ids = torch.tensor(
                    [row["input_ids"]], dtype=torch.long, device=config.runtime.device
                )
            else:
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_cfg.target_model_path,
                        revision=model_cfg.target_revision,
                        trust_remote_code=model_cfg.trust_remote_code,
                    )
                ids = tokenizer(row["prompt"], return_tensors="pt").input_ids.to(
                    config.runtime.device
                )
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
            )
            generated = output[0, ids.shape[1] :].tolist()
            answers.append(
                {"id": row.get("id", index), "token_ids": generated, "stats": stats}
            )
    return {"answers": answers, "model": model_cfg.model_dump()}


def _redline(config):
    from draftfit.benchmarks.redline import select

    with open(config.data.input_path) as handle:
        data = json.load(handle)
    result = select(
        data["correct"],
        data["cost"],
        names=data.get("names"),
        **config.calibration.model_dump(),
    )
    return {
        "deployed": result.deployed_name,
        "reference": result.names[result.reference],
        "valid": result.valid.tolist(),
        "pvalues": result.pvalues.tolist(),
        "risk": result.risk.tolist(),
        "n": result.n,
        "alpha": result.alpha,
        "delta": result.delta,
        "cost": result.cost.tolist(),
        "names": result.names,
    }


def run(config, recipe_path=None):
    if config.method == "alr_ira":
        from draftfit.algorithms.alr_ira.prepare import prepare

        return prepare(config)
    if config.method == "osprey":
        from .osprey import convert

        m = config.model
        destination = Path(config.output_dir)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=".osprey-", dir=destination.parent))
        try:
            converted = scratch / "draft"
            result = convert(
                m.source_checkpoint,
                str(converted),
                m.num_layers,
                m.target_hidden_size,
                m.target_layer_ids,
            )
            if m.target_tokenizer:
                from .tokenizer_alignment import align

                result["alignment"] = align(
                    converted,
                    m.target_tokenizer,
                    converted,
                    target_revision=m.target_revision,
                )
            (converted / "adaptation.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            converted.rename(destination)
            return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    destination = Path(config.output_dir)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if config.method not in {"cast", "redline"}:
        raise ValueError(f"unsupported research method: {config.method}")
    result = _cast(config) if config.method == "cast" else _redline(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix=".draftfit-research-", dir=destination.parent)
    )
    try:
        (scratch / "recipe.json").write_text(config.model_dump_json(indent=2) + "\n")
        (scratch / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return result
