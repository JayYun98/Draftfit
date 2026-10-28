"""Calibrate a frozen conditional scorer on disjoint target-greedy traces."""

import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memory import token_tuple
from .valuspec import ConditionalBranchFilter, conditional_score, lower_quantile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def token_rows(data, tokenizer=None):
    result = []
    with open(data.input_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= data.max_samples:
                break
            row = json.loads(line)
            if "input_ids" in row:
                ids = token_tuple(row["input_ids"])
            elif tokenizer is not None:
                ids = tuple(tokenizer(row["prompt"]).input_ids)
            else:
                raise ValueError("input_ids are required when no tokenizer is loaded")
            if not ids:
                raise ValueError("nonempty prompts are required")
            result.append((row.get("id", index), ids))
    if not result:
        raise ValueError("research input is empty")
    return result


def prepare_filter(config, target, depth):
    options = config.valuation
    model_cfg = config.model
    target_tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.target_model_path,
        revision=model_cfg.target_revision,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    scorer_tokenizer = AutoTokenizer.from_pretrained(
        options.scorer.target_model_path,
        revision=options.scorer.target_revision,
        trust_remote_code=options.scorer.trust_remote_code,
    )
    if (
        target_tokenizer.get_vocab() != scorer_tokenizer.get_vocab()
        or target_tokenizer.all_special_ids != scorer_tokenizer.all_special_ids
    ):
        raise ValueError("ValuSpec requires identical target/scorer token-ID mappings")
    evaluation = token_rows(config.data, target_tokenizer)
    calibration = token_rows(options.calibration, target_tokenizer)
    if {ids for _, ids in evaluation} & {ids for _, ids in calibration}:
        raise ValueError("ValuSpec calibration and evaluation prompts must be disjoint")
    scorer_cfg = options.scorer
    scorer = (
        AutoModelForCausalLM.from_pretrained(
            scorer_cfg.target_model_path,
            revision=scorer_cfg.target_revision,
            trust_remote_code=scorer_cfg.trust_remote_code,
            dtype=getattr(torch, scorer_cfg.torch_dtype),
            attn_implementation="eager",
        )
        .to(config.runtime.device)
        .eval()
        .requires_grad_(False)
    )
    if scorer.config.vocab_size != target.config.vocab_size:
        raise ValueError("ValuSpec scorer and target vocabularies differ")
    scores = []
    with torch.inference_mode():
        for _, ids in calibration:
            if len(ids) + depth + 1 > target.config.max_position_embeddings:
                raise ValueError("ValuSpec calibration exceeds target context capacity")
            prompt = torch.tensor([ids], device=target.device)
            output = target.generate(
                prompt,
                max_new_tokens=depth + 1,
                do_sample=False,
                eos_token_id=config.generation.stop_token_ids or None,
                pad_token_id=target_tokenizer.pad_token_id or 0,
            )[0].tolist()
            # CAST's pending anchor is an already target-generated input.
            prefix, branch = output[: len(ids) + 1], output[len(ids) + 1 :]
            if branch:
                scores.append(conditional_score(scorer, prefix, branch))
    threshold = lower_quantile(scores, options.delta)
    artifact = {
        "method": "valuspec",
        "paper": "2610.31338v1",
        "implementation": "conditional_leaf_branch_reference",
        "threshold": threshold,
        "delta": options.delta,
        "quantile_method": "linear",
        "calibration_count": len(scores),
        "calibration_prompt_digest": digest([ids for _, ids in calibration]),
        "evaluation_prompt_digest": digest([ids for _, ids in evaluation]),
        "tokenizer_vocab_digest": digest(target_tokenizer.get_vocab()),
        "target": model_cfg.model_dump(),
        "scorer": scorer_cfg.model_dump(),
        "target_resolved_revision": getattr(target.config, "_commit_hash", None),
        "scorer_resolved_revision": getattr(scorer.config, "_commit_hash", None),
        "branch_policy": "root_to_leaf_ancestor_union",
        "max_branch_depth": depth,
        "calibration_scores": scores,
    }
    return ConditionalBranchFilter(scorer, threshold), artifact, evaluation
