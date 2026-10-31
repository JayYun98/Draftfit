"""Run bounded model-free memory recipes against a real frozen target."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .memory import NgramMemory, Proposal, SpecMemMemory, token_tuple


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _rows(config):
    with open(config.data.input_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= config.data.max_samples:
                break
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("input_ids"):
                raise ValueError(f"memory row {index} needs nonempty input_ids")
            token_tuple(row["input_ids"])
            yield index, row


def run_memory(config):
    import torch
    from transformers import AutoModelForCausalLM

    from .diagnostics import profile_generate

    model = config.model
    target = (
        AutoModelForCausalLM.from_pretrained(
            model.target_model_path,
            revision=model.target_revision,
            trust_remote_code=model.trust_remote_code,
            dtype=getattr(torch, model.torch_dtype),
            attn_implementation="eager",
        )
        .to(config.runtime.device)
        .eval()
    )
    target.requires_grad_(False)
    if target.config.model_type not in {"llama", "qwen2", "qwen3"} or any(
        kind != "full_attention"
        for kind in getattr(target.config, "layer_types", ["full_attention"])
    ):
        raise ValueError(
            "memory reference decoding currently supports dense full-attention Llama/Qwen text targets"
        )
    identity = {
        "model": model.model_dump(),
        "config_sha256": _digest(target.config.to_dict()),
    }
    tokenizer_file = Path(model.target_model_path) / "tokenizer.json"
    if tokenizer_file.is_file():
        identity["tokenizer_sha256"] = hashlib.sha256(
            tokenizer_file.read_bytes()
        ).hexdigest()
    settings = config.memory.model_dump(exclude={"state_path"})
    if config.method == "ngramrl":
        memory = NgramMemory(
            **config.memory.model_dump(exclude={"draft_tokens", "state_path"})
        )
    else:
        memory = SpecMemMemory(
            capacity=config.memory.capacity,
            max_users=config.memory.max_users,
            threshold=config.memory.threshold,
            max_tokens_per_entry=config.memory.max_tokens_per_entry,
            max_embedding_dimensions=config.memory.max_embedding_dimensions,
        )
    if config.memory.state_path:
        with open(config.memory.state_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if (
            previous.get("schema") != "draftfit-memory-v1"
            or previous.get("method") != config.method
            or previous.get("identity") != identity
            or previous.get("settings") != settings
        ):
            raise ValueError(
                "memory state must match method, model/tokenizer identity and policy settings"
            )
        memory.load_state_dict(previous["state"])
    elif config.method == "specmem" and config.memory.shared_seed_path:
        with open(config.memory.shared_seed_path, encoding="utf-8") as handle:
            seeds = json.load(handle)
        if not isinstance(seeds, list):
            raise ValueError(
                "shared seed must be an explicit list of population entries"
            )
        for entry in seeds:
            memory.add_shared_seed(entry["query_embedding"], entry["token_ids"])
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    answers = []
    for index, row in _rows(config):
        ids = token_tuple(row["input_ids"])
        source = None
        if config.method == "ngramrl":
            step = row.get("step")
            if (
                len(ids) + config.generation.max_new_tokens
                > config.memory.max_tokens_per_rollout
            ):
                raise ValueError(
                    "prompt plus output cap exceeds max_tokens_per_rollout"
                )
            prompt_key = _digest({"scope": row.get("prompt_id", ""), "tokens": ids})
            snapshot = memory.snapshot(prompt_key, step=step)

            def propose(context, remaining, rng):
                return snapshot.propose(
                    context,
                    min(remaining, config.memory.draft_tokens),
                    sampling=config.generation.temperature > 0,
                    generator=rng,
                )

        else:
            user, embedding = row.get("user_id"), row.get("query_embedding")
            if config.generation.max_new_tokens > config.memory.max_tokens_per_entry:
                raise ValueError("output cap exceeds SpecMem max_tokens_per_entry")
            if not isinstance(embedding, list):
                raise ValueError("SpecMem rows require query_embedding vectors")
            candidate, source = memory.retrieve(
                user, embedding, schema_tokens=row.get("schema_draft_tokens", ())
            )

            def propose(context, remaining, rng):
                committed = context[len(ids) :]
                if candidate[: len(committed)] != committed:
                    return Proposal()
                tokens = candidate[
                    len(committed) : len(committed)
                    + min(remaining, config.memory.draft_tokens)
                ]
                return Proposal(tokens, tuple({token: 1.0} for token in tokens))

        output, stats, _ = profile_generate(
            target,
            torch.tensor([ids], device=config.runtime.device),
            lambda: propose,
            config.generation,
            config.diagnostics,
            generator,
        )
        generated = output[0, len(ids) :].tolist()
        writeback = None
        if config.method == "ngramrl":
            memory.add_verified(prompt_key, ids + tuple(generated), step=step)
        else:
            complete = bool(
                generated and generated[-1] in config.generation.stop_token_ids
            )
            store = complete or config.memory.writeback_mode == "verified_sequence"
            if store:
                memory.add_verified(
                    user, _digest(row.get("query", embedding)), embedding, generated
                )
            writeback = {
                "stored": store,
                "eos_terminated": complete,
                "mode": config.memory.writeback_mode,
            }
        answers.append(
            {
                "id": row.get("id", index),
                "token_ids": generated,
                "retrieval": source,
                "writeback": writeback,
                "stats": stats,
            }
        )
    if not answers:
        raise ValueError("memory input must contain at least one row")
    state = {
        "schema": "draftfit-memory-v1",
        "method": config.method,
        "identity": identity,
        "settings": settings,
        "state": memory.state_dict(),
    }
    return {
        "answers": answers,
        "identity": identity,
        "tool_execution_performed": False,
        "implementation": "full_prefix_reference",
        "memory_state": state,
    }
