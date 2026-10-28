"""Budgeted Adaptive Replay Refinement, independently implemented from the paper.

arXiv:2610.28294v1. Uses an explicit stationary revision operator, greedy
one-way replay, root retirement/replacement and one final vote per valid root.
Full-prefix work is charged as processed target positions, not cached FLOPs.
The public ARR code has no verified reuse license; no source/prompt was copied.
"""

from collections import Counter, deque
from dataclasses import dataclass

import torch

from .memory import Proposal
from .reference_decode import generate


@dataclass
class PositionBudget:
    limit: int
    used: int = 0

    def consume(self, positions):
        if self.used + positions > self.limit:
            return False
        self.used += positions
        return True


class ReplayProposer:
    def __init__(self, prefix_length, previous, width):
        self.prefix_length, self.previous, self.width = (
            prefix_length,
            tuple(previous),
            width,
        )
        self.on_draft = True

    def __call__(self, context, remaining, generator):
        generated = context[self.prefix_length :]
        if generated != self.previous[: len(generated)]:
            self.on_draft = False
        if not self.on_draft:
            return Proposal()
        tokens = self.previous[
            len(generated) : len(generated) + min(remaining, self.width - 1)
        ]
        return Proposal(tokens, tuple({t: 1.0} for t in tokens))


def schedule_roots(initial, revise, *, active_roots, max_roots):
    """Serial bounded reference scheduler. Callbacks return tokens/completeness/stats.

    `initial` returns None when worst-case new-root admission is unaffordable;
    `revise` returns None when no revision can run. No pending work is hidden
    from accounting. Incomplete revisions keep their root's last complete answer.
    """
    active, completed, events, next_id = deque(), [], [], 0

    def admit():
        nonlocal next_id
        while len(active) < active_roots and next_id < max_roots:
            result = initial()
            if result is None:
                return
            root = {"root_id": next_id, "tokens": result["tokens"], "revisions": 0}
            next_id += 1
            events.append({"root_id": root["root_id"], "event": "initial", **result})
            if result["complete"]:
                active.append(root)

    admit()
    while active:
        root = active.popleft()
        result = revise(root["tokens"])
        if result is None:
            root["retirement"] = "budget_exhausted"
            for remaining in active:
                remaining["retirement"] = "budget_exhausted"
            completed.append(root)
            completed.extend(active)
            active.clear()
            break
        events.append({"root_id": root["root_id"], "event": "revision", **result})
        root["revisions"] += 1
        if not result["complete"]:
            root["retirement"] = "incomplete_revision_preserved_previous"
            completed.append(root)
        elif result["tokens"] == root["tokens"]:
            root["retirement"] = "fixed_point"
            completed.append(root)
        else:
            root["tokens"] = result["tokens"]
            active.append(root)
        admit()
    return completed, events


def run_arr(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .valuation_runner import token_rows

    cfg, opts, gen = config.model, config.refinement, config.generation
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.target_model_path,
            revision=cfg.target_revision,
            trust_remote_code=cfg.trust_remote_code,
            dtype=getattr(torch, cfg.torch_dtype),
            attn_implementation="eager",
        )
        .to(config.runtime.device)
        .eval()
        .requires_grad_(False)
    )
    if target.config.model_type not in {"llama", "qwen2", "qwen3"} or any(
        kind != "full_attention"
        for kind in getattr(target.config, "layer_types", ["full_attention"])
    ):
        raise ValueError(
            "ARR reference supports dense full-attention Llama/Qwen targets"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.target_model_path,
        revision=cfg.target_revision,
        trust_remote_code=cfg.trust_remote_code,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    answers = []
    for identifier, ids in token_rows(config.data, tokenizer):
        budget = PositionBudget(opts.max_processed_positions)
        question = tokenizer.decode(ids, skip_special_tokens=False)

        def decode(prefix, proposer, temperature=0.0, top_p=1.0):
            output, stats = generate(
                target,
                torch.tensor([prefix], device=target.device),
                proposer,
                max_new_tokens=gen.max_new_tokens,
                stop_token_ids=gen.stop_token_ids,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
                forward_budget=budget,
            )
            tokens = output[0, len(prefix) :].tolist()
            return {
                "tokens": tokens,
                "complete": bool(tokens and tokens[-1] in gen.stop_token_ids),
                "stats": stats,
            }

        def initial():
            cap = gen.max_new_tokens
            reserve = cap * len(ids) + cap * (cap - 1) // 2
            if reserve > budget.limit - budget.used:
                return None
            return decode(
                ids, lambda *_: Proposal(), opts.root_temperature, opts.root_top_p
            )

        def revise(previous):
            text = opts.revision_template.format(
                question=question,
                solution=tokenizer.decode(previous, skip_special_tokens=False),
            )
            prefix = tokenizer(text).input_ids
            if (
                not prefix
                or len(prefix) + gen.max_new_tokens
                > target.config.max_position_embeddings
            ):
                return {
                    "tokens": [],
                    "complete": False,
                    "stats": {"context_capacity_exceeded": True},
                }
            if len(prefix) > budget.limit - budget.used:
                return None
            return decode(
                prefix, ReplayProposer(len(prefix), previous, opts.replay_width)
            )

        roots, events = schedule_roots(
            initial, revise, active_roots=opts.active_roots, max_roots=opts.max_roots
        )
        votes = []
        for root in roots:
            tokens = root["tokens"]
            if opts.answer_extractor == "token_sequence":
                answer = ",".join(map(str, tokens))
            else:
                lines = (
                    tokenizer.decode(tokens, skip_special_tokens=True)
                    .strip()
                    .splitlines()
                )
                answer = lines[-1].strip() if lines else ""
            if answer:
                votes.append(answer)
            root["answer"] = answer
        counts = Counter(votes)
        winner = (
            sorted(counts, key=lambda answer: (-counts[answer], answer))[0]
            if counts
            else None
        )
        answers.append(
            {
                "id": identifier,
                "winner": winner,
                "votes": dict(counts),
                "roots": roots,
                "events": events,
                "processed_target_positions": budget.used,
                "budget_limit": budget.limit,
                "no_vote": not counts,
            }
        )
    return {
        "answers": answers,
        "implementation": "stationary_full_prefix_arr_reference",
        "budget_unit": "actual_full_prefix_processed_target_positions",
        "production_engine_speed": False,
        "answer_extractor": opts.answer_extractor,
    }
