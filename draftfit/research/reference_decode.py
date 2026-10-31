"""Exact batch-one full-prefix verification for model-free proposal policies."""

from __future__ import annotations

import math
import time

import torch

from .memory import Proposal


def distribution(logits, temperature, top_p=1.0):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scores = logits.detach().to(device="cpu", dtype=torch.float64)
    if (
        torch.isnan(scores).any()
        or torch.isposinf(scores).any()
        or not torch.isfinite(scores).any()
    ):
        raise ValueError("target logits do not define a finite distribution")
    if temperature == 0:
        result = torch.zeros_like(scores)
        result[int(scores.argmax())] = 1
        return result
    result = torch.softmax((scores - scores.max()) / temperature, dim=-1)
    if top_p < 1:
        ordered, indices = result.sort(descending=True, stable=True)
        # Keep the first token whose inclusion crosses the nucleus boundary.
        keep = ordered.cumsum(-1) - ordered < top_p
        result = torch.zeros_like(result).scatter(0, indices, ordered * keep)
        result /= result.sum()
    return result


def verify_token(p, token, q, *, generator=None):
    """Standard speculative accept/reject with the actual sampled proposal q."""
    if any(type(index) is not int or not 0 <= index < len(p) for index in q):
        raise ValueError("proposal token is outside the target vocabulary")
    dense_q = torch.zeros_like(p)
    for index, value in q.items():
        dense_q[index] = value
    acceptance = min(1.0, float(p[token]) / q[token])
    if float(torch.rand((), generator=generator)) < acceptance:
        return token, True
    residual = (p - dense_q).clamp_min(0)
    mass = float(residual.sum())
    if mass <= 0:
        raise ValueError("rejected proposal has no positive residual mass")
    return int(torch.multinomial(residual / mass, 1, generator=generator)), False


@torch.inference_mode()
def generate(
    target,
    input_ids,
    propose,
    *,
    max_new_tokens=128,
    stop_token_ids=(),
    temperature=0.0,
    generator=None,
    top_p=1.0,
    forward_budget=None,
    record_acceptance=False,
):
    """Verify every committed token; no cache mutation or tool execution occurs here.

    ``propose`` receives (full committed context, remaining cap, CPU generator).
    It returns tokens and their immutable actual proposal distributions.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("reference decoding requires one nonempty prompt")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("prompt tokens must be integer IDs")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    if str(generator.device) != "cpu":
        raise ValueError("the reference sampler requires a CPU generator")
    context = input_ids[0].tolist()
    prompt_length = len(context)
    vocab = int(target.config.vocab_size)
    if min(context) < 0 or max(context) >= vocab:
        raise ValueError("prompt token is outside the target vocabulary")
    stops = set(stop_token_ids)
    if any(type(token) is not int or not 0 <= token < vocab for token in stops):
        raise ValueError("stop token is outside the target vocabulary")
    stats = {
        "target_forwards": 0,
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "generated_tokens": 0,
        "rounds": [],
        "timing_scope": "full_prefix_reference",
        "stopped_for_budget": False,
        "processed_target_positions": 0,
    }
    if record_acceptance:
        stats["token_acceptance"] = []
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    elif input_ids.device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    while len(context) - prompt_length < max_new_tokens:
        remaining = max_new_tokens - (len(context) - prompt_length)
        proposal = propose(tuple(context), remaining, generator)
        if not isinstance(proposal, Proposal) or len(proposal.tokens) > remaining:
            raise ValueError("proposal must respect the remaining token cap")
        if proposal.tokens and max(proposal.tokens) >= vocab:
            raise ValueError("proposal token is outside the target vocabulary")
        evaluated = context + list(proposal.tokens)
        maximum = getattr(target.config, "max_position_embeddings", None)
        if maximum is not None and len(evaluated) > maximum:
            raise ValueError(
                "prompt plus verification tokens exceeds target context capacity"
            )
        if forward_budget is not None and not forward_budget.consume(len(evaluated)):
            stats["stopped_for_budget"] = True
            break
        out = target(
            torch.tensor([evaluated], device=input_ids.device), use_cache=False
        )
        logits = out.logits[0]
        stats["target_forwards"] += 1
        stats["processed_target_positions"] += len(evaluated)
        stats["proposed_tokens"] += len(proposal.tokens)
        offset = len(context) - prompt_length
        committed, accepted = [], 0
        rejected = False
        for index, token in enumerate(proposal.tokens):
            p = distribution(logits[len(context) - 1 + index], temperature, top_p)
            if temperature == 0:
                emitted = int(p.argmax())
                ok = emitted == token
            else:
                emitted, ok = verify_token(
                    p, token, proposal.probabilities[index], generator=generator
                )
            if record_acceptance:
                q = proposal.probabilities[index]
                if any(i >= vocab for i in q):
                    raise ValueError(
                        "proposal distribution is outside the target vocabulary"
                    )
                overlap = sum(min(float(p[i]), qi) for i, qi in q.items())
                q_argmax = min(q, key=lambda i: (-q[i], i))
                stats["token_acceptance"].append(
                    {
                        "output_position": offset + index,
                        "proposed_token": token,
                        "accepted": ok,
                        "distribution_overlap": overlap,
                        "argmax_agreement": q_argmax == int(p.argmax()),
                    }
                )
            committed.append(emitted)
            accepted += int(ok)
            if not ok:
                rejected = True
            if not ok or emitted in stops:
                break
        if (not committed or (not rejected and committed[-1] not in stops)) and len(
            committed
        ) < remaining:
            p = distribution(
                logits[len(context) - 1 + len(proposal.tokens)], temperature, top_p
            )
            committed.append(
                int(p.argmax())
                if temperature == 0
                else int(torch.multinomial(p, 1, generator=generator))
            )
        if not committed:
            raise RuntimeError("verification made no progress")
        context.extend(committed)
        stats["accepted_tokens"] += accepted
        stats["rounds"].append(
            {
                "output_position": offset,
                "proposed": len(proposal.tokens),
                "accepted": accepted,
                "committed": len(committed),
            }
        )
        if committed[-1] in stops:
            break
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    elif input_ids.device.type == "mps":
        torch.mps.synchronize()
    stats["elapsed_seconds"] = time.perf_counter() - started
    stats["generated_tokens"] = len(context) - prompt_length
    return (
        torch.tensor([context], dtype=input_ids.dtype, device=input_ids.device),
        stats,
    )
