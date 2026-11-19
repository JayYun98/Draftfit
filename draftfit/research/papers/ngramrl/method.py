"""Synchronous n-gram proposals from bounded verified history (arXiv:2610.32881v1)."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

from ... import proposals


@dataclass(frozen=True)
class NgramSnapshot:
    counts: dict[tuple[int, ...], Counter]
    n_min: int
    n_max: int
    version: int

    def successors(self, context):
        context = proposals.token_tuple(context)
        for width in range(min(self.n_max - 1, len(context)), self.n_min - 2, -1):
            counts = self.counts.get(context[-width:] if width else ())
            if counts:
                return dict(
                    sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                )
        return {}

    def propose(self, context, length, *, sampling=False, generator=None):
        context = proposals.token_tuple(context)
        if length < 0:
            raise ValueError("proposal length must be nonnegative")
        tokens, probabilities = [], []
        for _ in range(length):
            counts = self.successors(context + tuple(tokens))
            if not counts:
                break
            total = sum(counts.values())
            distribution = {token: count / total for token, count in counts.items()}
            if sampling:
                import torch

                index = int(
                    torch.multinomial(
                        torch.tensor(list(distribution.values()), dtype=torch.float64),
                        1,
                        generator=generator,
                    )
                )
                token = list(distribution)[index]
            else:
                token = next(iter(counts))
                distribution = {token: 1.0}
            tokens.append(token)
            probabilities.append(distribution)
        return proposals.Proposal(tuple(tokens), tuple(probabilities))


class NgramMemory:
    """Per-prompt frequency tables reconstructed from bounded verified history."""

    def __init__(
        self,
        *,
        n_min=2,
        n_max=5,
        window_steps=8,
        max_prompts=128,
        max_rollouts_per_prompt=16,
        max_tokens_per_rollout=4096,
    ):
        if not 1 <= n_min <= n_max:
            raise ValueError("ngram orders require 1 <= n_min <= n_max")
        if (
            min(
                window_steps,
                max_prompts,
                max_rollouts_per_prompt,
                max_tokens_per_rollout,
            )
            < 1
        ):
            raise ValueError("memory limits must be positive")
        self.n_min, self.n_max = n_min, n_max
        self.window_steps, self.max_prompts = window_steps, max_prompts
        self.max_rollouts_per_prompt = max_rollouts_per_prompt
        self.max_tokens_per_rollout = max_tokens_per_rollout
        self.history = OrderedDict()
        self.step = -1
        self.version = 0

    def _advance(self, step):
        if type(step) is not int or step < self.step or step < 0:
            raise ValueError("cache steps must be nonnegative and monotonic")
        self.step = step
        minimum = step - self.window_steps + 1
        for key in list(self.history):
            retained = [(s, t) for s, t in self.history[key] if s >= minimum]
            if len(retained) != len(self.history[key]):
                self.version += 1
            if retained:
                self.history[key] = retained
            else:
                del self.history[key]

    def add_verified(self, prompt, tokens, *, step):
        values = proposals.token_tuple(tokens)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt scope must be a nonempty string")
        if len(values) > self.max_tokens_per_rollout:
            raise ValueError("verified rollout exceeds the configured memory bound")
        self._advance(step)
        history = self.history.setdefault(prompt, [])
        history.append((step, values))
        del history[: -self.max_rollouts_per_prompt]
        self.history.move_to_end(prompt)
        while len(self.history) > self.max_prompts:
            self.history.popitem(last=False)
        self.version += 1

    def snapshot(self, prompt, *, step):
        self._advance(step)
        counts = {}
        for _, tokens in self.history.get(prompt, []):
            for n in range(self.n_min, self.n_max + 1):
                for end in range(n, len(tokens) + 1):
                    context = tokens[end - n : end - 1]
                    counts.setdefault(context, Counter())[tokens[end - 1]] += 1
        return NgramSnapshot(counts, self.n_min, self.n_max, self.version)

    def state_dict(self):
        return {
            "step": self.step,
            "version": self.version,
            "history": [
                [key, [[s, list(t)] for s, t in rows]]
                for key, rows in self.history.items()
            ],
        }

    def load_state_dict(self, state):
        if (
            type(state["step"]) is not int
            or state["step"] < -1
            or type(state["version"]) is not int
            or state["version"] < 0
        ):
            raise ValueError("saved ngram step/version must be valid integers")
        restored = NgramMemory(
            n_min=self.n_min,
            n_max=self.n_max,
            window_steps=self.window_steps,
            max_prompts=self.max_prompts,
            max_rollouts_per_prompt=self.max_rollouts_per_prompt,
            max_tokens_per_rollout=self.max_tokens_per_rollout,
        )
        if len(state["history"]) > self.max_prompts:
            raise ValueError("saved prompt count exceeds the memory bound")
        rows = []
        for prompt, history in state["history"]:
            if len(history) > self.max_rollouts_per_prompt:
                raise ValueError("saved rollout count exceeds the memory bound")
            rows.extend((step, prompt, tokens) for step, tokens in history)
        if any(
            type(step) is not int or not 0 <= step <= state["step"]
            for step, _, _ in rows
        ):
            raise ValueError("saved rollout steps must precede the snapshot step")
        for step, prompt, tokens in sorted(rows, key=lambda row: row[0]):
            restored.add_verified(prompt, tokens, step=step)
        if state["step"] >= 0:
            restored._advance(state["step"])
        restored.version = state["version"]
        order = {key: index for index, (key, _) in enumerate(state["history"])}
        restored.history = OrderedDict(
            sorted(restored.history.items(), key=lambda row: order[row[0]])
        )
        self.history, self.step, self.version = (
            restored.history,
            restored.step,
            restored.version,
        )
