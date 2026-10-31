"""Bounded native reference policies from NgramRL and SpecMem.

Algorithms: arxiv.org/html/2610.32881v1 (3.2),
arxiv.org/html/2610.31840v1 (3.3). No unpublished upstream code is copied.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass


def token_tuple(tokens):
    values = tuple(tokens)
    if any(type(token) is not int or token < 0 for token in values):
        raise ValueError("token IDs must be nonnegative integers")
    return values


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[int, ...] = ()
    probabilities: tuple[dict[int, float], ...] = ()

    def __post_init__(self):
        token_tuple(self.tokens)
        if len(self.tokens) != len(self.probabilities):
            raise ValueError("each proposal token needs its actual distribution")
        for token, distribution in zip(self.tokens, self.probabilities):
            token_tuple(distribution)
            if (
                not distribution
                or any(not math.isfinite(p) or p <= 0 for p in distribution.values())
                or not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-7)
                or token not in distribution
            ):
                raise ValueError(
                    "proposal probabilities must define normalized support"
                )


@dataclass(frozen=True)
class NgramSnapshot:
    counts: dict[tuple[int, ...], Counter]
    n_min: int
    n_max: int
    version: int

    def successors(self, context):
        context = token_tuple(context)
        for width in range(min(self.n_max - 1, len(context)), self.n_min - 2, -1):
            counts = self.counts.get(context[-width:] if width else ())
            if counts:
                return dict(
                    sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                )
        return {}

    def propose(self, context, length, *, sampling=False, generator=None):
        context = token_tuple(context)
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
        return Proposal(tuple(tokens), tuple(probabilities))


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
        values = token_tuple(tokens)
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


def normalized_embedding(values):
    values = tuple(float(x) for x in values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("query embeddings must be finite nonempty vectors")
    norm = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("query embeddings must have a finite positive norm")
    return tuple(x / norm for x in values)


class SpecMemMemory:
    """Bounded user-local LRU; shared fallback is explicit immutable seed data."""

    def __init__(
        self,
        *,
        capacity=48,
        max_users=128,
        threshold=0.35,
        max_tokens_per_entry=4096,
        max_embedding_dimensions=4096,
    ):
        if (
            min(capacity, max_users, max_tokens_per_entry, max_embedding_dimensions) < 1
            or not -1 <= threshold <= 1
        ):
            raise ValueError("invalid SpecMem limits or cosine threshold")
        self.capacity, self.max_users, self.threshold = capacity, max_users, threshold
        self.max_tokens_per_entry = max_tokens_per_entry
        self.max_embedding_dimensions = max_embedding_dimensions
        self.users = OrderedDict()
        self.shared = []
        self.dimension = None

    def _embedding(self, values):
        if len(values) > self.max_embedding_dimensions:
            raise ValueError("query embedding exceeds the dimension bound")
        embedding = normalized_embedding(values)
        if self.dimension is not None and len(embedding) != self.dimension:
            raise ValueError("query embedding dimensions must match")
        self.dimension = len(embedding)
        return embedding

    def _tokens(self, tokens):
        values = token_tuple(tokens)
        if len(values) > self.max_tokens_per_entry:
            raise ValueError("SpecMem token entry exceeds configured memory bound")
        return values

    def add_shared_seed(self, embedding, tokens):
        if len(self.shared) >= self.capacity:
            raise ValueError("shared seed exceeds configured capacity")
        values = self._tokens(tokens)
        self.shared.append((self._embedding(embedding), values))

    def add_verified(self, user, query_key, embedding, tokens):
        if (
            not isinstance(user, str)
            or not user
            or not isinstance(query_key, str)
            or not query_key
        ):
            raise ValueError("user and query keys must be nonempty strings")
        values = self._tokens(tokens)
        entry = (self._embedding(embedding), values)
        store = self.users.setdefault(user, OrderedDict())
        store[query_key] = entry
        store.move_to_end(query_key)
        while len(store) > self.capacity:
            store.popitem(last=False)
        self.users.move_to_end(user)
        while len(self.users) > self.max_users:
            self.users.popitem(last=False)

    def retrieve(self, user, embedding, *, schema_tokens=()):
        if not isinstance(user, str) or not user:
            raise ValueError("user scope must be a nonempty string")
        query = self._embedding(embedding)
        store = self.users.get(user, {})
        candidates = [(key, item) for key, item in store.items()]
        for source, entries in (
            ("user", candidates),
            ("shared_seed", list(enumerate(self.shared))),
        ):
            if not entries:
                continue
            key, (vector, tokens) = max(
                entries, key=lambda row: sum(a * b for a, b in zip(query, row[1][0]))
            )
            score = sum(a * b for a, b in zip(query, vector))
            if score >= self.threshold:
                if source == "user":
                    store.move_to_end(key)
                    self.users.move_to_end(user)
                return tokens, {"source": source, "similarity": score}
        return self._tokens(schema_tokens), {"source": "schema", "similarity": None}

    def state_dict(self):
        return {
            "dimension": self.dimension,
            "shared": self.shared,
            "users": [
                [
                    user,
                    [
                        [key, [vector, tokens]]
                        for key, (vector, tokens) in store.items()
                    ],
                ]
                for user, store in self.users.items()
            ],
        }

    def load_state_dict(self, state):
        restored = SpecMemMemory(
            capacity=self.capacity,
            max_users=self.max_users,
            threshold=self.threshold,
            max_tokens_per_entry=self.max_tokens_per_entry,
            max_embedding_dimensions=self.max_embedding_dimensions,
        )
        if len(state["users"]) > self.max_users:
            raise ValueError("saved user count exceeds the memory bound")
        for embedding, tokens in state["shared"]:
            restored.add_shared_seed(embedding, tokens)
        for user, entries in state["users"]:
            if len(entries) > self.capacity:
                raise ValueError("saved user store exceeds the memory bound")
            for key, (embedding, tokens) in entries:
                restored.add_verified(user, key, embedding, tokens)
        if state["dimension"] is not None and (
            type(state["dimension"]) is not int
            or not 1 <= state["dimension"] <= self.max_embedding_dimensions
            or restored.dimension not in (None, state["dimension"])
        ):
            raise ValueError("saved embedding dimension is inconsistent")
        if state["dimension"] is None and restored.dimension is not None:
            raise ValueError("saved entries require an embedding dimension")
        self.users, self.shared, self.dimension = (
            restored.users,
            restored.shared,
            state["dimension"],
        )
