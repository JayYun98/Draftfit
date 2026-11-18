"""Bounded user-local retrieval with explicit shared seeds (arXiv:2610.31840v1)."""

import math
from collections import OrderedDict

from ... import proposals


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
        values = proposals.token_tuple(tokens)
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
