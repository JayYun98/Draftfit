"""Run with PYTHONPATH=$MLX_RUNTIME/src in the external MLX environment."""
import pytest
import mlx.core as mx

VOCAB = 64
PROMPT = [7, 8, 9, 10, 11, 12]


def _logits(ids):
    rows = []
    for token in ids:
        row = [-100.0] * VOCAB
        row[(int(token) + 1) % VOCAB] = 100.0
        rows.append(row)
    return mx.array([rows])


class _Cache:
    def __init__(self):
        self.offset = 0
        self.k = mx.zeros((1,))


class _Target:
    def make_cache(self):
        return [_Cache()]

    def reset_spec(self):
        pass

    def prefill(self, ids, cache, tap=None, **kwargs):
        cache[0].offset += ids.shape[1]
        return _logits(ids[0].tolist()), mx.zeros((1, ids.shape[1], 1))

    def plain(self, ids, cache):
        cache[0].offset += ids.shape[1]
        return _logits(ids[0].tolist())

    def verify(self, ids, cache, tap):
        cache[0].offset += ids.shape[1]
        return _logits(ids[0].tolist()), mx.zeros((1, ids.shape[1], 1))

    def rollback(self, cache, n_rejected, accepted):
        cache[0].offset -= n_rejected


class _Tok:
    eos_token_id = 999

    def decode(self, ids):
        return " ".join(map(str, ids))

    def convert_tokens_to_ids(self, token):
        return None



from mlx_dspark.generate import greedy_generate
from mlx_dspark.lookup import lookup_generate

@pytest.mark.parametrize('budget', [2, 3, 5, 32])
@pytest.mark.parametrize('eos', [15, 999])
def test_lookup_budget_and_committed_cache(monkeypatch, budget, eos):
    from mlx_dspark import lookup
    class Index:
        def __init__(self, **kwargs): pass
        def extend(self, ids): self.last = ids[-1]
        def propose(self, **kwargs): return [(self.last+i+1)%64 for i in range(8)]
    monkeypatch.setattr(lookup, 'NGramIndex', Index)
    target, tok = _Target(), _Tok()
    tok.eos_token_id = eos
    expected = greedy_generate(target, tok, prompt_ids=PROMPT, max_new_tokens=budget)
    cache = target.make_cache()
    rounds = []
    actual = lookup_generate(target, tok, prompt_ids=PROMPT, max_new_tokens=budget,
                             cache=cache, on_round=lambda **r: rounds.append(r))
    assert actual.token_ids == expected.token_ids
    assert cache[0].offset == len(PROMPT)+len(actual.token_ids)-1
    assert sum(r['committed'] for r in rounds) == len(actual.token_ids)-1
