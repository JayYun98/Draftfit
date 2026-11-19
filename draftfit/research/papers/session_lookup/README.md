# Session lookup

The `session_lookup` CPU recipe proposes a continuation from tokens already
present in one session. It takes the prompt and all tokens generated so far as
one list. It needs no model, tokenizer, or GPU.

## Try the two match orders

From the repository root, run the same context with two match orders. With
a wheel install, first run `python -m draftfit.assets export session-example`
and change into the new `session-example` directory.

```sh
python -m draftfit.research -c examples/configs/research/session-lookup.yaml
python -m draftfit.research -c examples/configs/research/session-lookup.yaml match=earliest output_dir=outputs/research/session-lookup-earliest
```

The example ends in `[1, 2, 3]`. The most recent eligible occurrence is
followed by `[8, 9]`, and the earliest by `[4, 5]`. Each run prints
`proposed_ids` and saves it in `result.json` under its own new `output_dir`.

## Lookup rules

Supply `context_ids` as a list of nonnegative integer token IDs.
Booleans, strings, and fractional values are invalid. Set `k` to the number
of tokens to propose. Use positive integers for `k`, `min_ngram`, and
`max_ngram`, with `min_ngram <= max_ngram`.

The lookup tries suffix lengths from `max_ngram` down to `min_ngram`. For a
suffix of length `n` in a context of length `L`, it searches for the same
tokens earlier in the context. If `s` is the index immediately after a match,
the match is eligible only when `s < L - n` and `s + k <= L`. The first
condition rules out a continuation starting exactly at the query suffix.
The continuation may extend into that suffix.

At the longest length with an eligible match, `match: earliest` picks the
first occurrence and `match: recent` picks the last. The proposal contains
exactly `k` observed tokens, or `[]` if no length qualifies.

In Python, import `propose` from
`draftfit.research.papers.session_lookup.method`. It takes
`propose(context_ids, *, k, min_ngram, max_ngram, match)` and returns the
list directly.

## Provenance and scope

Antal van den Bosch compares these two match orders in
[Text Speeding Itself Up: Recency and Restraint in Session-Local Drafting](https://arxiv.org/abs/2611.19455v1).
Draftfit independently implements the lookup and minimum match length described
in the author's [prompt_lookup_ref.py](https://github.com/antalvdb/olifant-speculative/blob/6d7d17fe37bbfca6e4f0c1e2ca6f8f4c0d853ab8/src/olifant_speculative/draft_models/prompt_lookup_ref.py).
The reference is [licensed under Apache-2.0](https://github.com/antalvdb/olifant-speculative/blob/6d7d17fe37bbfca6e4f0c1e2ca6f8f4c0d853ab8/LICENSE)
and credits [Apoorv Saxena's prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding).
The recipe neither bundles nor executes upstream code.

This recipe produces one proposal. It does not run session memory, an
agreement gate, or target-model verification, and it does not measure the
paper's decoding speed or energy use.
