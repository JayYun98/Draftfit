# SpecMem

[SpecMem §3.3](https://arxiv.org/abs/2610.31840v1) describes retrieval for speculative
tool use. This [independent
implementation](../../NOTICE.md#independently-implemented-published-algorithms)
retrieves by cosine similarity from a user's bounded history, then tries explicitly
supplied shared seeds and schema tokens. The target verifies every proposed token.

**Run.** Set a local dense target and JSONL input in the
[recipe](../../../../examples/configs/research/specmem.yaml). Each row supplies
`input_ids`, a `user_id`, and a finite nonzero `query_embedding` with the same dimension
as other entries. From the repository root:

```sh
python -m draftfit.research -c examples/configs/research/specmem.yaml --plan
```

Remove `--plan` to execute. The [memory runner](../../memory_runner.py) writes back
locally verified output according to `memory.writeback_mode`. It consumes supplied
embeddings and does not execute tools or validate their schemas.

**Tests.** [Memory regressions](../../../../tests/research/test_latest_research.py)
exercise user isolation, seed and schema fallback, eviction, state reload, and verified
writeback.
