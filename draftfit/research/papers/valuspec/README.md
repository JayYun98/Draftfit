# ValuSpec

[ValuSpec](https://arxiv.org/abs/2610.31338v1) scores proposed branches with a frozen
autoregressive model. This [independent
implementation](../../NOTICE.md#independently-implemented-published-algorithms)
calibrates a threshold on target-greedy traces. It scores each CAST leaf path, retains
its ancestors if the path passes, and lets the target verify the remaining tree. The
linear quantile and ancestor union are local conventions.

**Run.** The [ValuSpec + CAST
recipe](../../../../examples/configs/research/valuspec-cast.yaml) uses `method: cast`
and a `valuation` section. Supply a target, DFlash draft, and scorer with
target-compatible token IDs; use disjoint evaluation and calibration prompts. From the
repository root:

```sh
python -m draftfit.research -c examples/configs/research/valuspec-cast.yaml --plan
```

Remove `--plan` to execute. Filtering supports greedy decoding only. An empty retained
tree is valid; filtering does not guarantee a speed gain.

**Tests.** [KV/tree regressions](../../../../tests/research/test_latest_kv_tree.py)
compare conditional scores with explicit autoregressive scores, check calibration
leakage, and verify greedy parity when every candidate is filtered out.
