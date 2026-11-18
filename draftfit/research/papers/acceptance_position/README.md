# Acceptance by output position

The [position study](https://arxiv.org/abs/2610.30644v1) motivates this [local
diagnostic](../../NOTICE.md#independently-implemented-published-algorithms). The
[verifier](../../reference_decode.py) records each attempted token's output position,
acceptance, distribution overlap and argmax agreement. [Position bins](method.py)
summarize these observations in the [shared diagnostics harness](../../diagnostics.py).

**Use.** Enable `diagnostics.record_acceptance` in a memory or RIFT recipe; set
`position_bin_size` for the bin width. The [NgramRL
recipe](../../../../examples/configs/research/ngramrl.yaml) enables recording. Inspect
`answers[].stats.token_acceptance` and `answers[].stats.position_bins` after running it.

Only attempted tokens enter the denominator. Verification stops at a rejected token, so
its untried suffix is censored. These bins cannot establish the paper's position-decay
result.

**Tests.** [RIFT integration
coverage](../../../../tests/research/test_latest_kv_tree.py) exercises position bins and
paired timing on a local target. [Diagnostic
tests](../../../../tests/research/test_memory_tree_layout.py) check the observed
distribution and denominator.
