# CAST

[CAST at revision
`2eed73f`](https://github.com/js-lee-AI/CAST/tree/2eed73f38f9210f1d5ddf01de8059cf3ffb43c19)
builds a best-first tree from a single DFlash draft pass. [Tree construction](tree.py),
[verification](verify.py), [decoding](decode.py), and [width prediction](width.py)
retain the upstream [MIT license](LICENSE).

**Run.** Start with the [CAST
recipe](../../../../examples/configs/research/valuspec-cast.yaml) and omit its optional
`valuation` section to run plain CAST. Supply a compatible local DFlash/DFlash2
checkpoint, dense full-attention Llama/Qwen2/Qwen3 target, and JSONL rows with
`input_ids` or `prompt`. From the repository root:

```sh
python -m draftfit.research -c path/to/cast.yaml --plan
```

Remove `--plan` to execute. The [decoding
guide](../../../../docs/specs/research-methods.md#cast-reference-decoding) explains
sampling and tree budgets. This is a batch-one reference; its timings do not establish
production serving speed.

**Tests.** [Integration tests](../../tests/test_research_methods.py) check the
tree mask, greedy parity and target sampling distribution. [Boundary
cases](../../tests/test_research_review_regressions.py) cover temperature, EOS and
the final bonus token.
