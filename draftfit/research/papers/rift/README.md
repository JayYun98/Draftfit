# RIFT draft cache compression

[method.py](method.py) uses eager attention scores to select older draft KV
entries per head and retain recent positions. It gathers those entries into the
active cache, proposes tokens at their logical positions, and verifies the tokens
with the target.

## Run

Set the local target, autoregressive draft and evaluation JSONL paths in the
[recipe](../../../../examples/configs/research/rift.yaml). Both models must use
dense full attention and the same token-ID mapping. From the repository root:

```sh
python -m draftfit.research -c examples/configs/research/rift.yaml --plan
python -m draftfit.research -c examples/configs/research/rift.yaml
```

The [cache tests](../../tests/test_latest_kv_tree.py) check the
physical gather, active KV bytes and exact target output on a tiny model.

## Source and limits

Independent implementation of the attention selector in
[RIFT v1](https://arxiv.org/abs/2610.31098v1); see the [source notice](../../NOTICE.md).
Each round rebuilds the full committed history before compressing the proposal
cache. A smaller active cache therefore does not imply lower peak memory.
Adaptive refresh, offload and compression-aware distillation are outside this
reference.
