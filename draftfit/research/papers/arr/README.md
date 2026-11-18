# Adaptive Replay Refinement (ARR)

[method.py](method.py) revises several candidate answers under one budget per
question. Each root casts one vote for its last complete answer. An incomplete
revision keeps the previous answer; an incomplete initial answer cannot vote.

## Run

In the [recipe](../../../../examples/configs/research/arr.yaml), set the local
target checkpoint, question JSONL path and stop token IDs for that tokenizer.
Run from the repository root:

```sh
python -m draftfit.research -c examples/configs/research/arr.yaml --plan
python -m draftfit.research -c examples/configs/research/arr.yaml
```

The result records votes, retirement reasons and processed target positions.
The [replay and budget tests](../../tests/test_latest_control.py)
exercise both a deterministic fixed point and a budget too small to admit a root.

## Source and limits

Independent implementation of [ARR v1](https://arxiv.org/abs/2610.28294v1).
No author code or prompt was copied; see the [source notice](../../NOTICE.md).
This serial reference runs full prefixes on dense Llama/Qwen targets and charges
positions processed. Its budget does not measure cached FLOPs or batched serving
latency.
