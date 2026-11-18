# Phase-scoped query layouts (PQ/QB)

[method.py](method.py) compares two ways of computing grouped-query attention:
packing queries into one sequence or treating occupied queries as a batch.
Each query retains its additive mask. The diagnostic reports both outputs and
their maximum absolute difference, or why the role, phase or query count was
not admitted.

## Run

From the repository root, run the [recipe](../../../../examples/configs/research/query-layout.yaml)
against its [tensor fixture](../../../../examples/configs/research/data/query-layout.json):

```sh
python -m draftfit.research -c examples/configs/research/query-layout.yaml
```

The [layout tests](../../../../tests/research/test_latest_control.py) compare
against per-head attention and cover noncontiguous tensors, masked keys, phase
admission and the CLI.

## Source and limits

Draftfit implements a floating-point diagnostic based on
[phase-scoped query layouts v4](https://arxiv.org/abs/2610.03228v4);
see the [source notice](../../NOTICE.md). The experiment admits at most four
occupied queries. It does not include the HeteroSpec runtime or establish GGML
quantization, graph reuse or endpoint token parity.
