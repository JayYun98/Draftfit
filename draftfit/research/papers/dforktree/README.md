# DForkTree scoring

[method.py](method.py) implements equations 2 and 3 of DForkTree on CPU.
It takes candidate probabilities and branch evidence supplied by the caller.

Run the [sample recipe](../../../../examples/configs/research/dforktree-scoring.yaml)
with the [scoring guide](../../../../docs/research/dforktree-scoring.md).
The [specification](../../../../docs/specs/dforktree-scoring.md) describes the
numerical contract exercised by the [tests](../../tests/test_dforktree.py).

See [source attribution](../../NOTICE.md#dforktree-scoring-reference) for the
upstream revision and implementation differences. The original
[license](DFORKTREE_LICENSE) and [notice](DFORKTREE_NOTICE) are included here.
