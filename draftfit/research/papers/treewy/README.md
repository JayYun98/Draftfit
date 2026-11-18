# TreeWY single-tree reference

[method.py](method.py) validates a parent-before-child tree, invokes the
single-tree Gated DeltaNet solver, and returns the output for each node.
Set `accepted_node` to reconstruct the state on that path; use null to retain
the initial state.

## Run

From the repository root, run the [recipe](../../../../examples/configs/research/treewy.yaml)
with its [branching tree fixture](../../../../examples/configs/research/data/treewy.json):

```sh
python -m draftfit.research -c examples/configs/research/treewy.yaml
```

The [state tests](../../../../tests/research/test_latest_kv_tree.py) compare node
outputs and every possible commit against an independent recurrence, including
branching siblings and a non-square state.

## Source and limits

[TreeWY v2](https://arxiv.org/abs/2608.20961v2) describes the method.
[upstream.py](upstream.py) preserves the first 142 lines of the
[author's reference at the pinned revision](https://github.com/sneha5gsm/vllm/blob/b073ed6cacfa1cf5ae23111854729e662a72a6d1/vllm/third_party/flash_linear_attention/ops/tree_wy_ref.py)
under [Apache-2.0](TREEWY_LICENSE); see the [source notice](../../NOTICE.md).
The validated state covers Gated DeltaNet only. Convolution history, attention
KV commits, Triton kernels and vLLM scheduling are outside this experiment.
