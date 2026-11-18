# D3D corrected forward KL

[method.py](method.py) implements the label-margin correction and weighted
forward KL from D3D. The [CPU recipe](../../../../examples/configs/research/d3d-corrected-kl.yaml)
evaluates the loss on a synthetic fixture.

See the [usage guide](../../../../docs/research/d3d-corrected-kl.md) and
[numerical contract](../../../../docs/specs/d3d-corrected-kl.md).
The [component tests](../../../../tests/research/test_d3d_corrected_kl.py)
cover values and gradients; the [recipe tests](../../../../tests/research/test_d3d_recipe.py)
cover CLI behavior.

The implementation adapts the [authors' loss code](https://github.com/fudan-generative-vision/D3D/blob/cd7762436217fb51d7e7a310b310b72766bb816f/training/pruning/losses.py).
See the [source attribution](../../NOTICE.md#d3d-corrected-kl) and [MIT license](D3D_LICENSE).
