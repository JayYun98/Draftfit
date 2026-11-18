# REAL loss

[The tensor component](method.py) computes the target-sampled accepted-prefix
loss for proposal blocks. Run the [CPU recipe](../../../../examples/configs/research/real-loss.yaml)
using the [research guide](../../../../docs/research/real-loss.md).

The [specification](../../../../docs/specs/real-loss.md) describes the estimator
and its limits. The [loss tests](../../../../tests/research/test_real_loss.py)
check values and gradients; the [recipe tests](../../../../tests/research/test_real_recipe.py)
exercise the CLI and saved results.

See the [source notice](NOTICE.md) for the pinned upstream revision and local
changes, and [REAL_LICENSE](REAL_LICENSE) for the upstream license.
