# REAL loss component

## Problem and scope

Draftfit users need a small way to inspect accepted-prefix training objectives
before changing a drafter's training pipeline. Add an explicit research tensor
component and a reproducible CPU fixture. Production training and serving retain
their current defaults.

This adopts the branch estimator from [REAL Loss](https://arxiv.org/abs/2611.14399v1).
It does not implement D-REAL's draft-sampled rollout objective, a new training
strategy, the author's full curriculum, or a trained checkpoint.

## Mathematical contract

Let `p` be the frozen target distribution and `q` the draft distribution on the
same vocabulary and target-sampled prefix. For each sampled token `y`, compute
`b = min(1, q(y) / p(y))`. Let `a = sum_v min(p(v), q(v))` and let `W_prev`
be the product of earlier `b` values within the same proposal block.

The branch estimate is `J = sum_k W_prev[k] * a[k]`. Its normalized loss is
`1 - J / gamma`, where `gamma` is the fixed proposal width. The default mixed
loss is `0.9 * normalized_loss + 0.1 * CE`. CE uses the argmax of target logits
as the hard label, following the [pinned eager CE callable](https://github.com/King-ty/speculators/blob/3dff4579c5f93f692108df8b5178c9e9200b8536/src/speculators/losses/eager.py). Sampled target IDs select path weights, not the CE label.
The standalone CE has no position decay or author training curriculum.

The estimate excludes the anchor, correction, and bonus tokens. Both the path
weights and vocabulary overlap retain draft gradients. Target tensors are
detached. The `q/p` path ratio is different from the verifier's `p/q` acceptance
ratio. See [the method](https://arxiv.org/html/2611.14399v1#S3).

Unbiasedness requires target-sampled trajectories, aligned conditional
distributions, and the paper's stochastic chain verification assumptions.
The API cannot authenticate sampling provenance. Arbitrary offline logits do
not establish that assumption, real acceptance, or serving speed.

## Input and compatibility contract

- Each tensor block contains proposals only. Callers exclude anchors and split
  documents or packed sequences into separate blocks.
- Draft and target logits have aligned `[blocks, gamma, vocabulary]` dimensions.
- Sampled token IDs and the optional Boolean mask have `[blocks, gamma]` shape.
- The first inactive position terminates its block. Later positions have no
  objective or gradient contribution, even if their input mask is true.
- Inactive labels may use `-100`. Active labels must index the shared vocabulary.
- Empty active batches and malformed or nonfinite inputs fail explicitly.
- The EAL term averages nonempty blocks and retains the fixed `gamma` divisor.
  CE averages retained token positions. An empty block does not dilute either.
- FP64 inputs retain FP64 arithmetic. FP16/BF16 inputs use FP32 loss arithmetic.
  Nonrepresentable normalization or loss results fail explicitly. Vanishingly
  small survival weights can underflow in finite precision.
- Existing training configs, checkpoints, capture, export, and serving adapters
  gain no new strategy flag or support claim from this component.

## Acceptance and evidence

| Requirement | Observable acceptance | Status |
| --- | --- | --- |
| Exact branch weighting | Two-step enumeration matches values and gradients; WTV has the opposite gradient | pass, scoped CPU |
| Prefix gradient | Detaching a path prefix changes the result's draft gradient | pass, scoped CPU |
| Block and mask boundary | A stopped suffix has zero gradient and cannot affect another block | pass, scoped CPU |
| Target boundary | Target gradients remain absent | pass, scoped CPU |
| Differentiability | A small CPU optimizer update changes parameters and lowers fixed loss | pass, scoped CPU |
| Recipe | Plan writes nothing; execution records input identity and result in a fresh directory | pass, actual CPU CLI |
| Distribution | Wheel and sdist retain the module, recipe, fixture, guide, spec, and source license | pass, archive tests |
| Locked Linux CPU | This revision's own public CPU and installed-wheel gate | not-run |
| macOS locked CPU setup | Resolve the existing Torch artifact hash discrepancy without bypass | blocked |
| CUDA parity | Same quantized inputs, target/loss/gradient checks on a compatible GPU | not-run, no current GPU budget |
| Real drafter quality | Target-sampled training data and controlled held-out comparison | not-run, training integration absent |
| Export and serving | Actual trained artifact, supported backend, correctness and speed comparisons | not-run, no trained artifact |

Use the [research guide](../research/real-loss.md) for commands and remaining
prerequisites. Synthetic numerical checks establish only component behavior.

## CPU execution record, 2026-11-13

The contributor environment used Python 3.12.8, Torch 2.11.0 CPU, Transformers
5.8.1, and Pydantic 2.13.4. Its Transformers version differs from the required
5.12.1 environment. These results do not pass the full locked CPU gate.

- Research discovery passed 42 tests, including 23 REAL-specific tests.
- Adjacent research regression checks passed 30 tests.
- Packaging checks passed three tests. The total is 75 distinct tests.
- The documented plan wrote no output. Actual fixture execution recorded 50
  active blocks, 100 positions, mean `J = 0.96`, and REAL loss `0.52`.
- Teacher-argmax CE was `log(10)/2`. The default mixed loss was
  `0.5831292546497023`; the finite draft gradient norm was approximately
  `0.03754328168927166`.
- A separate three-proposal binary-tree oracle enumerated eight target paths.
  Exact expected acceptance was `1.56235492074009`. Maximum value and parameter
  gradient errors were both `2.220446049250313e-16`, below `1e-12` and `1e-11`
  tolerances. The teacher gradient remained absent.

The tested module SHA256 was
`950e6140bec5f272a071fcffa3582fa72a6a69c9f1099727a27a39ec1dc0dc89`.
The fixture SHA256 was
`f8a1d122d97a3b1bf3f84c0c379227bb8121006a506ed07fab94e571e03bc2d4`.
The PR verification record distinguishes contributor execution from hosted
locked-environment checks for its exact head revision.

## Source and license

Adapt only the relevant loss mathematics from
[the author reference](https://github.com/King-ty/speculators/blob/3dff4579c5f93f692108df8b5178c9e9200b8536/src/speculators/models/dflash/metrics.py),
commit `3dff4579c5f93f692108df8b5178c9e9200b8536`, under Apache-2.0.
Retain its exact license and identify local changes. The source README excludes
the internal training implementation and full paper reproduction scripts.
