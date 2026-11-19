# DForkTree candidate scoring reference

## Problem and scope

Block diffusion candidates describe the drafter's preferred continuation.
Candidates on a different branch can need different scores.
Provide an opt-in CPU reference for DForkTree's probability correction.
Reuse the existing research recipe interface.

This component implements equations 2 and 3 from
[DForkTree v1](https://arxiv.org/html/2611.12356v1).
It consumes caller-provided conditional and unigram probabilities.
It does not collect a corpus, count n-grams, build a tree, or run a verifier.
Training, checkpoints, exports, and serving defaults retain their contracts.

## Mathematical contract

For candidate `c`, the evidence is pointwise mutual information, or PMI:

```text
T(c) = clip(log(P(c | branch context)) - log(P(c)), -clip, clip)
```

Let `q(c)` be the original candidate probability and
`rho = 1 - sum(q(c))` be the probability outside the candidate list.
The corrected probabilities are:

```text
Z = rho + sum(q(c) * exp(weight * T(c)))
corrected(c) = q(c) * exp(weight * T(c)) / Z
corrected_tail = rho / Z
```

The tail has no PMI correction before normalization.
Its final probability can change because `Z` changes.
Normalizing only the listed candidates violates this contract.

The first draft position has weight zero.
Later edges use weight `0.3` when the parent path remains on the original spine.
They use weight `0.6` after the parent path has left that spine.
A candidate leaving the spine still uses its parent's classification.
The component consumes that classification; it does not infer a tree history.
The default PMI clip is `10.0`.

Probability inputs must be finite and aligned with unique candidate identities.
Conditional and unigram evidence must be positive, including neutral cases.
Zero original candidate probabilities must remain zero.
Zero correction and the first position must recover the original marginals.
The candidate probabilities and corrected tail must sum to one.
Reject malformed inputs rather than silently renormalizing them.

The recipe selects `method: dforktree_scoring` and accepts `data`, `scoring`,
`input_kind`, and `output_dir`. Scoring options are `spine_weight`, `fork_weight`,
and `pmi_clip`. Their defaults are `0.3`, `0.6`, and `10.0`.
Weights and the clip must be finite and nonnegative.
The public function uses one-based `position` and `parent_path` equal to
`spine` or `forked`. It validates all fields before neutral shortcuts.

Rows contain `id`, `candidate_ids`, `marginals`, `conditional`, `unigram`,
`position`, and `parent_path`. The row schema rejects extra fields.
The adapter scores at most `data.max_samples` nonblank rows.
It hashes every raw byte of the input, including the unprocessed suffix.
The `input_kind` label is caller-supplied provenance, not authentication.

## Acceptance fixed before implementation

| Requirement | Observable check | Status |
| --- | --- | --- |
| Mathematical correction | Analytic PMI and corrected probabilities match equations 2 and 3 | pass |
| Fork classification | The same evidence preserves the spine ranking and reverses the post-fork ranking | pass |
| Tail class | Corrected candidates plus tail sum to one; candidates alone need not sum to one | pass |
| Neutral cases | First position, zero weight, and zero original candidate probability retain their contracts | pass |
| Input boundary | Invalid probabilities, identities, metadata, or shapes fail explicitly | pass |
| Real recipe | Plan reads no input and writes no output; execution persists results in a fresh directory | pass |
| Installed package | Exported recipe, fixture, guide, spec, and source notices work outside the checkout | pass |
| Existing behavior | Affected research and packaging regressions pass | pass |
| Locked Linux CPU | This revision passes its own hosted public CPU gate | not-run |
| Fresh macOS installation | Resolve the existing trusted-artifact hash discrepancy before retrying setup | blocked |
| GPU and serving | Separate implementation and real-runtime parity and performance evidence | not-run |

The fixed synthetic case uses candidates `[101, 102]`, original probabilities
`[0.5, 0.3]`, conditional probabilities `[0.2, 0.4]`, and unigram probabilities
`[0.4, 0.2]`. The outside probability is `0.2`.
The spine result is approximately `[0.4163391976, 0.3786313315]` with tail
`0.2050294709`. The post-fork result is approximately
`[0.3350392805, 0.4618308844]` with tail `0.2031298351`.
These are analytic acceptance values, not measured model outcomes.

## Provenance and compatibility

The reference source is the author's
[Apache-2.0 implementation](https://github.com/Perseus037/DForkTree/tree/eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3).
Relevant files are `dforktree/ngram_gpu.py`, `dforktree/tree_ops.py`, and
`dforktree/fc1_builder.py`.
Retain the author's license and notice with the packaged component.

At this source pin, `ngram_path_fn` returns `fn` without defining it.
This observation comes from source inspection.
The CPU reference does not import or execute that GPU path.
It provides no evidence that the published GPU builder runs.

The reference uses CPU floating-point arithmetic and preserves a true zero tail.
The author's builder uses float32 and clamps its tail to at least `1e-6`.
Those differences prevent a claim of exact GPU numerical parity.
The component accepts precomputed probabilities and does not reproduce the
author's raw count tables, smoothing pipeline, or tokenizer alignment.

## Verification and remaining gates

The [local receipt](../research/results/dforktree-scoring-20261112.json) records
45 research tests, 30 adjacent regression tests, and three packaging tests.
All passed without skips in the scoped contributor environment.
The 26 focused DForkTree tests are included in the research suite.
The documented CLI and installed-wheel recipe also passed.
The original author NOTICE remains byte-identical; only its EOF fixer is excluded.
All changed files passed their applicable pre-commit checks.
The receipt predates hosted CI. Read the PR evidence for that exact revision.
Keep the experimental PR draft while required checks remain open.
The known macOS Torch hash discrepancy must not be bypassed.
An older contributor environment can establish only its scoped test results.
Hosted locked Linux results apply to their exact revision.

No acceptance, quality, output-equivalence, or throughput improvement is claimed.
A later runtime integration needs matching tokenizer and target-corpus identity,
tree construction, verifier state correctness, and real model comparisons.
GPU execution requires separately authorized resources and ownership.
