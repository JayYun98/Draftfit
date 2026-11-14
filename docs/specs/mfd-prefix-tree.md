# MFD best-mode prefix tree reference

## Problem and scope

Collapsed position marginals can combine incompatible futures from different
draft modes. Multi-Future Drafting (MFD) retains those modes when it ranks
candidate prefixes for target verification.

Add an explicitly selected CPU research reference for Section 4.3 of
[MFD v1](https://arxiv.org/abs/2611.13017v1). It consumes supplied factorized
mode distributions and selects distinct prefixes within a node budget.
The reference does not load a model or modify the training and serving routes.

## Selection contract

For a nonempty token prefix `y[1:d]`, the score is
`max_k(log(prior[k]) + sum_i(log(q[k, i, y[i]])))`.
The score is a best-mode ranking value. It is not the log probability under
the normalized mixture, which would use a sum over modes.

The budget counts distinct non-root prefixes. Every selected child has a
selected parent, and the parent precedes its child. Equal scores prefer shorter
prefixes, then token order. Repeated prefixes from different modes consume one
node. Zero-probability prefixes do not appear in the output.

Supplied probabilities must describe a common vocabulary at every position
and mode. The selector must not silently normalize a truncated token subset.
It rejects invalid geometry, nonfinite values, negative probabilities, invalid
normalization, and invalid budgets or priors. The CLI writes JSON-safe results
through the existing research runner and refuses an occupied output directory.

## Input and CPU bounds

Each JSONL row contains `probabilities[K][depth][vocabulary]` and optional
`priors[K]`. Missing priors are uniform. Token IDs are vocabulary column indices.
Every probability row and the prior vector must sum to one within `1e-8`.
Accepted values are retained without rescaling. Values must be finite numbers
in `[0, 1]`. Booleans are invalid. Zero priors and zero edges have no support.

The CPU reference accepts at most 8 modes, 16 positions, 4096 vocabulary entries,
and 262144 probability values per row. The node budget is an integer in `[0, 256]`.
The recipe accepts at most 32 input rows. Each JSONL line is limited to 8 MiB.
These bounds support offline component experiments. They do not describe the
author engine's supported dimensions.

The selector uses binary64 log scores and a strict heap order. Nearly equal
mathematical scores can differ after floating-point accumulation. Even equal
exact products can produce different rounded log sums. The tie rule applies
when computed log scores are equal. No tolerance is used inside the heap
comparator. A caller can conceal truncation by supplying
already normalized subset probabilities. The input contract requires the
complete modeled vocabulary, and the selector cannot authenticate that claim.

Each mode-parent ranks children by its actual computed total and token ID.
It retains at most the node budget's number of children. An omitted child has
that many distinct sibling prefixes ahead of it. The global heap advances one
sibling at a time. This CPU reference evaluates each expanded parent's full
probability row. It does not claim the author's optimized construction cost.

## Recipe

Run from the repository root in the [CPU environment](../ENVIRONMENT.md):

```sh
python -m draftfit.research -c examples/configs/research/mfd-prefix-tree.yaml --plan
python -m draftfit.research -c examples/configs/research/mfd-prefix-tree.yaml
```

The included JSON fixture is synthetic. It has two modes over two tokens,
with priors `[0.6, 0.4]`. Both positions use mode rows `[0.9, 0.1]` and
`[0.1, 0.9]`. For a four-node budget, best-mode ranking selects
`[0]`, `[0, 0]`, `[1]`, and `[1, 1]` in that order.
Collapsing the same modes to independent marginals instead selects
`[0]`, `[1]`, `[0, 0]`, and `[0, 1]`.
This arithmetic example establishes a difference in selection, not model quality.

Use a new output directory for each execution. The shared runner persists the
recipe and result atomically. `--plan` loads no data or model and writes nothing.

## Acceptance criteria

| Behavior | Decisive check |
| --- | --- |
| Correct best-mode score and selection | Independent exhaustive enumeration on small distributions across budgets |
| Modes preserve coherent futures | Same-input comparison with collapsed independent marginals |
| Prefix closure and deterministic ties | Deterministic distributions, ties, zero probabilities, and duplicate modes |
| Single-mode reduction | Compare with exhaustive single-mode prefix ranking |
| Explicit recipe selection | Actual CLI plan and run, malformed-input failure, occupied-output rejection |
| Distributable reference | Wheel and source distribution contain the module, recipe, fixture, documentation, and source notices |
| Existing behavior | Relevant research and package regression checks |

## Source identity and rights

The [official project](https://hhhhhhy5.github.io/multi-future-drafting/) links
the [author repository](https://github.com/hhhhhhy5/multi-future-drafting).
The reviewed revision is `95040343de00f15aef8c6350a578c35e0cd53475`.
The algorithm reference is `inference/hf/ddtree.py::build_mixture_tree`.
The repository root has an Apache-2.0 license. The HF subtree has a separate
MIT license, Copyright (c) 2026 Liran Ringel. The source notices retain both
licenses and their scope. No checkpoints, patched inference engines, or external
setup scripts are included.

## Validation boundaries

This component selects a tree from supplied probabilities. It does not implement
MFD mode embeddings, winner-takes-all training, attention packing, target
verification, cache compaction, adaptive mode switching, or a GPU tree builder.
Training, model quality, export, serving correctness, and throughput remain
not-run. CPU arithmetic tests do not establish GPU correctness or a speed gain.

The known macOS hash-lock installation block remains outside this change.
Contributor checks in a reused environment and each revision's hosted Linux
locked gate must be reported separately.

## Recorded component checks

The [2026-11-14 receipt](../research/results/mfd-prefix-tree-20261114.json)
records 19 MFD tests and 52 existing research and packaging tests, with no skips.
The rational exhaustive oracle covered 72 seeded small-table cases across their
budgets. The fixture selected the four prefixes stated above.
A regression reproduced and fixed a hidden sibling's incorrect token order
when extreme parent probabilities rounded different child totals to equality.
Wheel and source distribution checks retained the module, assets, and both
license payloads. Installed-wheel CPU onboarding passed in a probe that reused
the contributor dependencies. The receipt retains an earlier probe isolation
failure and the unchanged macOS lock block.

These checks used Transformers 5.8.1. They do not establish the required 5.12.1
locked gate. Read the exact published head's PR evidence for hosted Linux CI.
MFD model, GPU, quality, and serving gates remain not-run.
