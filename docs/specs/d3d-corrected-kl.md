# Experimental D3D corrected KL component

## Problem and scope

Researchers need a reusable D3D recovery loss component before integrating a full
diffusion drafter or pruning system. A fixed full-depth drafter can disagree with
the target-generated response label. Plain distillation can preserve that error.

Provide an opt-in tensor helper and an offline research recipe for the corrected
forward KL term in D3D section 3.3 and Appendix A. This component does not implement
the complete recovery objective, a D3D model, layer selection, or asynchronous serving.
Existing training configuration, checkpoints, and inference defaults retain their contracts.

## Contract

Let `q` be the frozen full-depth drafter distribution and `y` the supervised
target-response token. With `m = q[y] - max(q[v] for v != y)`, use the smallest
nonnegative mixture `eta = max(0, (delta - m) / (1 - m))`. The corrected target is
`(1 - eta) * q + eta * one_hot(y)`. Leave distributions that meet the margin unchanged.

- Student and teacher logits use equal floating `[positions, vocabulary]` or
  `[batch, positions, vocabulary]` shapes. The vocabulary has at least two entries.
- Labels and nonnegative weights match the leading dimensions and device.
- Weights select valid prediction positions and define the positional decay.
  Callers must exclude anchors and padding, and reset decay for each packed block.
- Positive-weight labels must be valid vocabulary indices. Zero-weight positions
  may use `-100` and receive no student gradient.
- Require finite logits, positive finite temperature, `0 < delta < 1`, and a
  positive weight sum. Reject malformed or empty input explicitly.
- Detach the teacher. Compute a weighted mean of `KL(corrected_teacher || student)`
  and multiply by `temperature ** 2`.
- Use at least float32 arithmetic for half-precision inputs. Preserve float64
  inputs for mathematical checks.

Both distributions must use the same vocabulary, target response, block input,
and attention pattern. The tensor interface cannot authenticate these conditions.
The teacher is the original trained drafter, not the target verifier.

## Acceptance and evidence

| Requirement | Observable criterion | Status |
| --- | --- | --- |
| Minimal correction | Analytic cases meet the requested label margin and preserve other-token ratios | pass |
| Identity | A teacher that already meets the margin remains unchanged | pass |
| Correct objective | Small examples equal hand-computed forward KL with actual weight normalization | pass |
| Gradient boundary | Student gets a finite optimizer update; teacher and ignored positions get no gradient | pass |
| Input boundary | Invalid shapes, labels, weights, scalar parameters, and nonfinite values fail | pass |
| Actual CLI | Plan writes nothing; execution writes finite results into a fresh output directory | pass |
| Distribution | Wheel and rebuilt sdist retain the recipe, fixture, helper, guide, and author license | pass |
| Local locked CPU gate | Install the current macOS CPU profile and complete the full gate | blocked |
| Hosted locked CPU gate | Run this new revision through Linux CI | pending publication |
| Real training and serving | Full drafter recovery, export, parity, and speed | not-run |

## Sources and compatibility

- [D3D paper, v1](https://arxiv.org/html/2611.06425v1), section 3.3 and Appendix A.
- [Author implementation](https://github.com/fudan-generative-vision/D3D/blob/cd7762436217fb51d7e7a310b310b72766bb816f/training/pruning/losses.py), MIT.
- [Component usage](../research/d3d-corrected-kl.md).

The implementation adapts only the probability correction and KL computation.
It does not copy the author's distributed reduction, checkpointing, pruning
trainer, hidden-state loss, inference code, or model weights.

No quality, acceptance-length, memory, or throughput improvement is claimed.
Real model recovery and serving need their own model, data, runtime, and execution evidence.

## Execution evidence and remaining gates

The [execution receipt](../research/results/d3d-corrected-kl-20261111.json) records
the source, license, fixture hashes, commands, and environment.
The integrated research suite passed 46 tests, including 27 new component and CLI
checks. Existing research regressions passed 30 tests. Packaging passed three
tests, including wheel construction through the sdist. Applicable pre-commit
hooks and the isolated installed-wheel onboarding check passed.

The included fixture returned weighted forward KL approximately `0.0763915136152` and the
expected first target `[0.46, 0.36, 0.18]`. The optimizer check uses a small CPU
linear layer. It establishes a differentiable component, not full drafter recovery.

These local checks used Python 3.12.8, Torch 2.11.0, Transformers 5.8.1, and
Pydantic 2.13.4 on macOS arm64 CPU. The current CPU profile requires Transformers
5.12.1. Its known Torch download hash mismatch still blocks a fresh macOS setup.
The lock and download conditions are unchanged, so setup was not retried and no
hash check was bypassed. The new PR needs its own current Linux CI evidence.
Other PRs' successful gates do not verify this code.

After resolving the local artifact discrepancy, resume the full local gate with:

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-cpu.lock \
  --torch-backend cpu --require-hashes
uv run --no-project --python .venv/bin/python python -m tests.public_cpu
```

Full D3D recovery, pruning, export, and serving are not implemented by this
component. They require a separate model and trainer integration before their
real-runtime gates can run. No D3D strategy flag or executable end-to-end
reproduction is provided here. GPU work also needs an approved resource scope.
