# D3D corrected forward KL

This experiment provides the corrected distillation term from
[D3D section 3.3 and Appendix A](https://arxiv.org/html/2611.06425v1).
It is a reusable loss component and an offline recipe.
It does not register a D3D training strategy or serving backend.

## Run the included fixture

Use the [CPU environment](../ENVIRONMENT.md) from the repository root.
The fixture contains synthetic logits, supervised token IDs, and explicit weights.
It needs no model download, training data, or inference server.

```sh
python -m draftfit.research -c examples/configs/research/d3d-corrected-kl.yaml --plan
python -m draftfit.research -c examples/configs/research/d3d-corrected-kl.yaml
```

Plan mode writes nothing. Execution requires a new output directory and writes
`recipe.json` and `result.json`. For another execution, select a fresh directory:

```sh
python -m draftfit.research -c examples/configs/research/d3d-corrected-kl.yaml \
  output_dir=outputs/my-corrected-kl-check
```

For an installed package, export its assets into a new directory with
`python -m draftfit.assets export ./recipes` and run from that directory.

## Prepare inputs

The input file contains one JSON object per line. Each object has:

| Field | Meaning |
| --- | --- |
| `id` | Optional identifier for the result |
| `student_logits` | Floating logits shaped `[positions, vocabulary]` or `[batch, positions, vocabulary]` |
| `teacher_logits` | Frozen full-depth drafter logits with the same shape and vocabulary |
| `labels` | Integer target-response token IDs matching the leading dimensions |
| `weights` | Nonnegative weights with the same leading dimensions |

Set weights to zero for anchors, padding, or other positions outside the loss.
Only zero-weight positions may use the ignored label `-100`.
At least one weight must be positive. Shapes, scalar parameters, and numerical
values are checked before the objective runs.

Teacher and student must see the same block input and attention pattern.
Both must use the same vocabulary as the response labels.
The tensor helper cannot verify that provenance from logits alone.
The teacher is the original trained drafter, not the target verifier.
`input_kind` records the caller's description of the data. It does not
authenticate model provenance or a measurement.

Weights are explicit because each packed block must restart its position decay.
For a block of draft positions, the paper uses `exp(-offset / gamma)` with zero
offset at the first prediction. Multiply those weights by the valid-response mask.
The author recovery script uses `gamma=7.75` for block size 16 and `15.5` for size 32.
Do not apply one continuous decay across a flattened collection of blocks.

The fixture requests margin `0.1` to make the result easy to inspect.
Its first teacher distribution is `[0.1, 0.6, 0.3]`, and its label is token `0`.
The minimal mixture is `0.4`, giving `[0.46, 0.36, 0.18]`.
The second teacher already meets the margin and stays unchanged.
The last position has zero weight and contributes no gradient.

## Reuse the loss

```python
from draftfit.research.d3d import corrected_kl_loss

loss = corrected_kl_loss(
    student_logits, teacher_logits, labels, weights,
    temperature=1.0, probability_margin=0.01,
)
loss.backward()
```

The component computes `KL(corrected_teacher || student)` and normalizes by the
actual weight sum. Temperature scaling applies to both distributions, and the
loss is multiplied by `temperature ** 2`. The teacher is detached.
Half-precision inputs use float32 arithmetic; float64 inputs retain float64.

The [acceptance contract](../specs/d3d-corrected-kl.md) defines the numerical and
gradient boundaries. The included CPU tests check a real optimizer update.
That diagnostic does not establish recovery quality for a trained drafter.

## Integration limits

This recipe evaluates only the corrected KL component. D3D's complete recovery
objective also needs diffusion cross-entropy and autoregressive supervision.
The pinned author code additionally contains its own hidden-state recovery term.
None of those training loops are installed by this recipe.

Layer selection, full drafter recovery, checkpoint export, asynchronous execution,
CUDA graphs, token parity, and serving speed need separate integration and tests.
No quality or performance improvement is claimed.

The implementation adapts
[the MIT author source](https://github.com/fudan-generative-vision/D3D/blob/cd7762436217fb51d7e7a310b310b72766bb816f/training/pruning/losses.py).
The original license is retained in `draftfit/research/_vendor/D3D_LICENSE`.
