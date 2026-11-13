# Inspect REAL accepted-prefix loss

Use the opt-in `real_loss` research recipe to evaluate a small proposal-only
tensor component. This does not select a production training strategy or start
a model server.

## Run the fixture

Use the project's documented contributor Python environment. From the repository
root, run:

```sh
python -m draftfit.research -c examples/configs/research/real-loss.yaml --plan
python -m draftfit.research -c examples/configs/research/real-loss.yaml
```

The plan validates the recipe without reading the fixture or creating output.
Execution reads synthetic logits and sampled token IDs. It evaluates the loss,
runs a student-logit backward pass, and records the input identity and numerical
scope. Use a fresh output directory for every execution. The runner writes
`recipe.json` and `result.json` through its existing persistence path.

For an installed wheel, run `python -m draftfit.assets export ./recipes`.
Then run the recipe commands from `./recipes`.

## Use the tensor component

```python
from draftfit.research.real import real_loss

result = real_loss(
    draft_logits,
    target_logits,
    target_token_ids,
    prediction_mask=prediction_mask,
    real_weight=0.9,
)
result.loss.backward()
```

Logits use `[blocks, gamma, vocabulary]` shape. IDs are `torch.long`, and masks
are `torch.bool` with `[blocks, gamma]` shape on the logits device. The result
separates `real_loss`, `ce_loss`, and the mixed `loss`. It also exposes per-block
`acceptance_estimate` and per-position branch terms for inspection.

## Supply aligned proposal blocks

Each block must contain only proposed positions. Exclude the anchor and bonus
tokens before calling the component. Split document and packing boundaries into
separate blocks. Draft and target tensors must use the same token vocabulary,
conditional prefixes, and sampling distribution.

The Boolean mask ends each block at its first inactive position. It also removes
every later position in that block. Callers must construct EOS and padding masks
correctly. Token ID `-100` is permitted only at inactive positions.

The target-sampled estimator uses `min(1, q(y)/p(y))` on earlier positions and
the full vocabulary overlap at the current position. Both factors retain draft
gradients. The fixed proposal width normalizes the EAL term. The default mixed
loss adds a 0.1 CE contribution to the 0.9 EAL contribution. CE uses the target
logits' argmax as a hard label, following the pinned author's CE callable.
The sampled token IDs select the path weights. They do not select the CE label.
This component does not apply the full training wrapper's position decay.

Target tensors are detached. Half precision inputs use FP32 target/loss
arithmetic; FP64 is preserved. Tiny prefix-survival contributions can underflow.
Finite-precision evaluation does not establish exact statistical unbiasedness.

## Interpret the result

The component cannot establish that user-supplied trajectories came from the
frozen target. Its unbiasedness assumptions do not cover arbitrary off-policy
data, D-REAL's draft-sampled rollouts, or an arbitrary tree verifier. The fixture
checks tensor mathematics only.

A small optimizer update can verify differentiability. It does not establish
drafter quality, real acceptance length, or speed. No author checkpoint, target
weights, model download, rollout generator, or internal trainer is bundled.

## Verify changes and resume remaining gates

```sh
python -m unittest tests.research.test_real_loss -v
python -m unittest discover -s tests/research -t . -v
python -m unittest tests.test_packaging.test_assets -v
```

The [specification](../specs/real-loss.md) maps the numerical and integration
contract to observed checks. A matching hash-locked environment must run
`python -m tests.public_cpu` and the installed-wheel probe for the exact revision.
The known macOS Torch artifact discrepancy remains a setup blocker.

CUDA parity requires a compatible GPU environment and a separate resource
authorization. Real-model adoption additionally requires target-sampled data,
short training and resume checks, a controlled held-out comparison, export and
reload, and actual backend correctness and speed measurements. No result from
the separate D3D or DForkTree GPU checks validates this component.

## Attribution

The source reference is
[King-ty/speculators](https://github.com/King-ty/speculators/tree/3dff4579c5f93f692108df8b5178c9e9200b8536),
commit `3dff4579c5f93f692108df8b5178c9e9200b8536`, under Apache-2.0.
The exact license is packaged as `draftfit/research/_vendor/REAL_LICENSE`.
Local changes and source scope are listed in `draftfit/research/NOTICE.md`.
