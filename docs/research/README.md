# Research methods

[Platform workflow](../../README.md) | [Documentation index](../README.md)

These methods are experimental.
They are not production backends.
Use their separate recipes and execution limits.

## Existing research workflows

| Method | Purpose | Guide |
| --- | --- | --- |
| ALR/IRA | Prepare target rollouts for draft distillation | [Preparation and training](../specs/research-methods.md#alrira-preparation-and-training) |
| Osprey | Convert a compatible backbone into a draft model | [Conversion and training](../specs/research-methods.md#osprey-conversion-and-training) |
| CAST | Run reference tree decoding | [Tree decoding](../specs/research-methods.md#cast-reference-decoding) |
| Redline | Select from measured configurations | [Calibration](../specs/research-methods.md#redline-recipe) |

## Run a recipe

Copy a recipe from the [method guide](../specs/research-methods.md).
Check the plan before execution.

```sh
python -m draftfit.research -c recipe.json --plan
python -m draftfit.research -c recipe.json
```

The research schema is separate from the platform training schema.
The implementations live in [draftfit/research](../../draftfit/research/).

## Interpret results

Check correctness, answer quality, and latency separately.
A passing reference test does not establish production serving support.
Use the [platform support matrix](../PUBLIC_SUPPORT.md) for supported training and runtime combinations.
