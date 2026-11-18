# Research methods

[Platform workflow](../../README.md) | [Documentation index](../README.md)

These methods are experimental.
They are not production backends.
Use their separate recipes and execution limits.

## Find an experiment

| Start here | What you will find |
| --- | --- |
| [Paper implementations](../../draftfit/research/papers/README.md) | Each paper directory links its recipe or configuration, implementation, tests, and source. |
| [Experimental platform features](experimental-platform.md) | Opt-in capture, training, and evaluation paths, including features without a paper. Each guide names the runtime that owns the code. |
| [Recipe examples](../../examples/configs/research/README.md) | Inputs and CLI examples for the shared research runner. |
| [Paper coverage](latest-speculative-methods.md) | Methods implemented, measurement-only contributions, and papers with no executable integration. |

When adding a standalone experiment, use `draftfit/research/papers/<name>/` for a paper
or `draftfit/research/experiments/<name>/` for a platform idea. Put its implementation
and source notices beside its README. Link the opt-in command, fixture, tests, and known limits.
For an experiment in an existing trainer or backend, link that owner and its configuration from the feature folder.

## Paper intake and comparisons

[Paper intake](paper-intake.md) tracks paper revisions and prior decisions.
The [recipe list](../../examples/configs/research/README.md) includes TableTune, D3D, DForkTree, MFD and REAL.
The [100-step comparison](results/paper-components-small100-20261113.md) records the tested component revisions.
Admission to the research suite does not establish production training, serving parity, or speed.

## Existing research workflows

| Method | Purpose | Guide |
| --- | --- | --- |
| ALR/IRA | Prepare target rollouts for draft distillation | [Preparation and training](../specs/research-methods.md#alrira-preparation-and-training) |
| Osprey | Convert a compatible backbone into a draft model | [Conversion and training](../specs/research-methods.md#osprey-conversion-and-training) |
| CAST | Run reference tree decoding | [Tree decoding](../specs/research-methods.md#cast-reference-decoding) |
| Redline | Select from measured configurations | [Calibration](../specs/research-methods.md#redline-recipe) |

## Tested reference experiments

All eight research recipes passed CPU CLI checks.
All 52 affected regression and packaging tests passed.
GPU checks used a pinned Qwen3-0.6B model on an A100.
These checks cover reference implementations.

A target call is one forward pass.
The baseline runs the target model without a draft model.

| Algorithm | Observed result |
| --- | --- |
| [NgramRL](../../examples/configs/research/ngramrl.yaml) | One repeat case used 2 target calls instead of 14. Median time fell from 452.5 to 82.3 ms. |
| [SpecMem](../../examples/configs/research/specmem.yaml) | One repeat case used 1 target call instead of 14. Median time fell from 452.3 to 51.2 ms. |
| [ValuSpec + CAST](../../examples/configs/research/valuspec-cast.yaml) | ValuSpec rejected all seven candidates from an untrained DFlash in each round. Target fallback matched the baseline. No speed gain was demonstrated. |
| [RIFT](../../examples/configs/research/rift.yaml) | A 16-token budget reduced the active cache from 4.82–9.18 MB to 3.67 MB. Generation was 1.23–1.83 times slower. Peak allocation did not fall. |
| [ARR](../../examples/configs/research/arr.yaml) | The first `1+1` answer was wrong. Both systems later found four correct numbers. Both failed integer-only output (0/4). ARR did not improve answer quality. |
| [TreeWY](../../examples/configs/research/treewy.yaml) | CPU/CUDA comparisons gave a maximum state error of 5.96e-8. Serving performance remains untested. |
| [MoE-SpAc](../../examples/configs/research/moe-spac.yaml) | CPU policy tests passed. Expert execution and offload remain untested. |
| [Query layouts (PQ/QB)](../../examples/configs/research/query-layout.yaml) | CPU/CUDA attention comparisons passed. GPU output error was 0. Native GGML execution remains untested. |

Memory proposals used history from an earlier request.
Neither memory method reduced target calls for cold requests, new text, or changed numbers.
SpecMem also reduced calls from 10 to 8 for one paraphrase.
Its median time fell from 339.7 to 269.3 ms.
SpecMem used lexical embeddings.

RIFT used another copy of the target model as its draft model.
It rebuilt the full history each round.

Each timing condition used one warmup and three measured comparisons.
All 51 measured pairs produced the exact baseline tokens.
Times include resident model execution and policy work.
They exclude model loading, tokenization, and history preparation.
Shared GPU activity limits these measurements.
Production speed and general workload benefits remain unverified.

See the [test protocol and limits](latest-speculative-methods.md#non-oracle-effect-follow-up) and [recorded measurements](results/latest-methods-effects-20261028.json).

## Run a recipe

For platform draft training, see the separate
[MiniCPM DSpark/DFlash2/EAGLE4 comparison](eagle4-minicpm.md).
It records full dense training, candidate distillation and bounded serving gates.

Select a [research recipe](../../examples/configs/research/README.md).
Model recipes require compatible local checkpoints.
Numerical recipes use the included fixtures.
Check the plan before execution.

```sh
python -m draftfit.research -c examples/configs/research/treewy.yaml --plan
python -m draftfit.research -c examples/configs/research/treewy.yaml
```

The research schema is separate from the platform training schema.
Use `draftfit research` to run these recipes.

## Files

| Content | Location |
| --- | --- |
| Implementations | [draftfit/research](../../draftfit/research/) |
| Recipes and fixtures | [examples/configs/research](../../examples/configs/research/README.md) |
| Experiment drivers | [scripts/research](../../scripts/research/README.md) |
| Reference regression tests | [Research tests](../../draftfit/research/tests) |
| Protocol and coverage | [Latest method specification](latest-speculative-methods.md) |
| Recorded measurements | [Public receipt](results/latest-methods-effects-20261028.json) |

## Verify changes

Run the research CPU suite from a source checkout in the [CPU environment](../ENVIRONMENT.md):

```sh
python -m draftfit.research.tests.cpu
```

It runs the paper and experiment tests in [draftfit/research/tests](../../draftfit/research/tests/).
The regular public CPU gate covers supported platform contracts and runs separately.
For a built and installed wheel, run `python -I draftfit/research/tests/check_install.py` from the checkout.

Run `python -m draftfit.research.tests.cuda` in a provisioned CUDA environment for the device and asynchronous capture gates.
In the external MLX environment, run the [lookup budget cases](../../draftfit/research/tests/accelerator/test_mlx_lookup_budget.py)
with `PYTHONPATH="$MLX_RUNTIME/src" python -m pytest draftfit/research/tests/accelerator/test_mlx_lookup_budget.py`.
For an end-to-end CPU fixture with tiny local models, use a new output directory:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/research/run_latest_smoke.py \
  --output outputs/new-research-smoke
```

## Interpret results

Check correctness, answer quality, and latency separately.
A passing reference test does not establish production serving support.
Use the [platform support matrix](../PUBLIC_SUPPORT.md) for supported training and runtime combinations.
