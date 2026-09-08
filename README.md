# DSpark Train Platform

DSpark Train Platform is a configuration-first toolkit for adapting a
speculative draft model to a target model chosen by the user. It inspects target
metadata, generates an editable run/draft pair, prepares offline or online
features, trains or fine-tunes the draft, records provenance, and exports a
checkpoint for independent serving validation.

The private repository is `JayYun98/dspark-train-platform`.

The target stays frozen. The platform does not port target architectures into
the trainer and does not turn a recipe into a support claim. A generated
configuration is a starting point; capture, training, export, serving, and
state checks are separate gates.

## What is implemented

The live registry currently contains these draft methods:

| Method | Metadata-derived draft | Offline features | Online capture | Boundary |
| --- | --- | --- | --- | --- |
| DSpark | Yes | Yes | Yes | Requires target last-hidden features and auxiliary taps |
| DFlash | Yes | Yes | Yes | Requires algorithm-compatible hidden features and mask token |
| EAGLE3 | Yes | Yes | Yes | Requires the method's target features and vocabulary contract |
| PEagle | Yes | No | Yes | Streaming-only; flex attention is required |
| Domino | No | Yes | Yes | Supply a compatible draft JSON explicitly |

“Implemented” means code and configuration paths exist. It does not mean every
target, backend, GPU, sequence length, cache mode, export, or workload has been
validated.

The following are not public capabilities of this repository: DFlash2, DFly,
DFlare, MTP, and multimodal training. Changing `training.strategy` does not
convert an incompatible draft, feature cache, or target contract.

## Target boundary

The user configures the target with `model.target_model_path` and, when needed,
an exact `model.target_revision`. For local targets, use `--local-only`; for a
Hub target, pin the revision used for the weights, tokenizer, capture backend,
and serving check.

Online runs use the deployment configuration and a compatible target/capture
backend, which can be managed locally or externally.
The target's architecture and recurrent state remain the responsibility of
that backend.

`target inspect` is metadata-only: it reads configuration and tokenizer
metadata without downloading weights or executing remote modeling code. Its
recommendations are hypotheses. Verify tokenizer masks, embedding/head names,
feature taps, normalization, state semantics, and backend revision before
spending GPU time.

## Quick start

### CPU preparation

The locked profile is for onboarding and release checks, not inference or GPU
training. It is verified on Linux with Python 3.12 and uv 0.9.18:

```sh
uv venv --seed .venv-cpu
uv pip sync requirements-cpu.lock --python .venv-cpu/bin/python \
  --torch-backend cpu --require-hashes
.venv-cpu/bin/python -m tests.public_cpu
```

Current local gate: 218 tests ran, 211 passed and seven accelerator-only cases
were skipped on macOS with the existing CPU dependencies. A prior clean Linux
locked run passed 202 of 209 tests with seven skips.
A fresh macOS locked install remains blocked by the documented Torch wheel
hash mismatch; hash checking is not bypassed.

The lock and CPU tests are source-checkout assets. They are not a complete GPU
environment and are not copied into a serving image. After dependencies are
provisioned for the intended environment, build and inspect a local wheel:

```sh
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps dist/specforge-<version>-py3-none-any.whl
python -m specforge.assets list
python -I scripts/check_public_install.py
```

`--no-deps` does not make an arbitrary machine compatible or replace a pinned
GPU backend.

### Recipes without a training dependency

Recipe access uses the Python standard library and can run without Torch,
SGLang, a GPU, or a source checkout:

```sh
python -m specforge.assets list
python -m specforge.assets export ./my-draft-project
cd ./my-draft-project
```

Export requires a new destination and refuses to overwrite it. The exported
project preserves `configs/` and `examples/configs/` so its references remain
usable.

### Inspect and prepare a target

The command surface is:

```sh
specforge algorithms
specforge target inspect ORG/MODEL --revision EXACT_COMMIT
specforge target inspect /path/to/target --local-only
```

Generate an editable offline project from target metadata and pre-captured
features:

```sh
specforge target prepare /path/to/target --local-only \
  --strategy dspark \
  --hidden-states /path/to/features \
  --output-dir ./my-custom-draft \
  --set model.draft_num_hidden_layers=5
```

For managed online capture plus training, replace `--hidden-states` with raw
conversation JSONL:

```sh
specforge target prepare /path/to/target --local-only \
  --strategy dspark \
  --train-data /path/to/data.jsonl \
  --output-dir ./my-online-draft
```

The generated directory contains `train.json`, `draft.json`, `inspection.json`,
and a manifest. Preparation validates the target/data contract before writing
and refuses overwrites. Conversation JSONL uses a `conversations` array per
row; it is not a top-level `messages` document. This is structural validation,
not proof that tokenization masks or backend features are correct.

### Plan and run

Inspect the resolved process plan first:

```sh
specforge train -c ./my-custom-draft/train.json --plan
```

Then start a short smoke run in a new output directory before increasing the
budget:

```sh
specforge train -c examples/configs/qwen3-8b-dflash-offline.yaml \
  training.max_steps=5 training.save_interval=5 \
  output_dir=./outputs/first-smoke
```

Offline recipes require previously captured features. Online/disaggregated
recipes require a compatible capture server, transport, and deployment
configuration. `--plan` validates configuration and launch assembly; it does
not load real weights or prove a kernel, backend, or serving state path.

### Warm start versus resume

Use a draft export for a weights-only warm start:

```sh
specforge train -c ./my-custom-draft/train.json \
  model.draft_checkpoint_path=/path/to/draft-export
```

This starts a new optimizer, schedule, step counter, and data progression.

Use a training checkpoint to resume the training run:

```sh
specforge train -c ./my-custom-draft/train.json \
  training.resume_from=/path/to/training-checkpoint
```

This restores the checkpoint payload and its training state. Do not set both
options. Cross-world-size optimizer resharding and cold-host infrastructure
recovery are not supported claims.

### Export

Export the completed checkpoint with the exact draft configuration used to
train it:

```sh
specforge export --to hf \
  --checkpoint /path/to/completed-checkpoint \
  --draft-config /path/to/draft.json \
  --output-dir ./exports/my-draft
```

Some methods additionally require `--vocab-mapping`, `--embedding-source`, or
`--embedding-key`; check `specforge export --help`. A successful conversion is
not an inference compatibility result. Reload the exact artifact in its target
serving backend and compare target-only and speculative tokens/state on held-out
prompts.

## Validation commands

The dependency-light validation surface includes:

```sh
specforge validate parity --offline offline.json --online online.json
specforge validate tokens --expected target.json --actual speculative.json
specforge validate state --expected baseline.json --actual replay.json
specforge validate replay --expected baseline-state.json --actual replay-state.json
specforge validate acceptance --summary serving-summary.json
```

Functional correctness, acceptance length, quality, and speed are different
results. Report them separately. Do not infer quality or speedup from a passing
configuration, tensor comparison, or short smoke run.

## Current bounded evidence

| Target | What was actually tested |
| --- | --- |
| Ling-3.0-tiny | Real DSpark offline and online training, recovery/export, bounded serving/state checks |
| Llama-3.2-1B | Real-weight TP2 feature capture; not complete training/serving certification |
| LFM2.5-1.2B-Instruct | Real-weight CUDA load and chat smoke only |
| Qwen3-30B-A3B FP8 architecture | Dummy-weight MoE capture only |
| Tiny Llama fixture / reduced draft configs | Synthetic EAGLE3/DFlash capture and algorithm update/reload checks |

See the [model and training matrix](docs/MODEL_VALIDATION.md) for exact scope,
offline TP2 capture versus DP2 draft training, GPU0/GPU1 online training,
checkpoint steps, failures, and unknown revisions.

- Local CPU packaging and test gates pass in the current release snapshot;
  accelerator-only cases remain explicitly skipped on CPU.
- The pinned Ling native ReplaySSM route has a bounded H100 functional pass:
  67 committed state rounds matched bitwise and a separate three-request
  target-only comparison matched 144/144 generated token IDs. The route is
  single-request TP1/static, with Triton, radix cache disabled, CUDA graphs
  disabled, and float32 SSM state.
- The H100 result does not certify all GPUs, models, cache modes, concurrency,
  exported artifacts, quality, or speedup. The earlier SM120/RTX5090 native
  ReplaySSM state-parity failure remains unresolved and native replay is
  disabled there. Synthetic reload or token checks do not erase that failure.

For a new target or backend combination, record the target/tokenizer revision,
algorithm, draft configuration, backend revision, runtime, feature parity,
finite training updates, export reload, and serving/state results before calling
the combination supported.

## Packaging and publication status

The Python namespace and distribution name remain `specforge` for local
compatibility with the existing trainer and CLI. Publication is intentionally
disabled pending maintainer approval of the new product identity. CI is
configured for the private repository but currently manually disabled; no
package publication, deployment, or remote workflow is implied by this README.

## Credits and licenses

Upstream copyright headers and licenses are preserved. See [LICENSE](LICENSE)
and [THIRDPARTY_NOTICES.md](THIRDPARTY_NOTICES.md) for the source inventory,
borrowed boundaries, pinned references, and attribution requirements.

Read the detailed boundaries before running a target:

- [Public workflow](docs/PUBLIC_WORKFLOW.md)
- [Support matrix](docs/PUBLIC_SUPPORT.md)
- [CPU and release checks](docs/PUBLIC_RELEASE.md)
- [Native Ling replay](docs/NATIVE_LING_REPLAY.md)
