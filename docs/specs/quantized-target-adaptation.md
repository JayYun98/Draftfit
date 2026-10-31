# Quantized-target draft adaptation

## Problem and scope

A pretrained speculative draft can lose alignment when the target is quantized.
Support weights-only draft adaptation and fresh draft training from quantized-target features.
Reuse current algorithms and trainer objectives. Do not introduce a model-specific calibration loss.

Integration base: `dev` at `bfe0bbe0dda0c0a1d2cca3ec93cdb1a963122145`.
The initial implementation used the older `main` base; its CPU evidence remains historical.
Current `dev` includes EAGLE4 and the existing explicit Prism/GGUF readout adapters.
This PR preserves those implementations and adds the shared target-quantization contract.
The shared target-loading and identity contract does not dispatch on algorithm or model names.
Actual capture layout and serving support remain bounded by each backend and algorithm.

Draft tensor quantization, packed output-head training, target fine-tuning, and new inference kernels are excluded.
No acceptance or speed improvement is claimed.

## Contract and implementation

- `model.target_quantization=auto` preserves serialized target quantization metadata.
- Explicit `bitsandbytes-4bit` and `bitsandbytes-8bit` recipes require Transformers and a floating source checkpoint.
- Capture loads packed modules directly on the chosen device. It never moves packed models after loading.
- Existing draft initialization uses `model.draft_checkpoint_path`; fresh initialization omits it.
- Keep current warm-start shape checks and full-resume incompatibility checks.
- Target embeddings and output heads must remain floating point in the same checkpoint.
- Store quantization identity in capture manifests, resume provenance, and exported draft metadata.
- Before offline model assembly, validate train and evaluation cache identity, target revision, backend, compute dtype, and attention settings.
- Keep dense default behavior and legacy dense caches compatible.

## Acceptance and evidence

| Requirement | Frozen acceptance | Evidence | Status |
| --- | --- | --- | --- |
| Model-independent preparation | Fresh and existing draft projects preserve recipe and initialization | Unit tests and actual local CLI preparation | pass; metadata scope |
| Actual quantized target | Real packed NF4/INT8 modules, frozen target, exact prequantized capture replay | `tests/gpu/quantized_target.py` | not-run; environment setup failed |
| Additional and fresh training | Finite nonzero gradients and changed draft weights for both modes | Same GPU smoke; random tiny weights | not-run; no quantized optimizer update |
| Artifact integrity | Reloaded draft tensor bytes and target quantization metadata match | CPU real export/reload test; GPU smoke | CPU pass; GPU not-run |
| Cache isolation | Reject missing, dense, different-recipe, revision, backend, dtype, or attention caches before model allocation | `tests/test_quantized_target.py` | pass; CPU |
| Resume isolation | Preserve quantization identity in the existing strict provenance check | Existing resume-contract tests and targeted test | CPU pass; GPU resume not-run |
| CPU and package regression | Public CPU suite and isolated wheel check pass | Verification record | pass on dev; 733 passed, 53 skips; packed regressions 34 pass |
| Serving and quality | Actual target-only/original/adapted serving comparison on held-out prompts | Compatible target engine required | not-run |

## Usage

See [quantized targets](../QUANTIZED_TARGETS.md) for preparation, capture, training, and validation commands.
The two initialization modes share the existing training CLI.

## Experiment plan

Bounded integration smoke: random tiny Llama and Qwen3, seed91, hidden128, vocab128,
three target layers, sequence24; NF4 and INT8; DSpark and DFlash2; fresh and weights-only initialization.
Run three optimizer updates per case, with identical source weights for each quantization recipe.
Require finite gradients, real weight changes, frozen teacher, packed replay, and exact draft serialization.
These synthetic inputs prove integration only. They cannot establish useful acceptance or speed.

Large-model quality, production trainer GPU resume, native serving correctness, and serving performance remain separate gates.
Do not weaken strict correctness or label missing GPU checks as passes.

## Results and limitations

Initial implementation and CPU verification passed on the older `main` base.
The PR now targets `dev`. Its CPU suite passed 733 cases with 53 skips.
Existing packed readout regressions passed 34 cases; isolated installed-wheel onboarding also passed.
Required GPU and serving gates remain open.
The worker failed before model construction because Transformers was missing.
The bounded dependency repair then failed with insufficient instance disk space.
No quantized training result, acceptance result, or speed result exists.
See the [verification record](../reports/quantized-target-adaptation.md) and its raw logs.
