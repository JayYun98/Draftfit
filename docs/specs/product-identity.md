# DSpark product identity and responsibility boundaries

## Follow-up: Draftfit

Starting from `1c6dfff`, the accepted brand is Draftfit: **Your workload. Your draft.**
Use `draftfit` as the canonical Python package, distribution and CLI. Preserve
`speculative_train_platform`, `dspark` and `specforge` as import aliases and their
existing console commands. Algorithm identifiers and existing artifacts stay stable.
Current user-facing docs and unpublished blog/LinkedIn drafts adopt the brand;
historical evidence and upstream notices retain their original attribution.
No external publication, repository rename or claim of trademark availability.

Gates: namespace/pickle/CLI compatibility, CPU suite, wheel/sdist, installed-wheel
onboarding, lint, content link/claim review. CPU378run/8GPUskips, lint and content
metadata/relative-link review passed. Wheel/sdist built; isolated installed-wheel
onboarding passed for DSpark/DFlash/DFlash2. Seven articles remain drafts. Sphinx HTML builds with14 warnings
(9 pages absent from toctrees,4 external-source links,1 heading hierarchy).
GPU execution is not rerun for this branding change; prior failures remain failures.

## Follow-up: Speculative Train Platform

The requested canonical product name is now `speculative-train-platform`, not
DSpark. Starting at `622e244`, move implementation to `speculative_train_platform`,
rename the distribution and primary CLI, and retain both `dspark` and `specforge`
compatibility imports/commands. Preserve DSpark algorithm identifiers, model
auto_map contracts, checkpoint keys, environment variables and source notices.
Do not rename the GitHub repository or republish a package as part of this change.

Acceptance: canonical CLI/metadata and installed-wheel onboarding pass; all three
import paths share classes; old pickle and module execution work; public CPU gate
and pinned lint pass. Follow-up results: 378 CPU tests run, 370 passed and 8
accelerator-only skipped; pinned lint passed; wheel/sdist and isolated installed
wheel onboarding passed for DSpark/DFlash/DFlash2. Three namespaces share class
identity across all six pairwise import orders. Both old module/pickle paths pass.
Independent review found no remaining migration blocker. Tests used existing
macOS CPU dependencies, not a new locked Linux environment.
GPU validation is not run for this product-name-only change.
The prior results below remain evidence of the earlier revision, not this follow-up.

## Problem / scope

The platform owns its training composition but still exposes the upstream package
as its primary implementation namespace. Make `dspark` the implementation and
public Python entry point, retain a minimal legacy import bridge, and separate
independent training responsibilities. Do not rewrite algorithms for branding.
Baseline: `8126c96`. Preserve upstream copyright and notices.

## Contract / implementation

- `dspark` contains actual implementation, not wrappers over an installed SpecForge.
- Legacy `specforge` imports resolve to identical classes/modules where imported.
- Keep old CLI entry point and supported module execution paths.
- Keep checkpoint tensor keys, serialized config fields, wire formats, environment
  variables and algorithm identifiers stable. No model/loss changes.
- Separate metric reduction from the training lifecycle; no new framework or plugin layer.
- Bundle both package namespaces and existing recipes/notices in wheel and sdist.

## Acceptance and evidence

| Requirement | Gate | Status |
| --- | --- | --- |
| Native implementation namespace | CLI, imports, package inventory | pass |
| Legacy compatibility | Same class identity, old pickle globals, legacy CLI | pass; both import orders and three module entry points |
| Training responsibilities | Existing metric/step/checkpoint tests | pass; 81 run, 2 skips; extracted function ASTs unchanged |
| Distribution | public CPU suite, wheel build, isolated installed-wheel onboarding | pass; 378 run, 8 GPU skips; wheel/sdist and installed DSpark/DFlash/DFlash2 onboarding passed |
| Attribution | Original notices preserved; product metadata and docs consistent | pass; original source headers retained |

## Usage

Use `dspark algorithms` or `python -m dspark.cli algorithms` after installing
the platform wheel in the provisioned uv environment. Existing `specforge`
commands/imports remain compatibility entry points, not a second implementation.

## Results / limitations

No new target/backend support or speedup claim. Prior GPU results describe their
recorded revision. GPU launch after namespace migration is not yet run; CPU and
installed-package checks cannot certify CUDA execution. The existing vLLM BF16
parity failure remains unchanged. No learning or performance A/B is claimed.

Verification used the existing macOS Python 3.12.8 CPU dependencies with uv;
this is not a fresh hash-locked Linux installation. The wheel check uses a
separate environment and isolated Python (`-I`) to exclude source imports.
All applicable pinned pre-commit hooks passed. The change is predominantly
module relocation/import updates, not a new training algorithm.
