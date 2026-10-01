# Code ownership and runtime boundaries

“Owned” here means code maintained and shipped by this repository, not exclusive
copyright ownership. This platform is a maintained derivative of SpecForge;
upstream authorship and applicable notices remain in place. See
[SOURCE_ATTRIBUTION.md](SOURCE_ATTRIBUTION.md) for source references and licenses.

| Component | What this repository maintains | External boundary |
| --- | --- | --- |
| Draftfit | In-tree `draftfit/`: CLI/configuration, target metadata, draft algorithms, training, checkpoint/export, capture adapters and runtime orchestration; shipped in the platform wheel | PyTorch, Transformers and other installed libraries remain dependencies. No legacy namespace bridges or separate upstream trainer installation. |
| SGLang | In-tree adapters under `draftfit/offline_capture/sglang_backend/`, patch installers and versioned patch files under `patches/sglang/` | Actual `sglang` implementation is installed separately. The validated Ling route uses a pinned external image/source plus our capture patch. It is a customized backend installation, not SGLang source bundled into the platform wheel. |
| Mooncake | `draftfit/runtime/data_plane/mooncake_store.py` and managed process/config integration | Imports the installed official `mooncake.store` API. This tree does not vendor or patch Mooncake itself; install the official CUDA-compatible wheel/master executable required by the runtime. |
| TorchSpec references | `draftfit/torchspec_bridge.py` is a small stdlib-only launch/checkpoint planner; DFlash2 adapted model/loss/export code is maintained in-tree with source notices | The bridge emits the existing SpecForge `torchrun` plan. It does not import TorchSpec, run its distributed trainer, or require its repository/Ray runtime. |

The bridge's `executor` is `specforge_torchrun`; its name describes the referenced
resource/checkpoint contract, not delegation to another installed framework.
Likewise, the Mooncake wrapper is our code but its transport implementation is
external. The SGLang capture adapters import backend internals, so compatible
source revisions and patches matter beyond the Python package name alone.

The platform wheel contains the platform and explicitly bundled recipes/guides.
The separate SGLang patch/build recipe must be available when provisioning its
custom runtime. Installing the wheel does not itself build that runtime, install
model weights, or port an unsupported target architecture. A new target can use
configuration extensions when its existing backend and feature/state contracts
already suffice; otherwise backend or platform implementation work is required.

This boundary was checked against the in-tree bridge, CLI caller, Mooncake
wrapper, SGLang adapter imports, patch README/Dockerfile and package discovery.
It does not claim that every third-party source line has received legal review.

## Owned online teachers

The platform is an in-tree derivative with its own composition, not a CLI over
TorchSpec/AngelSpec. `training/disaggregated.py` assembles our producer/channel/
consumer; `runtime/` owns task leases, backpressure, acknowledgments and cleanup;
algorithm providers own model/loss/feature contracts. The canonical `draftfit`
package exposes these responsibilities directly. Removed legacy namespaces are
not import aliases; use `draftfit` for imports and command execution.

## Module responsibilities

| Module | Responsibility | Does not own |
| --- | --- | --- |
| `draftfit.cli` | Parse commands and call application entry points | Training mathematics or backend extraction |
| `draftfit.application` | Resolve config, validate capabilities, compose a run | Model forward passes or storage protocols |
| `draftfit.algorithms` | Draft models, objectives and feature contracts | Process supervision |
| `draftfit.training` | Training lifecycle, metric reduction, checkpoint and resume | Teacher architecture implementations |
| `draftfit.inference` / `offline_capture` | Teacher adapters and captured feature contracts | Draft optimization |
| `draftfit.runtime` | Worker supervision, control flow and feature transport | Algorithm-specific loss |
| `draftfit.export` | Artifact conversion and loader metadata | Serving performance certification |

Compatibility retains existing environment variables, serialized field names,
checkpoint tensor keys and wire contracts. Product naming is not a reason to
invalidate stored artifacts or change runtime safety controls.

## Repository levels

`draftfit/` is the installable Python product. Root-level directories contain
source inputs, contributor tools or documentation; they are not additional
installed training frameworks.

| Location | Responsibility | Installation boundary |
| --- | --- | --- |
| `draftfit/config/` | Configuration schemas, validation and resolution | Python package |
| `configs/` | Target/draft recipe source files | Selected files bundled at build time |
| `examples/` | Runnable usage examples and training configurations | Selected configs bundled; scripts remain checkout tools |
| `draftfit/data/` | Data preparation, rendering and masks | Python package |
| `draftfit/data/` | Dataset preparation, regeneration and validation | Installed package and CLI |
| `draftfit/training/evaluation.py` | Training-time validation metric aggregation | Python package |
| `draftfit/benchmarks/` | Installed inference benchmark/report functionality | Python package |
| `benchmarks/` | Source-only benchmark harnesses and trainer experiments | Source checkout only |
| `draftfit/assets/` | Access/export API for bundled recipes and guides | Python package; `_data` generated during build |
| `assets/` | Website/README visual assets | Not model weights or packaged recipes |
| `tests/integration/` and `tests/packaging/` | Opt-in runtime gates and installed-wheel checks | Source checkout only |
| `tests/` | Automated checks | Development only |
| `patches/` | Versioned external-backend patches and image recipes | Provision separately |
| `docs/`, `plans/`, `.skill/` | Guides, roadmap and agent workflows | Documentation; selected guides bundled |
| `.private/`, `.planning/` | Local archives and working records | Git-ignored; not public package contents |

Recipes have one source of truth: root `configs/` and `examples/configs/`.
`setup.py` copies them into the wheel; do not maintain a second hand-edited copy
inside `draftfit/assets/`. Keep runtime imports independent of checkout-only
tools. Packaging files, dependency locks, licenses and the README intentionally
remain at the root. New standalone performance probes belong in `benchmarks/`;
user-facing evaluation functionality belongs in the installed package.

The backend-specific producer constraint has been removed: `teacher_server.py`
runs our HF/vLLM extraction services, `capture_sink.py` maintains our existing
SGLang-patch wire format, and `TeacherServerCaptureAdapter` connects all three
to the same trainer. TorchSpec `bd64d936` and AngelSpec `d3412bed` were examined
for engine separation and feature delivery; neither becomes an installed training
runtime. The sink is adapted from our versioned SGLang patch with its source
headers retained. HF/vLLM remain actual inference dependencies, as PyTorch remains
the tensor/training dependency. This is source integration, not a claim of having
written those underlying engines or inherited all upstream support guarantees.

The first native service scope is single-host, TP1, Llama/Qwen2/Qwen3 text capture.
vLLM's intermediate extraction files remain a performance limit. Real GPU parity,
long-running recovery and supported serving combinations must be validated before
promoting either new online backend to first-tier production support.
