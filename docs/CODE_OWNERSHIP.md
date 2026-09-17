# Code ownership and runtime boundaries

“Owned” here means code maintained and shipped by this repository, not exclusive
copyright ownership. This platform is a maintained derivative of SpecForge;
upstream authorship and applicable notices remain in place. See
[SOURCE_ATTRIBUTION.md](SOURCE_ATTRIBUTION.md) for source references and licenses.

| Component | What this repository maintains | External boundary |
| --- | --- | --- |
| Speculative Train Platform | In-tree `speculative_train_platform/`: CLI/configuration, target metadata, draft algorithms, training, checkpoint/export, capture adapters and runtime orchestration; shipped in the platform wheel | PyTorch, Transformers and other installed libraries remain dependencies. `specforge/` is an import compatibility bridge, not an upstream runtime installation. |
| SGLang | In-tree adapters under `speculative_train_platform/offline_capture/sglang_backend/`, patch installers and versioned patch files under `patches/sglang/` | Actual `sglang` implementation is installed separately. The validated Ling route uses a pinned external image/source plus our capture patch. It is a customized backend installation, not SGLang source bundled into the platform wheel. |
| Mooncake | `speculative_train_platform/runtime/data_plane/mooncake_store.py` and managed process/config integration | Imports the installed official `mooncake.store` API. This tree does not vendor or patch Mooncake itself; install the official CUDA-compatible wheel/master executable required by the runtime. |
| TorchSpec references | `speculative_train_platform/torchspec_bridge.py` is a small stdlib-only launch/checkpoint planner; DFlash2 adapted model/loss/export code is maintained in-tree with source notices | The bridge emits the existing SpecForge `torchrun` plan. It does not import TorchSpec, run its distributed trainer, or require its repository/Ray runtime. |

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
algorithm providers own model/loss/feature contracts. The canonical `speculative_train_platform`
package exposes these responsibilities directly. Legacy imports resolve to
the same implementation objects, without a second algorithm registry or trainer.

## Module responsibilities

| Module | Responsibility | Does not own |
| --- | --- | --- |
| `speculative_train_platform.cli` | Parse commands and call application entry points | Training mathematics or backend extraction |
| `speculative_train_platform.application` | Resolve config, validate capabilities, compose a run | Model forward passes or storage protocols |
| `speculative_train_platform.algorithms` | Draft models, objectives and feature contracts | Process supervision |
| `speculative_train_platform.training` | Training lifecycle, metric reduction, checkpoint and resume | Teacher architecture implementations |
| `speculative_train_platform.inference` / `offline_capture` | Teacher adapters and captured feature contracts | Draft optimization |
| `speculative_train_platform.runtime` | Worker supervision, control flow and feature transport | Algorithm-specific loss |
| `speculative_train_platform.export` | Artifact conversion and loader metadata | Serving performance certification |
| `dspark` / `specforge` | Compatibility imports and legacy module execution | Independent implementation |

Compatibility retains existing environment variables, serialized field names,
checkpoint tensor keys and wire contracts. Product naming is not a reason to
invalidate stored artifacts or change runtime safety controls.

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
