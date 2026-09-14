# Code ownership and runtime boundaries

“Owned” here means code maintained and shipped by this repository, not exclusive
copyright ownership. This platform is a maintained derivative of SpecForge;
upstream authorship and applicable notices remain in place. See
[SOURCE_ATTRIBUTION.md](SOURCE_ATTRIBUTION.md) for source references and licenses.

| Component | What this repository maintains | External boundary |
| --- | --- | --- |
| SpecForge-derived platform | In-tree `specforge/`: CLI/configuration, target metadata, draft algorithms, training, checkpoint/export, capture adapters and runtime orchestration; shipped in the platform wheel | PyTorch, Transformers and other installed libraries remain dependencies. The retained `specforge` namespace does not mean an upstream checkout is required at runtime. |
| SGLang | In-tree adapters under `specforge/offline_capture/sglang_backend/`, patch installers and versioned patch files under `patches/sglang/` | Actual `sglang` implementation is installed separately. The validated Ling route uses a pinned external image/source plus our capture patch. It is a customized backend installation, not SGLang source bundled into the platform wheel. |
| Mooncake | `specforge/runtime/data_plane/mooncake_store.py` and managed process/config integration | Imports the installed official `mooncake.store` API. This tree does not vendor or patch Mooncake itself; install the official CUDA-compatible wheel/master executable required by the runtime. |
| TorchSpec references | `specforge/torchspec_bridge.py` is a small stdlib-only launch/checkpoint planner; DFlash2 adapted model/loss/export code is maintained in-tree with source notices | The bridge emits the existing SpecForge `torchrun` plan. It does not import TorchSpec, run its distributed trainer, or require its repository/Ray runtime. |

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
