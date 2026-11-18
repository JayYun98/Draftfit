# Third-party implementation notes

Draftfit is derived from SpecForge and retains applicable upstream
licenses and copyright headers. Its locally maintained application, training and
runtime modules do not replace or erase the authorship of adapted source.

| Source | Pinned workspace revision | Used for |
|---|---|---|
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) | `bd64d93` | resource/checkpoint design reference and FSDP2 boundary |
| [z-lab/dflash](https://github.com/z-lab/dflash) | `07ebd93` | DFlash2 grouped convolution and candidate selector |
| [AngelSpec](https://github.com/Tencent/AngelSpec) | `d3412be` | packing/objective design reference |
| [SGLang](https://github.com/sgl-project/sglang) | `8ba213f` | pinned target serving and capture backend |
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | `4143092` | MLX inference reference, not a trainer dependency |
| [ALR-IRA](https://github.com/js-lee-AI/ALR-IRA) | `fa473642d48ea2667675d706ca8f16a74111183e` | Rollout/IRA preparation and sparse distillation |
| [Osprey](https://github.com/LeanModels/Osprey) | `f62cd48844b03f3cb929543da25c15756e1f1095` | Draft backbone conversion and aligned embeddings |
| [CAST](https://github.com/js-lee-AI/CAST) | `2eed73f38f9210f1d5ddf01de8059cf3ffb43c19` | Reference tree construction and corrected speculative sampling |
| [Redline](https://github.com/js-lee-AI/Redline) | `d78ede9b035420f3c8c8c63c185ed55be5ea038f` | Paired correctness-risk calibration |
| [TableTune](https://github.com/asm582/tabletune) | `38170878d901cb06f696a122a0b4140a3d365f02` | Offline draft-length table construction |
| [D3D](https://github.com/fudan-generative-vision/D3D) | `cd7762436217fb51d7e7a310b310b72766bb816f` | Corrected forward-KL component for offline recovery experiments |
| [DForkTree](https://github.com/Perseus037/DForkTree) | `eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3` | CPU reference for fork-aware candidate probability correction |
| Prism extended runtime | `e3de715222b2667c54a267441159a5729f11155e` | Explicit external native capture API; runtime is not bundled |
| [Multi-Future Drafting](https://github.com/hhhhhhy5/multi-future-drafting) | `95040343de00f15aef8c6350a578c35e0cd53475` | Algorithm reference for an independently written CPU best-mode prefix selector |
| [REAL loss reference](https://github.com/King-ty/speculators) | `3dff4579c5f93f692108df8b5178c9e9200b8536` | Target-sampled branch estimator and auxiliary teacher-argmax CE |

## Borrowed boundaries

- The MFD best-mode selector follows the paper's Section 4.3. The reviewed
  author HF tree reference has a separate MIT license beneath an Apache-2.0
  repository root. Both notices and source digests are retained in
  [the MFD source notice](draftfit/research/papers/mfd/NOTICE.md). No author runtime or model weights are bundled.

- D3D's probability correction and forward-KL calculation were adapted from
  `training/pruning/losses.py`. The full MIT notice is retained in
  `draftfit/research/papers/d3d/D3D_LICENSE`. No author trainer, pruning loop,
  hidden-state loss, model, inference engine, or weights are bundled.
- TorchSpec `torchspec/config/train_config.py`, `torchspec/ray/placement_group.py`,
  and `torchspec/training/checkpoint.py` were reviewed at `bd64d93`. Only the
  declarative launch/checkpoint contract was adapted in
  `draftfit/torchspec_bridge.py`; Ray actors and DCP state wrappers are not
  copied. The reference project's full MIT notice is bundled in
  `licenses/torchspec-LICENSE`.
- z-lab/dflash DFlash2 grouped-convolution, candidate-selector, configuration,
  and serving-state boundaries were reviewed at `07ebd93`. The local model and
  training wrapper adapt those boundaries to the platform's inherited Qwen3 DFlash
  seams; no z-lab runtime dependency is added. The source notice is bundled in
  `licenses/z-lab-dflash-LICENSE`.
- AngelSpec `angelspec/training/data_fetcher.py` and
  `angelspec/tests/test_packing_safety.py` were reviewed at `d3412be`. The
  strict validation ideas and anchor-policy behavior were adapted into the
  existing DSpark model/config; Mooncake packing/controller code is not copied.
  The reference project's full license and third-party exceptions are bundled
  in `licenses/angelspec-LICENSE`; this does not imply all its components are copied.
- mlx-dspark `src/mlx_dspark/config.py` and `src/mlx_dspark/server.py` were
  reviewed at `4143092` for serving-state separation only. No MLX code is
  imported by this platform.
- ALR/IRA, Osprey, CAST and Redline adaptations retain source headers and full
  MIT notices in `draftfit/research/NOTICE.md`. The local recipes reuse the
  maintained trainer and do not import external training launchers. Redline's
  binomial/Holm calculation uses log probabilities to avoid underflow.
- The optional packed MLX target adapter retains the Apple MIT notice in
  `draftfit/offline_capture/PRISM_RUNTIME_LICENSE`. The external Prism runtime,
  target weights and native build must be provisioned and reviewed separately;
  a bridge revision string does not authenticate a linked native library.

## Sync policy

The TableTune recipe adapts the dependency-light `dsd_tuning/core.py` numerical
kernel under Apache-2.0. Its full license is bundled in
`draftfit/research/papers/tabletune/TABLETUNE_LICENSE`.
The source boundary and digest are recorded in `draftfit/research/NOTICE.md`.
The vLLM plugin, live controller, installers, and profiling workers are not bundled.

The DForkTree reference implements paper equations 2 and 3 using supplied
conditional and unigram probabilities. The reviewed author sources are
`dforktree/ngram_gpu.py`, `dforktree/tree_ops.py`, and `dforktree/fc1_builder.py`.
The original Apache-2.0 license and notice are retained as
[DFORKTREE_LICENSE](draftfit/research/papers/dforktree/DFORKTREE_LICENSE) and
[DFORKTREE_NOTICE](draftfit/research/papers/dforktree/DFORKTREE_NOTICE).
The local component does not copy the SGLang worker, Flash Linear Attention
recurrence, GPU builder, or verifier. See `draftfit/research/NOTICE.md` for
the source identities and intentional numerical differences.

The latest research adapter additionally bundles the Apache-2.0 single-tree
TreeWY reference from `sneha5gsm/vllm` at
`b073ed6cacfa1cf5ae23111854729e662a72a6d1`. Its source header, exact scope
and original-file digest are recorded in `draftfit/research/NOTICE.md`;
`draftfit/research/papers/treewy/TREEWY_LICENSE` contains the full license.

Upstream code is copied only when a failing gate identifies a concrete
requirement. Preserve the upstream license header, record the source commit in
this table, add a focused contract test, and keep the local adapter boundary
independent from the upstream trainer loop. Do not merge duplicate controller,
Ray, or checkpoint implementations without a passing single-node gate first.


## REAL loss reference

The opt-in REAL component adapts the accepted-prefix estimator and loss
mathematics from King-ty/speculators at
`3dff4579c5f93f692108df8b5178c9e9200b8536`. The relevant source is
`src/speculators/models/dflash/metrics.py`. The paper is
https://arxiv.org/abs/2611.14399v1.

Source: https://github.com/King-ty/speculators/tree/3dff4579c5f93f692108df8b5178c9e9200b8536.
License: Apache-2.0, retained in `draftfit/research/papers/real/REAL_LICENSE`.
The pinned source tree has no root NOTICE file.

Local changes provide a standalone, proposal-only tensor boundary, explicit
validation, FP64 preservation, and a synthetic CPU research recipe. No author
training package, fused kernel, model weight, or serving pipeline is included.
This experimental component does not reproduce the paper's full training.
