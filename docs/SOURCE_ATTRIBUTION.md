# Source inventory (not a legal clearance certificate)

Retain the root LICENSE and per-file headers. The upstream root is MIT but
some source files, including the CLI, carry Apache-2.0 headers; do not erase or
relabel them under a single new blanket notice.

| Source | Recorded reference | Observed notice | Local use |
| --- | --- | --- | --- |
| SpecForge / sgl-project | Initial normalization base `f9f8c3b`; current checkout contains later commits | Root MIT, copyright 2025 sgl-project; mixed per-file headers | Existing trainer, draft methods, configs, export and runtime |
| TorchSpec / lightseekorg | `bd64d93674ad304e52bd7d0c3027c22e822fc19d` | MIT, copyright 2026 LightSeek Foundation | Placement/checkpoint design reference and adapted DFlash2 model/loss/export; no Ray runtime dependency |
| z-lab/dflash | `07ebd93db9f472af339b644bb70221ad8428328a` | MIT; full notice in `licenses/z-lab-dflash-LICENSE` | DFlash2 convolution and selector implementation, via the pinned TorchSpec adaptation |
| AngelSpec / Tencent | `d3412bed231025ea85d35dbf5e0af44f1ac62a5b` | Apache-2.0 with third-party exceptions in its LICENSE | Anchor/FSDP design references, not wholesale trainer import |
| mlx-dspark / ARahim3 | `4143092` | MIT, copyright 2026 erahim3 | Inference/renderer reference, not packaged distributed trainer |
| SGLang Ling backend | `8ba213fc67550aea80dfbd5ef66f4574ec25e653` | Preserve upstream and patch headers | Separate pinned backend patch, not bundled in the Python wheel |
| ALR-IRA / js-lee-AI | `fa473642d48ea2667675d706ca8f16a74111183e` | MIT, copyright 2026 Jungseob Lee | Explicit rollout preparation and sparse KD adapter |
| Osprey / LeanModels | `f62cd48844b03f3cb929543da25c15756e1f1095` | MIT, copyright 2025 sgl-project | Qwen3/Llama draft conversion and model adapters |
| CAST / js-lee-AI | `2eed73f38f9210f1d5ddf01de8059cf3ffb43c19` | MIT, copyright 2026 Jungseob Lee | Batch-one reference tree decoding |
| Redline / js-lee-AI | `d78ede9b035420f3c8c8c63c185ed55be5ea038f` | MIT, copyright 2026 Jungseob Lee | Paired correctness-risk selection kernel |
| MLX packed target adapter | Local capture port; native Prism API reference `e3de715222b2667c54a267441159a5729f11155e` | Apple MIT notice retained, copyright 2023 Apple Inc. | Optional packed target capture/readout; external native runtime not redistributed |

Full selected research notices ship in `draftfit/research/NOTICE.md`; the packed
adapter notice ships in `draftfit/offline_capture/PRISM_RUNTIME_LICENSE`. Native
Prism builds require separate dependency and patch provenance as described in
[quantized capture](specs/quantized-capture.md).

This inventory records the prior implementation ledger and local license files.
It does not assert that every upstream line or transitive dependency has been
audited. Before publication, inventory actual copied/modified files against
their source revision, include applicable full notices, and document exceptions.
Target model weights, tokenizer assets and datasets each have their own terms;
package licensing does not grant permission to redistribute those assets.

## Bounded release audit (2026-09-08)

This is a source/package hygiene check, not a legal clearance review. A local
`specforge-0.2.0-py3-none-any.whl` build included the root `LICENSE` and this
inventory, but did not include the existing root `THIRDPARTY_NOTICES.md`.
Remediation: that notice and the verbatim TorchSpec/AngelSpec license files now
ship in the release assets; their references no longer require sibling checkouts.

The same wheel included
`examples/configs/kimi-k3-dspark-disaggregated.yaml`, which at that time
contained deployment-local paths, hostnames, and run/checkpoint identifiers.
Remediation: the example now uses explicit placeholder paths and `.invalid`
hostnames, generic run IDs, and disabled tracking. Its architecture/options
remain examples, not a new real-model support claim.
At that audit, `VAST_HANDOFF.md` was a tracked operational handoff, not a release
notice. It is absent from the consolidated 0.1.0 main tree. Earlier history is
retained in local backups and the refactor branch; remote backup refs were later
removed. Review all refs that will become public, not only the current
tree. Removing a file from the current tree does not remove historical copies.

A bounded scan of the current non-test source found no obvious private-key or
credential values; secret-shaped matches were environment-variable names,
redaction tests, or documentation examples. This does not replace a complete
secret scanner or history review. Do not treat this result as evidence that
publication is safe.
