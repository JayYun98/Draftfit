# Public support boundaries

Config generation is not model certification. The target executes in the
existing inference backend; the platform does not port its architecture.
Use `draftfit algorithms` for the live registry and feature tensor contracts.

## Owned native teacher services

`draftfit target prepare ... --teacher-backend transformers` or
`--teacher-backend vllm` selects the platform's own teacher service. It publishes
features through the same Mooncake transport and producer/consumer trainer as
the SGLang path; it does not launch a TorchSpec/AngelSpec training wrapper.
Native capture is limited to supported single-device Llama/Qwen2/Qwen3 text
decoders, with real GPU evidence limited to the pinned Qwen3-0.6B campaign.

HF BF16 feature parity passed. Native vLLM BF16 cross-engine elementwise parity
failed; FP32 parity and deterministic next-step replay passed. Both managed
BF16 CLI runs completed two DSpark updates and a checkpoint, which establishes
functional execution but does not override the BF16 parity failure. No native
serving, throughput gain, long-run quality or broad architecture support is
certified. See [exact evidence](MODEL_VALIDATION.md#native-teacher-two-gpu-functional-evidence-2026-09-14)
and the [reproducible entrypoints](OWNED_GPU_ENVIRONMENT.md).

## Implemented draft methods

| Method | Metadata-derived draft | Offline features | Online capture | Important requirement |
| --- | --- | --- | --- | --- |
| DSpark | Yes | Yes | Yes | Target last-hidden features as well as auxiliary taps |
| DFlash | Yes | Yes | Yes | Algorithm-compatible hidden features and mask token |
| DFlash2 | Yes | Yes | Yes | Full vocabulary, grouped convolution/selector; eager/SDPA/flex; bounded CUDA training passed, live serving pending |
| EAGLE3 | Yes | Yes | Yes | Fixed one draft layer; shared vocab mapping for disaggregated online |
| PEagle | Yes | No | Yes | Flex attention; algorithm-specific layout and vocabulary contract |
| Domino | No: explicit draft JSON | Yes | Yes | Compatible draft architecture/config supplied by user |

“Yes” means an implementation exists, not that every target has passed.
See [model validation](MODEL_VALIDATION.md) for the bounded real-teacher DFlash2
training/resume/export evidence; it is not a live-serving or speedup result.

### DFlash2 serving runtime

DFlash2 uses grouped convolution and candidate-selector training; tune
`training.dflash2_selector_loss_alpha` and its draft configuration. It requires
the full target vocabulary and homogeneous full or sliding attention. USP
attention and vocabulary pruning are unsupported. HF export is for fine-tuning
reload; SGLang export requires `input_embedding_scale=1.0`.

Do not reuse the old Ling capture image as DFlash2 serving evidence: the locally
pinned Ling/SGLang sources do not contain `DFlash2DraftModel`. SGLang source
[`db272201`](https://github.com/sgl-project/sglang/blob/db272201a2dbd72e5699e443240a851f1313ad45/python/sglang/srt/models/dflash.py)
contains that architecture and candidate-selector support. This is a source
compatibility prerequisite, not a tested image or a validated deployment.
Pin the actual image digest and source revision before the GPU gate; verify
convolution/selector weight loading and exact tokens with the exported artifact.
The selector requires a dense FP16/BF16/FP32 target output head in this revision.

DFly, DFlare, MTP and multimodal training are not public capabilities
of this fork. Switching a strategy name does not convert weights or features.

## Architecture/backend evidence

| Target lane | Local coverage | Bounded real evidence | Still needed for new targets |
| --- | --- | --- | --- |
| Dense + KV | Metadata/config + tiny draft tests | Small dense EAGLE3/DFlash capture; bounded training | Exact export reload and complete per-method serving gate |
| MoE + KV | Metadata/config fixtures | Prior MoE capture used dummy weights | Real target capture/train/export/serve |
| MoE/linear/conv hybrid (Ling) | Config, renderer and contract tests | Pinned H200 bounded capture/replay; RTX5090 GPU0 target/GPU1 8-step fine-tune, new-export 24-token exact serving; H100 native replay67 state rounds and144 unhooked tokens ([pins and scope](NATIVE_LING_REPLAY.md)) | RTX5090 native ReplaySSM state parity FAILED (disabled); broader serving/quality and revision regression |
| Linear + KV (Qwen family) | Config/tap/state fixtures | No blanket family certification | Actual backend features and rollback/replay |
| Conv + KV (LFM family) | Config/state fixtures | LFM model-load/chat smoke only | Draft training and exact speculative state/serving |
| Recurrent (RWKV family) | Metadata/state fixtures | No real end-to-end evidence | A compatible backend first, then the full gate |

Prior GPU scope and pins are described in [runtime profiles](RUNTIME_PROFILES.md).
The user's separate Ling50K training is not repeated and is not certification of
other targets or this packaging change. Quality/speedup and functional correctness
are separate results. CUDA/OS/kernel profile generation is not hardware testing.

## Local gate versus release gate

`python -m tests.public_cpu` runs bounded CPU tests with HF network access and
CUDA visibility disabled. It checks configuration, parser masks, real tiny draft
construction/loading, checkpoint/export contracts and wheel assets. Seven
accelerator-dependent cases may skip. It does not run the whole GPU test suite.

Those seven named CUDA cases already passed in the retained H100 platform
regression: CLI config-to-training (1), checkpoint/resume continuity (2), and
HF/SGLang export/key checks (4). A CPU skip is not an outstanding GPU failure.
These are prior-version generic tests, not a GPU rerun of each release wheel or
certification of every target's exported draft.

For a newly supported combination record exact target/tokenizer revision,
algorithm and draft config, backend/patch revision, GPU/runtime, real feature
parity, finite updates, export reload and serving/state results. An external
server's revision must be independently checked; config pinning cannot prove it.
Full infrastructure-loss recovery and cross-world optimizer resharding remain
outside the current evidence. Do not advertise them as supported.

The RTX5090 FSDP2 auxiliary-head regression passed same-world two-rank synthetic
reload and resumed updates. This is not optimizer resharding or cold-host
recovery. PEagle's separate CUDA smoke does not certify its FULL_SHARD training
path. DFlash/DSpark/Domino synthetic CUDA update/reload checks do not substitute
for real target-family feature and serving gates. The pinned live dense capture
gate disables radix caching; cache-enabled capture is not certified.
