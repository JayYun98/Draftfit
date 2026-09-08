# Model validation boundary

**Status:** bounded functional evidence as of 2026-09-08. This is not blanket
model certification, a quality benchmark, or a speedup claim. Configuration
generation is not model validation; see [public support boundaries](PUBLIC_SUPPORT.md)
and the [public workflow](PUBLIC_WORKFLOW.md).

## What is actually pinned

| Campaign | Target and revision | Backend/runtime pin | Evidence class |
| --- | --- | --- | --- |
| H200 Ling functional run | `inclusionAI/Ling-3.0-tiny@a2ee06c0f2de5b171701aee7f73f70a1da75483b` (download metadata) | SGLang `8ba213fc67550aea80dfbd5ef66f4574ec25e653`; Ling `BailingMoeV3ForCausalLM`, 24 layers, hidden size 1536 | Real target and draft weights |
| H100 native replay runs | `inclusionAI/Ling-3.0-tiny@e3a47d5b986e7141b6efd62597d598ebb392060d` | SGLang `8ba213fc67550aea80dfbd5ef66f4574ec25e653`; Triton, TP1, radix off, graphs off | Real target and draft weights |
| RTX 5090 public run | `inclusionAI/Ling-3.0-tiny@e3a47d5b986e7141b6efd62597d598ebb392060d` | Torch `2.13.0+cu129`, CUDA 12.9, Triton 3.7.1, same SGLang source | Real target and draft weights |
| Dense live capture fixture | Locally generated 8-layer `LlamaForCausalLM`; no Hub revision | Hidden size 64, vocab 256, seed 1234 | Synthetic random weights |
| Dense TP2 capture | `Llama-3.2-1B`; revision not recorded in the retained report | H100 TP2 offline capture | Real target weights; revision unknown |
| MoE TP2 capture | `Qwen3-30B-A3B-Instruct-2507-FP8`; revision not recorded | H100 TP2 offline capture | Architecture test with dummy weights |
| Hybrid-family load smoke | `LiquidAI/LFM2.5-1.2B-Instruct`; revision not recorded | H100 CUDA load/chat smoke | Real target weights; no draft training or serving gate |

The production DSpark draft used for Ling replay is a normalized
`Qwen3DSparkModel` with three draft layers, target taps `[3, 7, 11, 15, 19]`,
block size 7, and Markov rank 256. Its weight hash is recorded in the linked
replay reports; it is not interchangeable with a draft from another target or
revision.

## Model-family lanes

| Lane | Evidence | What it does not prove |
| --- | --- | --- |
| Dense + KV | Tiny synthetic Llama live capture; DFlash feature parity; EAGLE3 capture/HF parity and one real training step. Real `Llama-3.2-1B` TP2 offline capture also passed. | A production dense target's complete train/export/serve gate |
| MoE + KV | `Qwen3-30B-A3B-Instruct-2507-FP8` TP2 architecture capture with dummy weights | Real MoE feature capture, training, export, or serving |
| MoE/linear/conv hybrid | Real Ling capture, DSpark training, export, and bounded replay on pinned H100/H200 routes | Other hybrid targets, cache-enabled multi-request replay, or universal Blackwell support |
| Linear + KV | Qwen-family configs and state fixtures only | Blanket Qwen-family certification |
| Conv + KV | Real LFM2.5 load/chat smoke only | Draft training or exact speculative state replay |
| Recurrent | Metadata/state fixtures only | End-to-end RWKV-family support |

The fixture checks for DFlash, DSpark, and Domino use Qwen3-8B-shaped draft
configs with reduced dimensions and synthetic tensors. They prove finite
updates and save/reload behavior, not real target-family compatibility.

## Method forms and evidence

| Method | Implemented form | Retained evidence |
| --- | --- | --- |
| DSpark | Metadata-derived draft; offline features; online capture | Real Ling online and offline paths below; no blanket target support |
| DFlash | Metadata-derived draft; offline features; online capture | Tiny dense live capture and synthetic update/reload; no real target training gate |
| EAGLE3 | Metadata-derived draft; offline features; online capture | Tiny dense live capture/HF parity and one training step; shared-vocabulary requirement remains |
| P-EAGLE | Metadata-derived draft; online capture | Synthetic CUDA train/checkpoint/reload smoke only; its FULL_SHARD training path is not certified |
| Domino | Explicit compatible draft JSON; offline features; online capture | Synthetic update/reload only; changing the strategy name does not convert weights/features |

DFlash2, DFly, DFlare, MTP, and multimodal training are not public capabilities
of this fork. “Implemented” in the table means code exists, not that every
model/runtime combination has passed GPU validation.

## Training topologies

### Online: GPU0 teacher/capture, GPU1 trainer

The H200 campaign ran the real Ling target/capture process on GPU 0 and the
DSpark trainer consumer on GPU 1, with a live producer and Mooncake store. The
trainer itself was one rank and target serving was TP1. It completed 100 finite
optimizer steps and produced 128 references with zero failed prompts. Loss fell
from 4.1703 to 2.5251; this is functional learning evidence on changing batches,
not held-out quality evidence.

Checkpoints were durably written at steps 25, 50, 75, and 100. A test-only
consumer crash after checkpoint 25 was recovered by a new consumer while the
original target server, producer, and store stayed alive. Adam moments, FP32
masters, scheduler state, CPU RNG, and bound CUDA RNG matched exactly; steps
26–100 completed once each. This does not prove cold-store or cold-host
recovery, nor cross-world optimizer resharding.

The same GPU0/GPU1 shape was exercised on RTX 5090 for an 8-step weights-only
DSpark fine-tune, with checkpoints at 4 and 8. The original export and the new
8-step export each matched the pinned target for one 24-token greedy request.
The new export transfer was interrupted, so its weights are not a retained
release artifact.

### Offline: TP2 capture, two-rank training

The earlier H100 production route captured real Ling hidden states with target
TP2 over 900 training rows and 98 holdout rows. The draft trainer then ran as a
two-rank FSDP1/DP2 job, resumed from step 100 through step 450, and produced a
normalized 370 MB HF/SGLang DSpark export. Exact-chat execution completed 24/24
requests without infrastructure errors, but the acceptance result remained
below the long-run gate: train `1.3456`, holdout `1.5080`, holdout position-2
`0.1085`, and full block `0/24`. This is a functional training/export result,
not a production-quality result.

The same campaign passed dense `Llama-3.2-1B` TP2 offline capture on both ranks.
The `Qwen3-30B-A3B-Instruct-2507-FP8` MoE run used dummy weights and therefore
does not count as real MoE validation. An earlier 2xH100 campaign completed TP2
layer and small dense/MoE capture checks but was preempted before real Ling
parity/training gates; it is not a second production certification.

## Serving and state outcomes

| Route | Result | Boundary |
| --- | --- | --- |
| H200 Ling feature parity | PASS: same 13 input tokens, taps 3/7/11/15/19; four tensors bitwise equal, max error 0 | One short exact-token sample |
| H200 Ling native ReplaySSM | PASS: 18 real-request rounds, all 24 output tokens, full committed temporal/convolution state within `atol=1e-5`, `rtol=1e-4` | TP1, one active request, radix off, static verify; not a benchmark |
| H100 Ling native ReplaySSM | PASS: 67 real committed rounds across three prompts; temporal/conv state and output tokens matched | Same bounded TP1/static route; see [native replay scope](NATIVE_LING_REPLAY.md) |
| H100 unhooked native-vs-target serving | PASS: 144/144 exact tokens across three 48-token requests; actual verify counts 44/44/46, accepted draft tokens 3/3/1 | Token-only check; no state hook |
| RTX 5090 Ling native ReplaySSM | FAIL: 68.9% of compared elements differed; max absolute difference `0.0684608221` | Native replay remains disabled/unsupported on SM120 |
| H100 step-450 export | 24/24 exact-chat requests completed; acceptance below threshold | No quality or speedup claim |
| H200 newly trained export | Inference launch interrupted after SSH closed and the instance exited; no completion claim | Export was backed up, but no new-export serving result was retained |

Native H100 results are bounded functional evidence: 67 state rounds and 144
token-only checks, not a claim for all prompts, TP sizes, cache modes, CUDA
graphs, or other targets. The RTX 5090 state mismatch is a separate hardware/runtime
gate and must not be hidden behind the H100 pass.

## Failures and unsupported claims

- The older Ling image lacked the patched SGLang capture sink and
  `--enable-spec-capture`; online capture was blocked until the pinned campaign
  patch was used. No force-apply or unverified backend substitution was made.
- The first SM120 preflight reproduced the fused-A-GEMM `K=1536` startup
  blocker. A later RTX 5090 run passed CUDA, target/draft serving, and short
  training, but native ReplaySSM state parity still failed as shown above.
- A100 sm80 passed basic CUDA/NCCL probes but Ling serving was blocked by an
  unsupported `fused_topk_deepseek` path. This is not Ling model validation.
- Cross-world optimizer resharding, managed cold-store recovery, full
  infrastructure-loss recovery, cache-enabled multi-request replay, and broad
  quality/speedup benchmarks remain unsupported or unmeasured.
- The maintainer's separate Ling50K training is not repeated here and is not
  certification of this packaging change or of other targets. Historical
  Ling 900/98, 1K, 5K, and 10K-equivalent experiments are likewise separate
  quality experiments, not blanket release evidence.

## CPU skip accounting

The public CPU gate's seven accelerator-dependent skips are historical test
coverage, not seven outstanding failures. The retained H100 GPU regression
passed the corresponding seven cases: one CLI config-to-training case, two
checkpoint/resume continuity cases, and four HF/SGLang export/key cases. This
does not mean the final wheel was rerun on a GPU, and it does not certify every
new model or draft export. New-model gates still require real feature parity,
finite updates, export reload, and serving/state checks.

## Release conclusion

The defensible release claim is: **the fork has working metadata/configuration
and bounded real Ling functionality on pinned H100/H200 routes, including
online GPU0/GPU1 DSpark training, offline TP2 capture plus two-rank training,
checkpoint recovery, export, and bounded H100 state replay.**

Do not claim universal dense/MoE/hybrid support, long-run quality, speedup,
Blackwell native replay, cold infrastructure recovery, or cross-world resharding.
