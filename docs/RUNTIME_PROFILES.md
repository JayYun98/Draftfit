# Runtime profiles

Compose one model/workflow YAML with one hardware override list. Model identity,
draft architecture, dataset, capture contract and GPU placement stay in the model
YAML; the profile selects runtime kernels. These lists use the existing dotted
CLI overrides and require no loader or additional dependency.

The Ling tiny Linux 2xH200 route has **bounded GPU validation** from campaign
2026-09-08 (instance 50190940), using the profile's Triton attention default. This does not validate every SM90 model or a
production workload. A subsequent 2xRTX5090 campaign validated the bounded default
route below, but failed native ReplaySSM state parity. Other combinations remain
**GPU_PENDING**.

| Profile in `configs/runtime_profiles/` | Intended hardware | Current limitation |
| --- | --- | --- |
| `linux-sm90-cu129.yaml` | H100/H200, SM90 | Ling tiny on 2xH200 passed with default Triton; other model/device combinations pending |
| `linux-sm120-cu129.yaml` | RTX5090 32 GB, SM120 | Ling 8-step online fine-tune/export/exact-token smoke passed; native ReplaySSM state parity FAILED and is disabled |
| `linux-sm80-cu129.yaml` | A100, SM80 | Only basic torch probe, no Ling validation; historical fused-top-k failure |

The subsequent H100 PCIe campaign validated native replay on three prompts:
67 committed rounds, bitwise-equal temporal/convolution states and exact tokens.
See [the complete pinned native replay recipe](NATIVE_LING_REPLAY.md); a hardware
profile alone does not set every required serving option.

The 2xH200 run measured bitwise-equal online/offline tensors for the same
13-token Ling input at taps 3,7,11,15,19; completed 100 finite online trainer steps
with target TP1 on GPU0 and training on GPU1; and passed native ReplaySSM parity
over 18 verification rounds, comparing full committed temporal/convolution states
and generated token IDs. The separate tiny dense gate passed DFlash feature
parity and Eagle3 online capture/HF parity plus a real training step. These are
functional correctness checks, not held-out model quality or throughput claims.

The separate Blackwell campaign (instance 50247513) used GPU0 for the pinned
Ling target and GPU1 for eight weights-only fine-tuning steps. Its new export
matched target-only greedy output for one 24-token request. Two-rank synthetic
FSDP2 auxiliary-head updates and same-world reload passed. DFlash/DSpark/Domino
CUDA updates and weight reload used synthetic features, not certified target
families. Native ReplaySSM versus default committed state failed with maximum
absolute difference 0.0684608221; the SM120 profile disables that option. This
does not isolate hardware as the cause or invalidate a differently configured
prior H200 experiment. The two native fold-kernel unit passes are not full
state-replay certification. Peak memory/throughput are not certified.

The expected Linux stack for these initial Ling profiles is image
`lmsysorg/sglang:dev-cu12-Ling-3.0-tiny`, SGLang revision
`8ba213fc67550aea80dfbd5ef66f4574ec25e653`, torch `2.13.0+cu129`, CUDA runtime
12.9, plus the platform's existing spec-capture patch. The image tag is mutable;
record the resolved digest, source/patch state, package versions, GPU capability
and host driver on each run. Profile comments document expectations, not an
installation or an enforcement mechanism. Driver/runtime suitability must be
checked on the selected machine.

Run from the SpecForge repository using its configured Python environment:

```sh
python - <<'PY'
import subprocess
import sys
import yaml

with open('configs/runtime_profiles/linux-sm90-cu129.yaml') as f:
    overrides = yaml.safe_load(f)
subprocess.run([
    sys.executable, '-m', 'specforge.cli', 'train',
    '-c', 'examples/configs/ling-3.0-tiny-dspark-online.yaml',
    '--plan', *overrides,
], check=True)
PY
```

The example prints the plan only. On a GPU host, after preflight, remove `--plan`
to execute. This selects the campaign-tested Triton route. For direct Python callers, pass the same list as the second argument
to `specforge.config.load_config(model_yaml, overrides)`. Append explicit per-run
overrides last when necessary; later values win. Select one hardware profile per
run. A capture server's explicit `attention_backend` overrides the model default,
so remove conflicting server settings in the base YAML before composing.

The Ling online example already reserves GPU0 for the live TP1 capture server and
GPU1 for training, with one running request and 4103 capture tokens (4096 training
tokens plus capture headroom). Profiles preserve this headroom and placement.
Older TP1 H100 target+draft serving logs showed about 15.74 GB initial allocation;
that is evidence for trying 32 GB, not a 5090 peak-memory guarantee.

SM90 and SM120 profiles use Triton full attention. SM120 disables native
ReplaySSM speculative verification after its failed full-state comparison.
FA3 is an optional Hopper override, but this online composition has not been
GPU-validated with it; this pinned server has no `fa2` backend name. Datacenter Blackwell
SM100 CuTe KDA kernels do not imply SM120 support. Deterministic inference bypasses
the known Ling fused QKV-A GEMM path that asserts on hidden size 1536 on SM120.
The Ampere profile does not claim to fix Ling's independent fused-top-k failure.
The tested capture route disables CUDA graphs. The pinned patch now rejects
capture startup with graph modes below FULL; native `--return-hidden-states-mode
full` is an allowed alternative when graphs are enabled, but that graph path
has not passed this campaign's GPU checks.

macOS may parse these files and print plans with installed CPU dependencies;
these are Linux CUDA execution profiles, not macOS/MLX runtime support. New models
reuse profiles only after reviewing their architecture-specific kernel needs, and
each model/GPU/OS/version combination remains GPU_PENDING until its own checks
pass. Do not create a duplicate model YAML for every hardware combination.

Before promoting a combination, record a CUDA/bf16 runtime smoke, resolved launch
plan, live capture, one trainer step, generation/replay parity where applicable,
and measured GPU memory with producer and trainer active. Preserve failure
tracebacks and mark failed combinations accordingly; a config-only check cannot
promote support status.
