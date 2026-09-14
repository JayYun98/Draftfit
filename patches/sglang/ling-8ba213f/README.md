# Ling capture, exact revision 8ba213f

Status: patch applicability, Python compilation and CPU transport tests passed.
The patched base-image route passed bounded Linux 2xH200 GPU checks on 2026-09-08:
Ling TP1 online/offline tensors bitwise equal for 13 tokens, GPU0 live target with
GPU1 training for 100 finite steps, and native ReplaySSM parity for 18 verify rounds.
These runs used Triton attention, Triton linear attention and verification,
deterministic inference, and disabled CUDA graphs.
Tiny dense DFlash and Eagle3 capture gates also passed. A standalone Docker build,
other GPU/model combinations and production workloads are not validated by this.

This ports the existing SpecForge Kimi capture transport, excluding its unrelated
model/Marlin changes, onto SGLang commit
`8ba213fc67550aea80dfbd5ef66f4574ec25e653`. It uses Ling's native DSpark layer hook
and the new full/last hidden-state mode contract. No target architecture is copied
into the dense draft backbone.

## Install

Build from the SpecForge root:

```sh
docker build -f patches/sglang/ling-8ba213f/Dockerfile -t specforge-ling-capture:8ba213f .
```

Or on the exact base image after uploading this repository:

```sh
bash scripts/apply_sglang_spec_capture_patch.sh --target ling-8ba213f
# Bootstrap uv only; preserve the backend's Python/CUDA dependencies.
python -m pip install --no-deps uv==0.9.18
uv pip install --python "$(command -v python)" --no-deps accelerate==1.14.0 yunchang==0.6.4
uv pip install --python "$(command -v python)" --no-deps --no-build-isolation -e .
```

The installer checks the real source SHA and all patch hunks before changes;
unknown revisions or existing incompatible patches fail without a partial apply.
Do not substitute the v0.5.14 or Kimi patch, or replace the CUDA12 Mooncake wheel.

## Capture contract and GPU order

- Capture requires disabled radix cache, unchunked prefill, max_new_tokens=1,
  no session and no streaming. Features stay in Mooncake; only metadata returns.
- The runtime-tested route uses disabled CUDA graphs (`--disable-cuda-graph` or
  both native decode/prefill graph backends set to `disabled`). With graphs
  enabled, startup requires `--return-hidden-states-mode full`; the native FULL
  option is preserved but not GPU-validated here. NULL/LAST graph capture cannot
  serve full hidden-state requests and previously caused a runtime crash.
- Run the Ling online example with a pinned local target directory and real
  rendered training JSONL. The managed launch isolates target on GPU0 and
  optimizer on GPU1. Start with a few steps, then 100 only after tensor checks.
- Verify online/offline feature tensors for identical token IDs, revision,
  layer IDs and dtype; schema/manifest equality alone is not parity.
- After stopping the online stack, run `scripts/validate_ling_replay.py` twice
  (legacy, native ReplaySSM) and compare real committed temporal/conv states and
  output IDs. This does not validate radix checkpoint slots or every model family.
- Back up small logs, config, results and exported draft after each stage. Raw
  replay tensors may be large; compare remotely and retain small results locally.

CPU tests do not authorize a production-pass claim. The bounded GPU results above
cover separate concurrent-training, online/offline tensor and ReplaySSM gates;
they do not establish held-out draft quality or support for all SM90 models.
