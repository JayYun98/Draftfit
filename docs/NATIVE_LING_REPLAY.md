# Native Ling replay: validated H100 route

Native replay is an inference-state commit mechanism, not teacher/draft training
disaggregation. It folds only the accepted prefix into recurrent state and rolls
convolution state back after rejected drafts. The existing pinned SGLang backend
implements it; this platform exposes configuration and validates its behavior.

## Evidence and boundaries

H100 PCIe, Linux, SGLang `8ba213fc67550aea80dfbd5ef66f4574ec25e653`,
Torch `2.13.0+cu129`, CUDA12.9. Target `inclusionAI/Ling-3.0-tiny` revision
`e3a47d5b986e7141b6efd62597d598ebb392060d`. Existing online100 draft SHA256:
`abcd0da270620bba3c8ddfd7fd0093ea95778b03c8c28dc5728adc4b507c50d4`.

Three prompts (explanation, Python code, Korean) passed22/22/23 committed rounds:
all temporal and convolution tensors bitwise equal, maximum absolute error0.
Every case includes actual accepted and rejected nonterminal draft prefixes;
generated token IDs also match. No simulated acceptance or tolerance relaxation.

A separate H100 SXM run (same backend/weights, driver580.178.04) compared native
against target-only generation with no state hook installed: three48-token
requests,144/144 exact token IDs. Actual verify counts were44/44/46, with3/3/1
draft tokens accepted; native ring fold was active. The state gate above ran on
H100 PCIe (driver580.173.02), not this second host. Together these establish a
bounded H100 functional pass, not a quality or speedup benchmark.

This certifies the tested single-request TP1/static configuration, not radix-on,
CUDA graphs, tree verification, other targets, or all GPUs. The earlier RTX5090
state mismatch remains unresolved; H100 success does not establish its cause.
The SM120 profile still disables native replay. GPU0 teacher/GPU1 training is
separate evidence and was not rerun by this single-GPU campaign.

## Reproduce with your target/draft directories

Use the pinned image digest
`lmsysorg/sglang@sha256:3076afb2abf0e927663e9b7a475277a8b1e87cbd7ae175e22a54ade33aef3e89`.
Record target/draft hashes before the run; changing weights requires a new gate.
Run from the repository on the GPU host with fresh output directories:

```bash
export CUDA_VISIBLE_DEVICES=0 SGLANG_RAGGED_VERIFY_MODE=static SGLANG_SIMULATE_ACC_LEN=0
LING_ARGS=(--model-path "$LING_TARGET" --trust-remote-code --tp-size 1
  --dtype bfloat16 --attention-backend triton --linear-attn-backend triton
  --linear-attn-verify-backend triton --enable-deterministic-inference
  --disable-cuda-graph --disable-radix-cache --mamba-ssm-dtype float32
  --random-seed 0 --context-length 1024 --max-total-tokens 1024
  --max-running-requests 1 --mem-fraction-static 0.72
  --speculative-algorithm DSPARK --speculative-draft-model-path "$LING_DRAFT"
  --speculative-num-draft-tokens 8 --speculative-eagle-topk 1)
python scripts/validate_ling_replay.py capture legacy --port 30110 -- "${LING_ARGS[@]}"
python scripts/validate_ling_replay.py capture native --port 30111 -- "${LING_ARGS[@]}" \
  --enable-linear-replayssm-spec --linear-replayssm-cache-len 16
python scripts/validate_ling_replay.py compare legacy native
```

Set `LING_TARGET` and `LING_DRAFT` to existing local directories first. Repeat
with `--prompt` and fresh directories. The state observer runs only after the
real commit by default. `--capture-initial` is diagnostic: its extra pre-commit
synchronization can hide races, so it cannot be the sole completion evidence.
`--tokens-only` installs no state hook and supports a separate native-versus-
target-only serving check; token-only output cannot pass the state comparator.

The corresponding platform fields are
`model.sglang_enable_linear_replayssm_spec=true` and
`model.sglang_linear_replayssm_cache_len=16`. They do not alone set every condition
above. The pinned backend defaults native speculative SSM state to float32;
the legacy comparison must explicitly request float32 too. A hardware profile
alone is not a complete serving recipe or a new target certification.
