# Model environments: reuse by default

The maintained SpecForge-derived engine ships inside `draftfit` for all targets.
Do not install upstream SpecForge separately. Custom SGLang may be needed for
target execution/state support or hidden-feature capture into Mooncake.
A working chat API alone proves neither capture nor speculative serving.

| Change | Default action | Check |
| --- | --- | --- |
| Data, learning rate, steps | Reuse environment | Format, loss masks, holdout separation |
| Size/checkpoint in a supported architecture | Update config; consider environment reuse | VRAM, tokenizer/template, depth, taps, head paths |
| DSpark → DFlash2 | Check backend capabilities first | Capture layout, selector/convolution export loading, serving |
| KV → linear/conv/recurrent/hybrid | Check backend/state contracts | Rollback/replay, capture boundaries; implement missing support |
| GPU/CUDA/driver | Select compatible runtime profile | Kernels, Mooncake wheel, attention backend; config alone is insufficient |
| SGLang source/patch | Pin separate environment | Patch applicability and affected real-runtime checks |

## Choose an environment

- CPU configuration/data/tests: [hash-locked uv environment](ENVIRONMENT.md).
- Ling online: [runtime profiles](RUNTIME_PROFILES.md) and
  [exact capture revision](../patches/sglang/ling-8ba213f/README.md). Preserve the
  base-image/source/patch combination instead of replacing it with stock PyPI
  SGLang. New Dockerfile builds are distinct from historical base-image evidence.
- Native Transformers/vLLM: [owned GPU environment](OWNED_GPU_ENVIRONMENT.md).
- DFlash2: [serving prerequisites](PUBLIC_SUPPORT.md); the Ling image is not
  automatically compatible.
- Other targets: [target extension](TARGET_EXTENSION.md). Config connects
  supported targets; it does not implement missing backend architectures.

## When a new environment is necessary

Use a separate venv/image tag. Record target revision, tokenizer/template,
platform wheel SHA256, backend source SHA, patch SHA256, image digest and
Torch/CUDA/Mooncake versions. Check patches against exact source before
installation. Stop on patch failure; never force another model's patch.

In a provisioned environment, install the platform wheel with uv and
`--no-deps`. For new environments, provision exact backend dependencies first;
that flag cannot supply missing dependencies. Keep the CPU lock out of GPU
environments. Rerun affected capture/training/export/serving/state checks,
not every historical test merely because configuration changed.

## Repository agent skills

- [GPU rental and cleanup](../.skill/dspark-rent/SKILL.md)
- [Model/config/environment setup](../.skill/dspark-config/SKILL.md)
- [Training and validation](../.skill/dspark-train-validate/SKILL.md)

`.skill/` is a repository location, not universal auto-discovery. Ask agents
to read the relevant file explicitly or configure their skill search paths.
Keep skills with the repository or adjust relative links when copying them.
No global skill installation is required.
