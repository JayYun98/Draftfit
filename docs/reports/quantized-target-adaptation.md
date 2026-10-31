# Quantized-target adaptation verification

Result: CPU checks pass. Quantized GPU execution and serving are not verified.
Base: `8d17a4032750e664ae432c10e6ea289f3ef81789`; include this change's working files.
The [machine record](quantized-target-adaptation/verification.json) pins the tested source content with SHA256 values.
This is a self-review and execution record, not an independent review.

## Completed checks

Environment: macOS arm64, Python 3.12.8, Torch 2.11.0, Transformers 5.8.1.
Dependencies use `requirements-cpu.lock` and hash verification.
The initial PyTorch CPU index download failed its hash check on macOS.
The standard PyPI index then satisfied the unchanged hash lock. Hash checking remained enabled.

| Check | Command or action | Result |
| --- | --- | --- |
| Public CPU suite | `python -m tests.public_cpu` | 390 tests: 382 pass, 8 CUDA skips; exit 0 |
| New boundary tests | Included `tests.test_quantized_target` | 9 pass; recipe loading tests use mocks |
| Actual preparation CLI | Tiny local Qwen3 metadata, DSpark, Transformers, NF4 recipe; fresh and saved-draft initialization | Both prepare commands pass; no quantized capture |
| Export and reload | Actual CPU checkpoint → HF export → reload | Exact tensor values and quantization metadata; floating draft |
| Build and installation | `uv build --no-build-isolation`; install resulting wheel with `--no-deps` | pass |
| Installed artifact | `python -I tests/packaging/check_public_install.py` | pass; imports installed package |
| Repository checks | `pre-commit run --files <changed files>`; `git diff --check` | pass; inapplicable hooks skip |

Logs: [CPU](quantized-target-adaptation/cpu.txt),
[CLI preparation](quantized-target-adaptation/cli-preparation.txt),
[installed artifact](quantized-target-adaptation/installed-wheel.txt),
[repository checks](quantized-target-adaptation/pre-commit.txt).
Filesystem paths are replaced with placeholders in public logs.

Earlier checks exposed sandbox socket/cache permission failures and a missing configuration documentation row.
The documentation row was added; the final suite ran with local socket access.
An export fixture initially omitted architecture metadata, then relied on an unregistered AutoConfig model type.
It now uses the registered DFlash2 configuration, as production draft initialization does.
The first standalone installation check used a retired script path and exited 2.
The current `tests/packaging/` entry above passed.
These harness and environment failures do not establish GPU behavior.

## GPU attempt and open gates

The existing A100 had 81,153 MiB free and no GPU lease owner during the read-only preflight.
The controller acquired the existing lease before launching the bounded smoke.
The worker exited 1 after 2.22 seconds: Transformers was absent before model construction.
The first bounded repair found its presumed parent package path absent.
The second repair failed during dependency download/extraction with `ENOSPC`.
No quantized model, optimizer update, or checkpoint was created.
See the [failure log](quantized-target-adaptation/gpu-failure.txt).

The repair budget is exhausted. No automatic restart or shared environment replacement occurred.
Shared service identities stayed unchanged; the observed Neo4j endpoint returned HTTP 200.
Validation evidence was transferred locally and verified by size, SHA256, and safe reads.
Only this new validation root was removed afterward. Other campaign folders remain outside this cleanup.

| Required gate | Status | Needed evidence |
| --- | --- | --- |
| Actual NF4/INT8 capture and packed checkpoint replay | not-run | Real supported GPU/runtime |
| Fresh and weights-only quantized-target updates | not-run | Finite gradients, changed weights, frozen teacher |
| GPU checkpoint resume and export reload | not-run | Production trainer lifecycle |
| Quantized-target native serving correctness | not-run | Strict target-only/original/adapted token comparison |
| Held-out acceptance and serving speed | not-run | Controlled comparison on the same target and workload |

After provisioning an isolated compatible runtime and explicitly acquiring its GPU lease, run:

```sh
timeout 600 python tests/gpu/quantized_target.py --output /path/to/NEW_DIRECTORY
```

The smoke uses random tiny Llama and Qwen3 weights, not a quality workload.
Its planned 16 cases and 48 optimizer updates are not completed measurements.
Production resume and serving need the existing training and performance gates afterward.
Do not mark the PR ready for deployment from CPU results alone.

## Review and compatibility

Reviewed all changed source, configuration, tests, and documentation against the pinned base.
Review covered capture placement, frozen tensor loading, cache checks, warm starts, provenance, export, and launch forwarding.
Review found two boundary omissions: FP8 embedding/head tensors could be cast without scales, and nested raw quantization metadata could be lost.
Both are corrected with regression cases. No additional confirmed code defect remains in this review scope.
GPU numerical behavior remains unresolved by execution.

Dense defaults and legacy dense caches remain supported.
Quantized runs require fresh, matching capture manifests.
Packed embeddings/output heads and native NVFP4 execution are not certified.
Experimental vLLM quantized capture still needs engine-specific validation.
The current public base has no EAGLE4 implementation; this PR does not add one.
See [usage](../QUANTIZED_TARGETS.md) and the [spec](../specs/quantized-target-adaptation.md).
