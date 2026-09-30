# Release gates and remaining evidence

Source/workflow audit date: 2026-09-12; bounded native GPU evidence updated
2026-09-14. This is not a remote CI result or security certification. GPU
functional evidence does not establish blanket production readiness.
The product supports explicit target metadata/configuration extensions; it does
not automatically implement arbitrary architectures in the inference backend.

Priorities below apply to the claim being released: an unverified optional lane
can remain explicitly unsupported without blocking the narrower tested lane.

## Already reproduced: do not queue these again as missing work

| Completed bounded gate | Retained evidence | Boundary |
| --- | --- | --- |
| Real Ling offline TP2 capture and two-rank training/resume/export | H100 campaign: 900 train + 98 holdout rows, world-size 2 resume step 100→450, 370 MB HF/SGLang export, 24/24 serving requests completed | Acceptance remained below the quality threshold; MoE TP2 used dummy weights |
| Real Ling online training and consumer recovery | H200: GPU0 TP1 teacher/capture + GPU1 trainer, 100 finite steps, loss 4.1703→2.5251; consumer restarted after checkpoint 25 with exact optimizer/RNG continuity | Original producer/server/store remained alive; not whole-host recovery |
| Ling native committed-state and token parity | H200 18 rounds; H100 67 committed state rounds across three prompts; separate H100 unhooked serving 144/144 exact tokens | Pinned TP1 static route; RTX5090 native state parity failed and remains disabled |
| DFlash2 real-teacher training/export/resume | RTX3090 real Qwen3-0.6B teacher: 20 updates, exact next-step replay and BF16 export reload; 18/18 checkpoint regressions | Direct model/loss/export loop, not distributed CLI, SGLang live capture or DFlash2 serving |
| Owned native teacher capture/training | Qwen3-0.6B on two RTX3090 GPUs, real Mooncake TCP: HF BF16 parity and native vLLM FP32 parity, 20 DSpark/DFlash2 updates each, deterministic next-step replay and HF export/reload | vLLM BF16 elementwise parity FAILED; no serving or speedup claim |
| Owned managed HF/vLLM CLI | Two DSpark updates and step-2 checkpoint per backend, teacher GPU0/trainer GPU1 | BF16 functional execution, not BF16 cross-engine parity or restart certification |

The first three results were cross-checked against retained original campaign
reports and the H200 online-training summary; DFlash2 details are retained in
[MODEL_VALIDATION.md](MODEL_VALIDATION.md). The private source evidence includes
`runs/vast-dspark-production-20260907/GPU_VALIDATION_REPORT.md`,
`runs/blackwell-preflight-20260908/h200-results/online-training-summary.json`,
and `runs/native-replay-repair-20260908/REPORT.md` in the original workspace.
Those private files are not runtime package assets.
The native campaign's exact pins, passing settings and retained failures are in
[model validation](MODEL_VALIDATION.md#native-teacher-two-gpu-functional-evidence-2026-09-14);
run commands and checked-in managed fixtures are in the
[native GPU runbook](OWNED_GPU_ENVIRONMENT.md).

These passes remain valid for their recorded code/runtime scope. A packaging or
documentation change alone is not a reason to repeat a GPU campaign. Re-run only
the affected gate when its executable model, training, capture or backend path
changes, or when adding a new advertised combination.

## Remaining work by intended claim

| Priority / class | Gate | Existing evidence | Required before claiming completion |
| --- | --- | --- | --- |
| P0 / local + Linux CI | Reproducible CPU release artifact | Hash-locked Python 3.12 CPU profile; historical Linux install PASS; wheel/sdist and isolated CLI checks implemented | Run current candidate through `python -m tests.public_cpu` and the installed-wheel check in `.github/workflows/public-cpu.yaml`; retain revision, environment and result. A configured workflow is not a remote PASS. |
| P0 / local packaging; GPU only for a new image claim | Standalone distributable runtime recipe | Patched Ling base-image execution already passed; `patches/sglang/ling-8ba213f/Dockerfile` pins its base by SHA-256 | Check standalone build inputs and installed package compatibility. The standalone Docker build is not yet evidenced; this is a packaging gap, not missing Ling reproduction. Only a newly built/changed image needs its affected smoke gate. |
| P0 / real GPU, new support only | DFlash2 serving or another new combination | Ling train/export/serve and state checks already passed above; DFlash2 training/export passed separately | For DFlash2 serving, load its actual export in a compatible backend and compare real tokens. For other new combinations, run only their missing capture/train/export/serve/state gates. Do not reclassify the existing Ling results as pending. |
| P0 / real GPU, BF16 equivalence claim only | Native vLLM BF16 feature parity | Original elementwise parity gate failed; FP32 deterministic route passed | Keep BF16 cross-engine equivalence unclaimed unless the discrepancy is resolved and the unchanged gate passes. Managed BF16 updates alone do not satisfy this gate. |
| P0 / operational | Publishable source and identity | Package allowlists/notices and manual main-only publication guard exist; privileged upstream GPU jobs excluded from this fork | Review complete intended public history and copied-file notices, approve distribution identity, and deliberately configure publication credentials/environment. No publication is performed by this audit. |
| P1 / real GPU | Useful improvement on personal held-out data | One-sample overfit and functional tests are useful correctness evidence | Compare target-only and draft serving under identical prompts/settings/hardware; retain exactness, latency, throughput, acceptance and peak memory. Set the desired benefit threshold before promotion. |
| P1 / operational + real GPU | Host-loss recovery | CPU recovery contracts and historical same-world GPU reload/resume evidence exist | Restore a completed checkpoint and required producer/consumer state on a replacement host with matching world size and runtime; demonstrate continued updates and record restore time/data loss. Cross-world optimizer resharding is outside current support. |
| P1 / operational | Sustained operation | Bounded campaign runs and runtime backpressure configuration exist | Choose supported workload/concurrency, monitor GPU memory, disk growth and failures, exercise interruption/restart, and define checkpoint retention and rollback to a known-good export/runtime. Short smoke tests do not establish a service-level guarantee. |
| P1 / local + platform | Fresh macOS CPU install | Existing local environment works; documented Torch 2.11.0 wheel hash mismatch prevents a fresh locked install | Keep Linux-only installation evidence explicit; investigate a trusted index/artifact resolution and regenerate/retest the lock if changed. Never bypass hashes or trust the mismatching download. macOS CUDA training is not supported. |
| P2 / local | Documentation consistency | Public support, release, history and attribution documents exist | Keep historical source-repository findings distinct from this clean-history repository, distinguish training CUDA evidence from serving evidence, and update counts only from executed gates. |

## Operational recovery checklist

Before a production run, record the config and overrides, target/tokenizer
revisions, exact draft JSON, image digest, backend patch, world size and run
identity alongside the checkpoint. Preserve every required per-rank optimizer
and RNG payload, not merely an exported draft. For disaggregated training also
preserve the required durable control/consumer state and feature data according
to the selected deployment. Keep a complete copy outside the failure domain of
the training host.

On recovery, validate that the checkpoint is complete, restore the matching
runtime and world size, and use `training.resume_from`; do not also supply a
weights-only `model.draft_checkpoint_path`. Check the resumed step and finite
updates before continuing the workload. An export can be used for serving or a
new weights-only fine-tune, but cannot restore optimizer continuity. Record a
failed rehearsal as pending rather than inferring recovery from file existence.

## Bounded repository observations

The inspected HEAD was `de43b88`, with eight reachable commits. Its tracked paths
and all reachable object path names contained no matches for the checked private
key, environment-file, model-weight, log or `VAST_HANDOFF` filename patterns.
A current-HEAD content search for private-key headers and selected common token
shapes matched only `tests/test_packaging/test_assets.py`, which contains scanner
test patterns. This is neither a complete content/history secret scan nor proof
that publication is safe. No secret values were reported.

`SOURCE_ATTRIBUTION.md` records a 2026-09-08 audit of the original repository,
including its then-tracked operational handoff file. Read dated findings in
their respective repository contexts; removing current files is not a history purge.

The CPU workflow uses hosted runners and read-only repository permissions.
Publication is manual and requires the CPU workflow plus explicit identity
confirmation. The inherited GPU workflow runs only for the upstream repository;
it supplies no automatic GPU coverage for this fork. The inherited documentation
deployment also excludes this fork. Actual Actions enablement and protected
environment settings were not inspected remotely.

This checklist does not provision a runtime. Execute contributor commands from
the source checkout; the wheel's exported guides do not include the CPU lock or
test suite.
