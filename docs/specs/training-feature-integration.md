# Additive training feature integration

## Goal and retained contracts

Integrate additive local work on the current `dev` engine. Keep the existing RAM
feature queue, acknowledgement after all backwards and optimizer success,
checkpoint/cursor commit boundary, exact optimizer/RNG resume, fixed holdout,
registered policy phases and durable exports. Do not introduce another trainer,
disk queue, scheduler or model-lifecycle controller.

## Additions

| Entry point | Behavior | Boundary |
| --- | --- | --- |
| `draftfit calibrate batch` | Isolated capture/backward/update probes with constant effective batch | Fresh state only; no live optimizer tuning |
| `draftfit calibrate forecast` and unattended `forecast` | Measured time/cost estimate and optional next-interval admission | Missing measurements and hardware contention remain visible |
| Explicit `prism_gguf` joint producer | Local GGUF/bridge hash binding, bounded context, native hidden capture | Pinned Prism build and architecture required; native logits verify tokens |
| MLX offline preparation | Persist compatible features from an Apple Silicon capture runtime | PyTorch remains the draft trainer; MLX is not an online backend |
| Benchmark runtime identity | Bind prompts, tokenization and declared runtime conditions | Operator declaration is not server attestation; legacy evidence is weak |
| `draftfit research` | ALR/IRA preparation, Osprey conversion, CAST reference decode, Redline selection | Explicit recipes; no external runtime launch or blanket serving support |
| Optional web workspace | Current CLI plans/jobs, saved controller state and evidence timeline | Local adapter; the engine owns policy and recovery |

New backend defaults must not invalidate existing non-Prism joint checkpoints.
New ALR objective fields must not invalidate other family checkpoints. Native
contexts close at runner teardown, while queue cleanup between training and
verification retains the resident target.

## Verification requirements

- Exercise actual tiny target capture, joint updates, resume, reload and export.
- Reject mismatched native source hashes, geometry and non-native verification
  logits; verify the native API adapter separately from a real pinned runtime.
- Verify probe isolation, effective batch, OOM/time handling and measured-cost
  admission after a committed checkpoint.
- Exercise research math with independent numerical/distribution references,
  real loader/optimizer updates and strict persisted-output behavior.
- Exercise benchmark CLI through a real local HTTP endpoint and tokenizer.
- Build the optional frontend and exercise the backend's current-engine adapter.
- Record the exact tested source/runtime and distinguish CPU, CUDA, fake native
  ABI, actual native Prism, MLX and production serving evidence.

See [calibration](../CALIBRATION.md), [quantized capture](quantized-capture.md),
[research methods](research-methods.md), [benchmark comparison](../PERFORMANCE_GATE.md)
and [the web workspace](../WEB_WORKSPACE.md).
