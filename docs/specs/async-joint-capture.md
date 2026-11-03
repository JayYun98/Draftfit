# Asynchronous joint feature capture

## Problem and scope

Joint training previously filled its RAM queue on the trainer thread. Capture stopped
while the trainer ran. An optional producer now captures frozen-target features
while the trainer consumes earlier batches. Both models can remain on one GPU.
The default remains synchronous.

This change covers the single-rank joint DSpark, DFlash and DFlash2 path with a
built-in Transformers teacher. It does not launch or change an existing experiment.
Prism and remote-code teachers are rejected in this mode. Managed vLLM/SGLang
services and multi-rank training keep their existing execution paths.

## Contract

Set `training.joint_async_capture=true` together with a nonzero
`training.joint_feature_queue_bytes`. The existing batch limit also applies.
Queued batches, optimizer leases and one capture reservation share these limits.
The producer reserves the estimated final CPU storage before capture. It waits
when credit is unavailable. Capture fails if actual storage exceeds its reservation.
Capture scratch space, model weights and transport buffers are outside this
feature-queue budget; it is not a total process RAM or GPU memory limit.

The trainer releases a batch only after its optimizer update succeeds. An
insufficient accumulation budget raises an error instead of waiting forever.
Producer errors reach the trainer with their original cause. Early termination
joins the producer, waits for any active capture and discards unconsumed lookahead.
An active native capture cannot be forcibly interrupted by a Python thread.

Evaluation, checkpoints, generation verification and observer memory handoffs
pause production and wait for active capture. Token rows and the consumed cursor
remain the recovery source. Prefetched features are not checkpointed or written
to disk. Resume regenerates lookahead after the committed cursor.

Teacher initialization occurs on the caller thread under saved RNG state. The
producer runs deterministic, frozen evaluation forwards without restoring shared
random state. CUDA capture uses a separate stream and completes CPU copies before
publication. Sufficient GPU memory permits coexistence; it does not establish
kernel overlap or a throughput gain. Both tasks still share GPU compute resources.

The new flag defaults to false and is omitted from old checkpoint fingerprints
when disabled. Changing it changes the resume contract. Keep it fixed on resume.

## Usage

Add these fields to the base configuration used by `family --joint`:

```yaml
training:
  joint_online_capture: true
  joint_async_capture: true
  joint_feature_queue_bytes: 2147483648
  joint_feature_queue_batches: 8
```

Keep `model.target_backend=transformers`, `model.trust_remote_code=false`, and the
existing tokenized train/eval inputs. Choose byte and batch limits that hold the
complete gradient accumulation window. Use a fresh output directory.

```sh
python -m draftfit.cli family --manifest family.json --joint --plan
python -m draftfit.cli family --manifest family.json --joint
```

`AsyncFeatureQueue` also accepts a capture callback and a conservative byte estimator.
The callback must return completed CPU tensors. It must not restore process-global
RNG state while training runs. Call `release(lease, consumed=True)` after the update,
`pause()` around target-changing operations, and `clear()` in teardown.

## Acceptance and evidence

| Requirement | Evidence | Result |
| --- | --- | --- |
| Capture progresses while the consumer holds a batch | Real tiny Qwen source and thread rendezvous | CPU pass |
| Ordered bounded credit, including active capture and optimizer leases | Queue regression tests | CPU pass |
| Backpressure, producer errors, early close and pause | Thread rendezvous and failure tests | CPU pass |
| Same updates as synchronous capture and exact checkpoint continuation | Tiny Qwen joint CLI, all three objectives | CPU pass |
| Epoch accumulation, backward failure and verification-policy resume | Joint queue integration regressions | CPU pass |
| No feature files or target-weight changes | Real CLI artifact checks | CPU pass |
| CUDA correctness, peak VRAM and throughput | Same-GPU runtime comparison | Not run |

Run focused checks in the contributor CPU environment:

```sh
SPECFORGE_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest tests.test_async_feature_queue tests.test_joint_async_capture -v
```

CUDA validation needs an idle, authorized device and a matching runtime:

```sh
DRAFTFIT_RUN_QUEUE_CUDA=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest tests.test_joint_async_capture.JointAsyncCaptureCUDA -v
```

This gate uses the same tiny CLI inputs and separate synchronous/asynchronous
output roots. Separately compare complete update time, peak VRAM,
checkpoint state and queue teardown. No GPU result or speedup is claimed here.
GPU allocator peaks during async execution include both workloads; they are not
isolated target-capture peaks. The existing Qwen3.8 FP8 experiment has not adopted
this change.

The full locked CPU environment installation stopped at a Torch SHA-256 mismatch.
The hash check was retained. Focused checks use the existing Python 3.12.8,
Torch 2.11.0 and Transformers 5.8.1 environment. That environment lacks Gemma4;
full locked-suite, installed-wheel and Gemma4 results remain unverified.
