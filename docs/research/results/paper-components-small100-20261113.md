# Small-model GPU component verification

**Scoped pass. The truncated students still produce poor text. No serving speed gain was measured.**

This follow-up verifies exact module snapshots from four separate experimental branches. These results do not imply branch integration or full-paper reproduction. See the [machine-readable receipt](paper-components-small100-20261113.json) for source commits, module hashes, settings and measured values.

## Controlled setup

- GPU: RTX 3080 Ti 12 GB. Python 3.12.3, Torch 2.11.0+cu128, CUDA 12.8, Transformers 5.12.1. FP32, eager attention, deterministic algorithms, TF32 disabled.
- Target: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0. 134,515,008 parameters, 30 layers, frozen.
- Frozen drafter teacher: first eight pretrained layers, 56,632,896 parameters. Student: first four layers, 42,472,512 parameters, all trainable. Each arm shares its exact initial state and schedule.
- Public data: [WikiText2 raw](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, CC-BY-SA-3.0/GFDL. Fixed 64 training rows, 32 target-path validation rows, and 32 separate natural validation rows. Unique source IDs, text digests, and token prefixes. The machine-readable receipt records the data digest.
- Seed 20261113; batch four; 16-token prefix; four target-sampled proposal positions; AdamW learning rate 1e-4; gradient clip one. Each arm runs 100 updates on the same fixed schedule. Holdout data does not select settings or the endpoint.
- CE uses target argmax labels. D3D uses the distinct eight-layer frozen drafter, target-sampled labels, margin 0.01 and per-block exponential weights. REAL uses its public 0.9 REAL/0.1 CE mixture.

This uses exact SHA-checked public component modules in an independent native Transformers harness. It does not execute the Draftfit training, capture, production export, or serving pipeline. The machine-readable receipt records public source heads and module hashes. No tracked production source changed for this verification.

## Held-out results

KL and natural negative log likelihood (NLL) are lower-is-better. Agreement and overlap are token means over 128 held-out predictions. NLL is token-weighted over 2,004 natural next-token predictions. Expected accepted proposals exclude the bonus token and are branch estimates on fixed target-sampled paths, not serving measurements.

| State | Target KL | Natural NLL | Argmax agreement | Vocabulary overlap | Estimated accepted proposals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared initial | 14.976989 | 19.179331 | 0.093750 | 0.112342 | 0.117359 |
| Teacher argmax CE | 9.846794 | 13.813911 | 0.117188 | 0.121624 | 0.154267 |
| D3D corrected KL | 9.808967 | 13.971605 | 0.085938 | 0.125644 | 0.132450 |
| REAL 0.9 + CE 0.1 | 9.506730 | 13.557116 | 0.125000 | 0.128699 | 0.166187 |

REAL is descriptively better than CE on the listed metrics at the predeclared step-100 endpoint. This is one seed with small held-out slices. CE has lower KL and natural NLL than REAL at step 50. D3D is mixed: slightly lower final KL and higher overlap than CE, but worse natural NLL, expected acceptance, and argmax agreement. Its final agreement is below the initial value. The six fixed final continuations remain incoherent or repetitive. This is not deployable quality or evidence of general superiority.

## Actual checks

- All three arms: exactly 100 finite loss, gradient and optimizer updates; all 38 parameter tensors changed. No CPU fallback. The common schedule, initialization, sample IDs and frozen models were checked.
- Checkpoint 50: model, gradients, AdamW state, Python/Torch/CUDA random states. Reload and replay step 51 exactly reproduced original loss bits and complete state hashes, for every arm.
- Native final save/reload: actual logits and complete state hashes matched exactly for every arm. This is not the Draftfit production export adapter gate.
- REAL and D3D: same quantized CPU/CUDA inputs in FP64, FP32, FP16 and BF16, two boundary cases each; values and gradients matched declared tolerances. Low precision target/loss arithmetic uses FP32. Independent REAL finite differences and D3D analytic gradients passed.
- DForkTree: 32 scoring rows per arm on target/student top-eight model probabilities, train-only counts, add-one smoothing, retained candidate tail and neutral first-position behavior. CPU reference matched independent CUDA equations. This does not implement a GPU product, tree builder or verifier.
- TableTune: 12 GPU forward microprofiles per arm, two warmups and five repetitions each. Every raw schedule matched independent argmax. Model-derived draft-path survival uses 32 held-out draft paths. Most selected K values were zero; D3D selected K=1 only at batch one. No KV cache or serving engine was profiled, and no speed gain is claimed.
- The real sampled paths contained no EOS-stopped suffix. That boundary was exercised by synthetic numerical cases and source checks, not by observed EOS in this pretrained run.

The measured harness duration was 107.546 seconds. This duration is diagnostic, not a performance comparison. Peak allocated CUDA memory was below 1.56 GB.

## Execution identity and limits

The independent harness completed in 107.546 seconds with exit zero. Its raw report SHA256 is `e78eeb069320d1319a1ef67ebfc25be90817aaabbc4470ad640c0da43d1089b4`. The receipt records all three Python harness hashes and exact source module identities.

The executed command used native Transformers and the pinned public modules:

```sh
python -B harness/run_experiment.py --model-dir assets/model \
  --data-json assets/data.json --output run --steps 100 --device cuda
```

This private experiment harness is not a packaged Draftfit command. The execution batch retains its source, manifests, logs and all training traces. The public receipt contains no private notes, personal paths, secrets or weights.

Independent review checked the frozen harness and raw execution evidence. It found zero supported blockers for this component experiment. It did not repeat GPU execution or audit provider billing.

The macOS locked installation hash discrepancy remains. Earlier hosted Linux results retain their exact source revisions. These component results do not replace a current head's required CI or the Draftfit training, capture, production export and serving gates.
