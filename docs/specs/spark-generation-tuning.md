# Spark fixes and resident generation batching

## Scope

Selectively integrate the Spark training fixes from `a3f2346` and the generation
batch controller from Humanizer `cc21374` into the current development code.
Historical training results motivate this work but do not establish new quality
or speed results for the integrated revision.

## Behavior and compatibility

| Path | Previous behavior | Integrated behavior |
| --- | --- | --- |
| DSpark tau diagnostics | A masked committed anchor zeros the prefix product | Begin at the first actual proposal when `sample_from_anchor=false` |
| Joint evaluation | Caller RNG affects sampled anchors | Use the configured seed and restore caller RNG even on setup failure |
| FP32 target residual to BF16 draft | Direct projection raises a dtype error | Convert feature dtype at the projection boundary |
| Joint save policy | Joint state plus duplicate arm checkpoints | Preserve the default; optional joint-only state retains optimizer and cursor |
| Spark reference prefix verification | Block-shaped target calls can disagree at low precision | Verify against sequential target prefix semantics |
| Data generation | Fixed server request concurrency | Keep the default and add a resident local Transformers path with opt-in CUDA batch tuning |

The DSpark change affects a diagnostic, not its training objective. A corrected
tau value must not be presented as newly measured rollout acceptance. Fixed
evaluation anchors make milestone comparisons comparable; they can change the
selected best checkpoint relative to historical noisy evaluation.

EAGLE4 candidate factors, native-logit alignment, native rotary precision and WSD
remain unchanged. Quantized feature provenance checks remain enforced.
Output-only checkpoint settings must not change the semantic resume fingerprint.
Generation tuning is independent of effective training batch and LR scheduling.

## Generation contract

Use the [resident generation command](../basic_usage/data_preparation.md#resident-local-generation).
One frozen target stays resident while generation probes and production batches
run serially. Model/runtime/generation identity limits receipt reuse. Probes
preserve RNG and cannot consume the output journal. Only CUDA OOM reduces a
batch, and the same pending rows are retried before any row commit.
Rendered chat templates are tokenized without adding a second set of special tokens.
Input paths cannot alias output journals or receipts. Resume preserves completed
records and discards only an interrupted last JSONL record; corruption in a
completed record is rejected.
Complete conversations are durably appended; changed dataset or generation
contracts require a new output. This is not exact RNG resume or a background
GPU-sharing scheduler. Existing server-based regeneration is unchanged by default.

## Acceptance

- Closed-form tau excludes committed anchors with and without objective chunking.
- Identical weights/data evaluate consistently without advancing training RNG.
- FP32 residual features give the same output as explicit BF16 conversion.
- Existing joint resume, quantized-source, EAGLE4 and WSD tests remain valid.
- Actual tiny CPU local-target generation completes multi-turn rows and resumes
  without duplicate rows; model errors cannot commit unfinished rows.
- Mocked CUDA tests cover memory selection, compatible receipt reuse and
  OOM-only retry. They do not establish actual CUDA behavior or speed.
- GPU training, native Spark runtime parity, CUDA tuning throughput and serving
  A/B are not run in the local CPU environment. They remain separate gates.
