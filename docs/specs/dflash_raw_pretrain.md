# Raw-text DFlash-family pretraining

## Problem / scope

Inspired by [speculators RFC #1127](https://github.com/vllm-project/speculators/issues/1127) and its `reed-meyerson/speculators@dflash-pretraining` implementation (inspected at `d7c6966`): distillation needs costly verifier feature capture. Add a bounded, single-device pretraining path for the existing Qwen3-shaped DFlash, DSpark and DFlash2 drafts, followed by exact auxiliary-projection weight widening and the existing weights-only distillation warm start. [TorchSpec](https://github.com/lightseekorg/TorchSpec) was inspected locally at `bd64d93`; its conversion remains an initialization, not a prediction-preserving serving conversion.

The RFC is open research, not a guarantee of improved acceptance. This implementation reads a local JSONL text corpus and the frozen target embedding/head; it never runs the verifier decoder. It does not replace on-policy distillation or certify a serving speedup.

## Contract / implementation

- `draftfit pretrain`: local text JSONL, target path/revision, draft config, strategy, fixed token/step budget, optional held-out JSONL; writes a normal HF draft checkpoint and metrics. Fail on empty/short corpus, incompatible geometry, or an occupied output directory.
- `draftfit expand-aux-layers`: a one-tap checkpoint with tap 0 and selected destination taps; copy old projection into the tap-0 slot, zero all new slots; preserve other tensors and block semantics. The output is a normal HF draft checkpoint.
- Existing `model.draft_checkpoint_path` starts a new distillation run with new optimizer. Existing `training.resume_from` remains full same-run resume. The target revision, tokenizer, block size, anchor alignment and attention layout must match.

## Acceptance and evidence

| Requirement | Acceptance | Check | Status |
| --- | --- | --- | --- |
| Verifier-free update | Real tiny target files + raw corpus produce finite loss, nonzero draft update, ordinary reload; no decoder construction | `python -m unittest tests.test_pretraining -q` | pass: target embedding/head only, no decoder load |
| All three family wrappers | One pretrain step on each with hard token targets | `python -m unittest tests.test_pretraining -q` | pass: DSpark, DFlash, DFlash2 |
| Exact widening weights | New projection columns are zero; old weights and all other tensors unchanged | `python -m unittest tests.test_pretraining -q` | pass on tiny CPU model |
| Bitwise output parity after widening | Full forward must be bitwise identical | CPU full-forward comparison | fail: 1–2 elements per 64 differed by at most 7.45e-9 due changed matrix width; numerical tolerance gate 1e-6/1e-7 passes |
| Existing distillation handoff | New checkpoint accepted by existing warm-start loader after widening | `python -m unittest tests.test_pretraining -q` | pass: weights-only loader |
| GPU quality and serving | Matched held-out acceptance, token parity and throughput against scratch and official drafts | real GPU A/B | not-run |

## Usage / results

See [DFlash family guide](../DFLASH_FAMILY.md#raw-text-pretraining). The final full CPU gate ran 401 tests, 8 skipped (`/tmp/draftfit-pretrain-public-cpu-final.log`). The rebuilt wheel passed installed-package onboarding and exposed both new commands. A stale config allowlist initially rejected two already-present LFM DSpark configs; it was updated to include those valid configs, then the gate passed. GPU A/B requires an independently authorized compatible runtime and budget. The CPU checks establish execution and shape/alignment, not useful draft acceptance. The independent pretrain loop is single-device and saves model weights, not optimizer state; restart/resume of a long pretrain is not implemented.

## GPU experiment plan (not run)

On a pinned Qwen3-shaped target and tokenizer, use the same disjoint train/validation feature caches and held-out prompts for each strategy. Compare scratch distillation against pretrain → widen → identical distillation, fixing seed, model geometry, precision, optimizer, on-policy examples, distillation steps and serving backend. Record the pretrained checkpoint hash and feature-layer order. Judge held-out acceptance and exact target-token/state correctness separately from loss; only then compare target-only, scratch and warm-start throughput at matched prompt set, concurrency and speculative length. Repeat runs for a quality claim. No previous MiniCPM5/MLX or TorchSpec zero-step throughput is evidence for this new path.
