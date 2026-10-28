# EAGLE-4 compatible training and MiniCPM comparison

## Problem and scope

Add `training.strategy=eagle4` and `LlamaForCausalLMEagle4` to Draftfit's
config → feature capture → training → checkpoint → export path. Compare the
existing DSpark and DFlash2 implementations with this draft on MiniCPM5-1B.

Reference: [EAGLE-4](https://github.com/eagle-hq/EAGLE-4), inspected at
`3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d`. This implementation extends
Draftfit's existing Apache-licensed EAGLE backbone; it does not vendor the
source-available eagle-harness code or implement its optimized engine kernels.
The external runtime retains its own license and installation requirements.

The upstream [new-target guide](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/new-model.md)
does not publish a reproducible revision of the inspected efficient-head
training bundle. This is a compatible independent training implementation,
not a claim to reproduce the authors' complete training procedure or results.

## Contract and implementation

- Dense, one-layer concatenation-fusion draft; exactly three target block
  outputs; full target vocabulary; rotary positions; SDPA training only.
- Optional sliding window over committed context. Recurrent speculative steps
  append their ancestors without evicting the committed window during a round.
- `norm_output=true` feeds normalized output into the next draft step and LM
  head. Candidate selection uses a rank-R projection; the selected rows are
  rescored with the full dense head.
- `training.eagle4_stage=draft` trains the backbone/full head while freezing
  candidate factors. `candidates` freezes the dense draft and minimizes forward
  KL from its full-head scores on supervised positions. This requires a dense
  checkpoint. `eagle4_initialize_candidates=true` initializes the factors with
  randomized truncated SVD on a stage-transition warm start; full resume must
  preserve existing factors.
- Reuse the EAGLE offline/server feature layout, teacher projection, unroll,
  compact-teacher option and checkpoint resume boundary. Vocabulary pruning,
  alternate attention blocks, MoE, NoPE and other training attention backends
  are excluded.
- Export includes `lm_head.weight`, `efficient_lm_head_down_proj.weight` and
  `efficient_lm_head.weight`. HF export includes the real frozen embedding;
  specialized SGLang export shares the target embedding. Neither exporter
  installs or qualifies an EAGLE-HARNESS engine.

## Acceptance and evidence

| Requirement | Criterion decided before execution | Check | Status |
| --- | --- | --- | --- |
| Candidate head | Full-head row scores exactly match selected candidate scores | `tests.test_modeling.test_eagle4` | pass, CPU |
| Attention and padding | Future/out-of-window inputs cannot affect visible queries; padding excluded | Same suite | pass, CPU |
| Candidate optimization | Only candidate factors update; masked forward KL decreases | Same suite | pass, CPU |
| Registry and config | CLI lists EAGLE4; full vocab, one layer and SDPA admitted | Registry/config tests and real CLI | pass, bounded CPU |
| Export | Native candidate tensors survive HF reload and SGLang export | Model/export tests | pass, CPU |
| Real training | Finite loss/gradient, actual optimizer updates for each method | Bounded MiniCPM CUDA smoke | pending |
| 5,000-step comparison | Same frozen data, target revision, order, seed and GPU | Source-only comparison runner | pending |
| Held-out generation | Actual exports generate target-verified tokens on unseen prompts | Generation gate | pending |
| Optimized serving | Compatible engine loads this exact draft and greedy correctness passes | Separate external runtime | pending |

## Usage

Use the environment instructions in [ENVIRONMENT.md](../ENVIRONMENT.md).
For an existing EAGLE-format offline cache with the target's three auxiliary
block outputs and normalized final hidden states:

```sh
draftfit target prepare /path/to/target --local-only --strategy eagle4 \
  --hidden-states /path/to/features --output-dir ./eagle4-dense \
  --draft-config configs/minicpm5-1b-eagle4.json \
  --set training.attention_backend=sdpa --set training.ttt_length=3 \
  --set training.max_steps=4000 --set training.save_interval=1000
draftfit train -c ./eagle4-dense/train.json --plan
draftfit train -c ./eagle4-dense/train.json
```

Use a fresh run config/output directory for candidate distillation, retaining
the same target, features and draft JSON. Set `model.draft_checkpoint_path` to
the dense checkpoint, `training.eagle4_stage=candidates`,
`training.eagle4_initialize_candidates=true`, `training.max_steps=1000`, and
`training.resume_from=null`. Full resume instead uses the same stage and sets
`eagle4_initialize_candidates=false`.

```sh
draftfit export --checkpoint /path/to/candidate-checkpoint \
  --draft-config configs/minicpm5-1b-eagle4.json --to hf \
  --output-dir ./eagle4-export --embedding-source /path/to/target
```

For external optimized serving, use the upstream EAGLE-HARNESS release recipe
and its compatible engine. SGLang uses `--speculative-algorithm EAGLE_HARNESS`,
not `EAGLE3`. Confirm the [upstream runtime contract](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/compatibility.md)
and measure new-model greedy correctness before calibration or performance.

## Experiment plan

- Existing shared A100 80GB only; do not stop its vLLM process. Allocator budget
  5,600 MiB; require an additional 1,536 MiB free before allocation.
- Target `openbmb/MiniCPM5-1B@87179e5c1f455ef22e6223592d2d61351b525bfc`.
- Real UltraChat conversations: disjoint `train_sft` and `test_sft`; immutable
  dataset revision and per-feature hashes recorded in `capture.json`.
- 1,024 training samples, 64 held-out samples, maximum 256 tokens, seed 29,
  batch 1, BF16, one draft layer, target width, taps `[1,11,20]`.
- 5,000 optimizer updates each. EAGLE4 uses 4,000 dense + 1,000 candidate
  updates; DSpark/DFlash2 use 5,000 method-specific updates. AdamW LR 1e-4,
  gradient clipping 0.5; common sample order. Parallel methods use block 7 and
  eight uniform anchors; EAGLE4 unrolls three recurrent training steps.
- Architecture, objective, number of supervised predictions, head ownership
  and trainable parameters differ by design. Total training losses are not a
  common quality metric. One seed is a bounded comparison, not evidence of
  general superiority. Shared GPU compute invalidates isolated speed claims.

```sh
python benchmarks/train_eagle4_comparison.py \
  --target /path/to/pinned-minicpm5-1b --output /path/to/new-comparison \
  --steps 5000 --dense-steps 4000 --train-samples 1024 --eval-samples 64 \
  --max-length 256 --memory-mib 5600
```

The runner never rents, stops processes or modifies the live serving
environment. Raw feature files, conversations, weights and operational SSH
details remain outside the PR. Final evidence belongs in an experiment report.
