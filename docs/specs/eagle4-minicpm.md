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
  and checkpoint resume boundary. Vocabulary pruning and compact teacher mode,
  alternate attention blocks, MoE, NoPE and other training attention backends
  are excluded.
- Export includes `lm_head.weight`, `efficient_lm_head_down_proj.weight` and
  `efficient_lm_head.weight`. HF export includes the real frozen embedding;
  specialized SGLang export shares the target embedding. Neither exporter
  installs or qualifies an EAGLE-HARNESS engine.

## Acceptance and evidence

See the [MiniCPM experiment record](../research/eagle4-minicpm.md) for recipes,
source identities, execution results and failed or unrun gates.

| Requirement | Criterion decided before execution | Check | Status |
| --- | --- | --- | --- |
| Candidate head | Full-head row scores exactly match selected candidate scores | `tests.test_modeling.test_eagle4`; held-out coverage | partial: CPU checks pass; BF16 dense-GEMM argmax differs |
| Attention and padding | Future/out-of-window inputs cannot affect visible queries; padding excluded | Same suite | pass, CPU |
| Candidate optimization | Only candidate factors update; masked forward KL decreases | Same suite | pass, CPU |
| Registry and config | CLI lists EAGLE4; full vocab, one layer and SDPA admitted | Installed-wheel onboarding and actual CUDA CLI | pass, bounded |
| Export | Native candidate tensors survive HF reload and SGLang export | Model/export tests; actual CLI HF reload | pass, CPU and bounded CUDA |
| CLI lifecycle | Candidate full resume equals uninterrupted fitting; dense tensors stay frozen | Actual CUDA train/resume/export CLI | pass, tiny target |
| Real training | Finite loss/gradient, actual optimizer updates for each method | Full MiniCPM CUDA runs | pass |
| 5,000-step comparison | Same frozen data, target revision, order, seed and GPU | Full dense fitting; EAGLE4 adds 1,000 candidate updates | pass, bounded single seed |
| Capture matches generation mode | Train on the selected target's generated continuations | Follow-up data audit and capture regression | pass, CUDA resumed 291 records; full 1,024/64 hash/prompt/mask audit passed; zero prompt/sample overlap |
| 10,000-update extension | Warm-start every dense draft; new common target-generated cache; separate EAGLE4 candidate fitting | Follow-up run | pass, all arms reached 15,000 dense updates; EAGLE4 added 1,000 candidates; export reload exact |
| Held-out generation | Actual exports generate target-verified tokens on unseen prompts | Generation gate | fail, original and extended BF16; extended equality 56/64, 56/64, 53/64; FP32 control passes 64/64 per arm |
| Optimized serving | Compatible engine loads this exact draft and greedy correctness passes | Separate external runtime, both arms CUSTOM attention | fail: original and extended exports load/serve; strict equality remains 7/8 |

The candidate tests check selected-row scoring against a full FP32 projection
with numerical tolerance, plus FP32 accumulation for BF16 inputs. They do not
certify bitwise equivalence with every BF16 dense GEMM. On the real held-out
states, original rescored argmax agreement is 99.118%. The extension records
98.372% on different positions; see the experiment record.

## Usage

Use the environment instructions in [ENVIRONMENT.md](../ENVIRONMENT.md).
For an existing EAGLE-format offline cache with the target's three auxiliary
block outputs and normalized final hidden states:

```sh
draftfit target prepare /path/to/target --local-only --strategy eagle4 \
  --hidden-states /path/to/features --output-dir ./eagle4-dense \
  --draft-config configs/minicpm5-1b-eagle4.json \
  --set training.attention_backend=sdpa --set training.ttt_length=7 \
  --set training.max_steps=5000 --set training.save_interval=1000
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
  started at 6,500 MiB with a 1,536 MiB initial reserve. After the external
  workload grew, remaining arms used 6,200 MiB plus a 1,024 MiB reserve.
- Target `openbmb/MiniCPM5-1B@87179e5c1f455ef22e6223592d2d61351b525bfc`.
- Real UltraChat conversations: disjoint `train_sft` and `test_sft`; immutable
  dataset revision and per-feature hashes recorded in `capture.json`.
- 1,024 training samples, 64 held-out samples, maximum 256 tokens, seed 29,
  batch 1, BF16 and target width. Capture taps `[1,6,11,16,20]`; EAGLE4
  consumes the `[1,11,20]` subset.
- Follow the current [family guide](../DFLASH_FAMILY.md) and
  [MiniCPM recipes](../RECENT_MODEL_EXPERIMENTS.md): five DSpark layers with
  Markov rank 256, three DFlash2 layers with selector rank 256, block 7 and
  up to 512 random anchors. All draft backbone parameters train.
- EAGLE4 keeps its documented single recurrent layer and seven training unrolls.
  Each method receives 5,000 dense optimizer updates. Candidate distillation
  adds 1,000 EAGLE4 updates after dense training; it does not replace dense updates.
  The upstream [two-stage recipe](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/new-model.md)
  requires a trained dense draft before candidate fitting.
- Use Draftfit's `BF16Optimizer` with FP32 masters, cosine scheduling and warmup.
  DSpark/DFlash2 use the example LR 6e-4, warmup 4%, clipping 1.0;
  EAGLE4 uses default LR 1e-4, warmup 1.5%, clipping 0.5.
- These are documented starting recipes adapted to MiniCPM5-1B, not tuned
  optima. The documented official DSpark warm start belongs to MiniCPM5-2B;
  its geometry cannot initialize MiniCPM5-1B. All three arms start from scratch.
- Architecture, objective, number of supervised predictions, head ownership
  and trainable parameters differ by design. Total training losses are not a
  common quality metric. One seed is a bounded comparison, not evidence of
  general superiority. Shared GPU compute invalidates isolated speed claims.

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True python benchmarks/train_eagle4_comparison.py \
  --target /path/to/pinned-minicpm5-1b --output /path/to/new-comparison \
  --steps 5000 --candidate-steps 1000 --train-samples 1024 --eval-samples 64 \
  --max-length 256 --memory-mib 6500
```

The runner never rents, stops processes or modifies the live serving
environment. Raw feature files, conversations, weights and operational SSH
details remain outside the PR. Final evidence belongs in an experiment report.

The runner now defaults to `--capture-source target`. The original raw-answer
run used the `reference` protocol. See the
[follow-up diagnosis and extension command](../research/eagle4-minicpm.md#diagnosis-and-extension)
for the confirmed mismatch, CUDA numerical replay, additional-update contract
and completed extension training. Capture progress commits per-record hashes so
a coordinated interruption can reuse verified records. Actual CUDA capture
resumed the 291-record prefix and completed 1,024/64 records. Extended exports
fail strict BF16 reference equality (56/64, 56/64, 53/64). The FP32 control passes
64/64 for every arm. Native EAGLE4 remains 7/8 and fails strict equality.
EAGLE4 uses CPU FP32 optimizer states with full CUDA dense forward/backward,
cap 6,200 MiB and reserve 768 MiB.
DSpark and DFlash2 retain fused GPU optimizers. Failed memory probes remain
in the experiment record; they do not count toward completed training.

## Additional GPU validation and implementation audit

The [2026-10-30 follow-up](../research/eagle4-minicpm.md#further-10000-updates-with-the-gpu-optimizer)
adds EAGLE4-only dense-head controls and extends its dense draft to 25,000 updates.
DSpark and DFlash2 retain their earlier 15,000-update results.

| Follow-up requirement | Actual result | Status |
| --- | --- | --- |
| Full GPU optimizer and exact resume | FP32 master/fused Adam; step-20 resume; 10,000 dense plus 1,000 candidate updates; all 16 dense tensors change | pass, bounded |
| Export and stage isolation | Reload exact; candidate fitting preserves all dense tensors | pass |
| Candidate versus full-head control | Both executed before/after with FP32 and BF16 target verification | pass, bounded; scoring precision differs |
| Primary acceptance improvement | FP32 candidate acceptance 2.052% → 1.927%; paired interval includes zero | no observed improvement |
| Strict target-greedy equality | FP32 64/64 before/after; BF16 53/64 before/after | BF16 fail |
| CUDA loss and gradient parity | One real seven-depth batch; all 16 dense gradients finite/nonzero within predeclared limits | pass, scoped |
| Official training equivalence | Source audit matches admitted dense model structure; author training revision unavailable; loss normalization and projection precision differ from public TorchSpec | unverified |
| New native serving and isolated speed | 25,000-update native export and serving benchmark not executed | not-run; earlier native strict failure remains |

The follow-up uses newly available capacity on the same GPU. This task changes
no shared service. Peak training allocation is 5,903.19 MiB under a 12,000 MiB
cap. The measured dense interval reaches 7.220 updates/second. This is training
speed for this recipe and does not establish serving speed or future quality.
The [aggregate receipt](../research/results/eagle4-minicpm-gpu-followup-20261030.json)
retains identities, all eight results, diagnostic intervals and audit limits.
