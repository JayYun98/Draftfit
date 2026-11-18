# MiniCPM5-1B: DSpark, DFlash2 and EAGLE4

## Scope

**Follow-up:** all three methods completed **10,000 additional dense updates**
on the corrected target-generated cache, reaching **15,000 cumulative dense
updates**. EAGLE4 then completed **1,000 candidate updates**. All exports reload
exactly. The strict BF16 reference fails for every arm. The FP32 control passes
64/64 for every arm. Native EAGLE4 equality remains 7/8, so required correctness
gates still fail.
The original 5,000-update results below remain historical evidence. See
[diagnosis and extension](#diagnosis-and-extension) and the
[latest receipt](results/eagle4-minicpm-extension-20261029.json).

This record tests Draftfit's EAGLE4 implementation against existing DSpark and
DFlash2 training. It uses spare memory on an existing shared A100-SXM4-80GB (driver 595.71.05).
No serving process was stopped and no GPU was rented.

EAGLE4 trains its complete dense draft for **5,000 updates**, then fits its
candidate factors for **1,000 additional updates**. Candidate-only fitting is
the second stage. It does not replace backbone training.

The [implementation specification](../specs/eagle4-minicpm.md) defines the
acceptance gates and upstream contract. The
[aggregate receipt](results/eagle4-minicpm-20261029.json) records source and
artifact identities. Raw conversations, feature tensors and weights are excluded.

## Original recipe selection

The [family guide](../DFLASH_FAMILY.md), [MiniCPM recipes](../RECENT_MODEL_EXPERIMENTS.md)
and [training search guide](../TRAINING_SEARCH.md) informed these starting recipes.
The published DSpark MiniCPM5-2B checkpoint cannot initialize the different 1B
geometry. All three drafts therefore start from scratch. These settings are
documented adaptations, not established optima for MiniCPM5-1B.

| Setting | DSpark | DFlash2 | EAGLE4 |
| --- | --- | --- | --- |
| Dense draft layers | 5 | 3 | 1 recurrent |
| Draft width / vocabulary | 1,536 / 130,560 | Same | Same |
| Target block taps | 1, 6, 11, 16, 20 | Same | 1, 11, 20 |
| Prediction geometry | Block 7, up to 512 random anchors | Same | Seven recurrent unrolls |
| Private heads | Markov rank 256, confidence | Selector rank 256, top 16; convolution kernel/group 2/16 | Full dense head; candidate rank/top 128/128 |
| Dense learning rate / warmup | 6e-4 / 4% | 6e-4 / 4% | 1e-4 / 1.5% |
| Gradient clipping | 1.0 | 1.0 | 0.5 |
| Dense updates | 5,000 | 5,000 | 5,000 |
| Extra candidate updates | 0 | 0 | 1,000 |

All arms use BF16 draft tensors, FP32 master parameters, fused GPU AdamW and
cosine scheduling. Every backbone parameter trains during dense fitting.
EAGLE4 freezes all dense parameters during candidate distillation; parameter
hashes check this boundary.

The target is `openbmb/MiniCPM5-1B` at
`87179e5c1f455ef22e6223592d2d61351b525bfc`.
The dataset is `HuggingFaceH4/ultrachat_200k` at
`8049631c405ae6576f93f445c6b8166f76f5505a`.
Training uses 1,024 conversations from `train_sft`; evaluation uses 64 from
`test_sft`. The runner selects the first usable conversations deterministically.
Maximum length is 256, batch size is one, and seed is 29. All methods reuse
the same source conversations, frozen target features and shuffled sample order.

Objectives, supervised prediction counts and parameter counts differ. Their
training losses cannot rank model quality. This single-seed, short-data run
does not establish general superiority. Earlier uniform one-layer trials used
a different optimizer and are excluded from the final comparison.

## Original 5,000-update execution and results

All three dense runs completed with finite loss and gradients. HF save/reload
matched exactly. EAGLE4's dense backbone changed during its 5,000 updates;
all dense parameter hashes stayed unchanged during the candidate stage.

| Arm | Dense trainable parameters | Dense / candidate updates | Elapsed seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| DSpark | 220,222,209 | 5,000 / 0 | 708.31 | 4,981.70 |
| DFlash2 | 167,559,936 | 5,000 / 0 | 658.05 | 3,977.23 |
| EAGLE4 | 239,872,512 | 5,000 / 1,000 | 920.65 | 5,903.19 |

EAGLE4's separate candidate stage trains 16,908,288 parameters. The receipt
records checkpoint and export hashes for each arm. Generation results follow
below; successful training does not establish lossless generation.

DSpark used a 6,500 MiB allocator cap with a 1,536 MiB initial reserve. An
external serving workload then grew. The next arm correctly stopped at its
initial memory guard. Remaining arms use a 6,200 MiB cap and 1,024 MiB reserve;
their model geometry, objectives and FP32 GPU optimizer remain unchanged.
The first EAGLE4 update exposed large temporary tensors in gradient-norm
calculation. A direct FP32 vector reduction removes those tensors. CUDA memory
regression and optimizer update checks passed. DSpark and DFlash2 used the
earlier algebraically equivalent reduction; source identities record this difference.
Training times include shared GPU contention and checkpoint work. They do not
measure isolated training throughput.

The source-only runner uses production draft models and losses. A separate
tiny CUDA gate executes the actual training, resume and export CLI. It compares
four uninterrupted candidate updates with two updates plus full resume.
The resulting tensors match exactly; dense tensors remain unchanged. HF export
reload also matches exactly. This lifecycle check does not establish MiniCPM quality.

## Generation and serving protocol

The common reference decoder uses the actual exports, 64 held-out prompts,
up to 64 new tokens and six draft proposals per round. It verifies proposals
with the frozen target head. Exact target-only greedy token equality is required
for every prompt. Proposal acceptance and committed tokens per round are
reported separately. Full-prefix recomputation is not an optimized serving benchmark.

The primary verifier uses BF16. A separate FP32 verifier control keeps proposal
networks in BF16. It diagnoses numerical path differences without changing the
primary BF16 pass criterion. Candidate-head diagnostics measure retention of the
dense draft's argmax on teacher-forced supervised root states, not recurrent
generation quality.

Native serving uses the separate EAGLE-HARNESS runtime with patched vLLM.
Both target-only and speculative arms use CUSTOM attention, eager execution,
context 512, one active sequence and a 128 MiB KV cache. The static tree has
depth three, node top-k two and maximum verification width eight. Eight held-out
prompts generate up to 32 tokens. This probe checks loading and greedy equality.
It performs no calibration or isolated latency/throughput measurement.

### Primary BF16 reference result

**Fail:** no arm met the required 64/64 target-greedy equality criterion.
All generated 4,096 tokens over 64 prompts. Both target EOS IDs, `1` and
`130073`, were enabled through the target generation configuration.

| Arm | Accepted / proposed tokens | Acceptance | Committed tokens / round | Exact greedy prompts |
| --- | ---: | ---: | ---: | ---: |
| DSpark | 403 / 10,568 | 3.813% | 2.197 | 55 / 64 |
| DFlash2 | 392 / 10,611 | 3.694% | 2.190 | 59 / 64 |
| EAGLE4 | 134 / 11,342 | 1.181% | 2.050 | 57 / 64 |

These values describe this full-prefix reference and these checkpoints.
Correctness failure prevents a lossless-generation or speedup claim.
A previous diagnostic reproduced BF16 target-prefix versus block-verification
numerical differences. It does not explain every final-run mismatch.

EAGLE4's candidate set retained the dense argmax at 9,384 of 9,406
teacher-forced supervised positions (99.766%). Rescoring agreed with the dense
GEMM argmax at 9,323 positions (99.118%). Candidate retention alone does not
establish recurrent draft quality or exact BF16 GEMM equivalence.

### FP32 verifier control

**Pass, diagnostic scope:** all three arms match target-only greedy generation
on 64/64 prompts. Proposal networks and exports stay BF16; target verification
uses FP32 tensors and the same pretrained weights. All arms generate
4,096 tokens. The EOS list and held-out prompts match the primary gate.

| Arm | Accepted / proposed tokens | Acceptance | Committed tokens / round | Exact greedy prompts |
| --- | ---: | ---: | ---: | ---: |
| DSpark | 392 / 10,609 | 3.695% | 2.192 | 64 / 64 |
| DFlash2 | 381 / 10,642 | 3.580% | 2.185 | 64 / 64 |
| EAGLE4 | 141 / 11,325 | 1.245% | 2.054 | 64 / 64 |

This control supports numerical-path sensitivity as a contributor to BF16
mismatches. It does not certify BF16 correctness or identify every mismatch's cause.
EAGLE4 has lower proposal acceptance in both controls for these checkpoints.
The run does not establish a tuned method ranking or general quality advantage.

### Native serving

**Fail:** both arms completed eight requests and generated 256 tokens each.
The trained EAGLE4 export loaded its actual candidate weights and native kernels.
Seven of eight token sequences matched the target-only baseline; one differed.
The probe returned exit code 1 and retained the strict 8/8 criterion.

The cause of this native mismatch remains unresolved. The reference FP32 pass
does not qualify this different BF16 runtime. Native latency, throughput,
peak memory and run-to-run variation were not measured. Calibration, production
serving and speedup claims remain blocked by correctness and measurement gaps.

## Local verification and limits

- Affected model, provider, configuration and export checks: 70 tests ran,
  five skipped and none failed on CPU. The BF16 candidate-row regression failed
  before its fix. Candidate products now accumulate in FP32 before casting to BF16.
- Optimizer and scheduler checks: 14 tests, two skipped and none failed on CPU.
- Full public CPU suite: 770 tests, one failure, 53 skips. The unchanged
  `FamilyConcurrencyTest.test_exit_zero_without_eval_is_failure` returned
  `cancelled` instead of `failed` under concurrent workers. It passed standalone.
- Wheel and source distribution build passed. Installed-wheel onboarding passed
  for DSpark, DFlash, DFlash2 and EAGLE4, including asset export.
- The full hash-locked CPU installation failed on the macOS Torch wheel hash.
  Hash checking was preserved. CPU execution used an alternate overlay with
  Torch 2.11.0 and hash-checked Transformers 5.12.1. This is not a passed full
  locked-profile provisioning gate.
- Actual registered CUDA capture, disk reader, variable-length collation, dense
  optimization and candidate optimization passed.

Vocabulary pruning, compact teacher mode, alternate draft blocks and training
attention backends other than SDPA are excluded. The separate external runtime
retains its license. This independent implementation does not reproduce a
publicly pinned author training bundle. This campaign trains from saved features.
EAGLE4 server streaming reuses the registered EAGLE contract but has no real
online-server check in this campaign.
Native DSpark/DFlash2 serving, SGLang serving, calibration, multi-GPU training
and production speed remain unqualified.

## Runtime identity

Training uses Python 3.12.3, Torch 2.13.0+cu130, Transformers 5.17.0,
Datasets 4.4.2 and Accelerate 1.15.0. The task environment reads existing
training dependencies and adds only its own packages. The shared runtime stays
unchanged.

The separate native environment uses Torch 2.14.0+cu130, Transformers 5.19.0,
vLLM 0.32.0rc1.dev283+eagle.4, EAGLE-HARNESS 0.1.3 and Triton 3.8.0.
Its dependency closure passed `uv pip check`. Package versions are retained;
a complete immutable engine image or downloaded-wheel digest was not retained.
This is a bounded local runtime check, not a published reproducible runtime profile.

Training source archives have separate hashes because fixes landed between arms.
The EAGLE4 archive matches implementation commit
`2cfae1a6f8dff3136fbfb3836190791bb9abc49e` across all 286 Python source files.
The final evaluator adds only the generation-EOS correction from
`127a33d1b05b106d71006fb77a8f4f12f7304ac1`. The receipt records its file hash.

## Reproduce

Use the [environment instructions](../ENVIRONMENT.md), a pinned local target
and new output directories. Preserve the existing patched serving environment.

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True python benchmarks/train_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/comparison \
  --steps 5000 --candidate-steps 1000 --memory-mib 6200 --reserve-mib 1024 \
  --capture-source reference
python benchmarks/check_eagle4_cli.py --output /path/to/new-cli-gate
python benchmarks/evaluate_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/comparison \
  --memory-mib 6200 --reserve-mib 1024
python benchmarks/evaluate_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/comparison \
  --memory-mib 6200 --reserve-mib 1024 \
  --verifier-dtype float32 --result-dir evaluation-fp32
```

In the separate patched engine environment:

```sh
python benchmarks/serve_eagle4_probe.py --target /path/to/minicpm5-1b \
  --comparison /path/to/comparison --arm target
python benchmarks/serve_eagle4_probe.py --target /path/to/minicpm5-1b \
  --comparison /path/to/comparison --arm eagle4
```

## Diagnosis and extension

The [follow-up receipt](results/eagle4-minicpm-diagnosis-20261029.json) retains
aggregate evidence. The original artifacts and results above remain unchanged.

### Data protocol mismatch

The original cache has 1,024 training answers containing 163,372 assistant tokens.
It has no `<think>` or `</think>` tokens. All 64 cached held-out answers also
omit these tokens. Every target-only generation starts with `<think>` (token 8).
The draft therefore trains on raw answers and encounters generated reasoning
during evaluation. This confirms a protocol mismatch. Its contribution to the
low acceptance remains unmeasured.

The comparison runner now defaults to greedy target-generated continuations.
It captures features after generation using the same tokenizer and target.
`--capture-source reference` retains the original raw-answer control.
Warm starts check source conversation identities and exact prompt token IDs.
All three methods consume the same new feature cache.

### Recurrent layout and numerical controls

An actual tiny Llama target and EAGLE4 draft produce matching training and
proposal logits at three recurrent depths, including sliding-window masking.
This checks FP32 token/feature alignment. CUDA BF16 and native execution require
their separate gates.

The retained BF16 failure reproduces on final held-out sample 7: EAGLE4 equality
is 0/1, with 3/174 accepted proposals. The first difference is generated token 26.
CPU FP32 replay selects token 7590 for both prefix lengths; its top-two gap is
0.01267. CPU BF16 rounds tokens 7590 and 14585 to the same logit, 16.125.
Both CPU BF16 shapes select 7590. These controls show near-tie sensitivity;
the later CUDA BF16 replay reproduces the exact flip: the canonical prefix
selects 14585 and block verification selects 7590. Each top-two gap is 0.125.
The hidden-state maximum difference is 0.75; changing only the head scoring
shape gives zero difference. This identifies target prefix-length sensitivity
for this specific retained failure. Strict greedy equality remains failed.
Native sample 3 first differs at token 13. The later native replay reproduces
that difference and records a target top-two logprob gap of 0.125.

Upstream [test results](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/support-matrix.md#how-these-were-measured)
report exact agreement, near ties and FP32 controls separately. Its near-tie
rule uses a no-speculation logprob gap of at most 0.13. Larger gaps receive
additional FP32 checks. This policy differs from our strict token-equality gate.
The upstream's published BF16 shape differences provide a relevant diagnosis
pattern; they do not establish the complete cause of our native mismatch.

The native probe now saves token logprobs and first-mismatch details, following
the [upstream diagnosis guidance](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/new-model.md#check-generation-before-calibration).
The updated probe has now run on both original and extended exports.
Strict equality gates remain unchanged.

### Additional 10,000 dense updates

The prepared extension loads each completed 5,000-update dense export.
EAGLE4 loads `dense-export`, then trains the complete dense draft for another
10,000 updates. A separate 1,000-update candidate stage follows.
Each arm starts a fresh optimizer and cosine schedule. This is a model warm
start, consistent with the [family guide](../DFLASH_FAMILY.md#existing-draft-identify-the-source-first).
It does not resume the completed schedule at zero learning rate.
The original dense EAGLE4 optimizer was superseded by candidate fitting.

GPU preflight loaded all three actual source exports into the fresh production
model factories. Its first attempt rejected a serialized full-vocabulary default:
`draft_vocab_size=None` becomes `130560` when loaded. The comparison now
canonicalizes that documented default; genuine vocabulary and same-shape RoPE
changes remain rejected. All three GPU warm starts then passed.

Six comparison regressions passed; the affected suite ran 57 tests with 17 skips
and no failures. Capture interruption/recovery uses a real tiny target on CPU,
verifies committed record hashes and preserves uncommitted files for diagnosis.
The raw-answer capture regression failed before the correction and passed after it.
These checks do not establish extended MiniCPM quality or GPU correctness.

Corrected MiniCPM capture resumed the verified **291-record prefix** on CUDA
and completed **1,024 training and 64 test records**. The coordinated FinBench
hold was released before this continuation. Original prompts and source sample
identities remain fixed. The completed capture manifest SHA256 is
`4a2d361c6ff3577bcd2fb7e15360db9933d1337d9adba97900b9c6ed04edce9d`.
The final CPU audit verifies all 1,088 record hashes, original prompts, source
identities and masks. All 1,024 training and 64 test continuations begin with
`<think>`. They contain 371 and 15 `</think>` tokens respectively. Supervised
positions total 179,505 in training and 10,687 in test. Exact prompt sequences
and source sample IDs have zero overlap between splits.

### Completed extension training

| Arm | Additional / cumulative dense updates | Candidate updates | Changed dense tensors | Peak allocated MiB | Export reload |
| --- | ---: | ---: | ---: | ---: | --- |
| DSpark | 10,000 / 15,000 | 0 | 62 | 4,981.70 | Exact |
| DFlash2 | 10,000 / 15,000 | 0 | 51 | 3,977.23 | Exact |
| EAGLE4 | 10,000 / 15,000 | 1,000 | 16 | 4,044.24 | Exact |

All three runs completed with finite loss and gradients. EAGLE4's dense
parameters remain unchanged during its candidate stage. DSpark and DFlash2
use fused GPU FP32 master AdamW. EAGLE4 uses the existing CPU optimizer
offload: FP32 masters and Adam states reside on CPU; full dense forward and
backward computation remains on CUDA. Full vocabulary, width, seven recurrent
unrolls, objective, clipping and learning-rate schedule are unchanged.

The allocator cap stays at **6,200 MiB**. A shared workload increase initially
left the 1,024 MiB reserve short by 68 MiB before any optimizer update.
Subsequent runs use a 768 MiB reserve and a separate 512 MiB CUDA startup
allowance. No shared service was stopped or resized.

EAGLE4's full GPU optimizer exceeded the cap during backward. The first
bounded repair used CPU optimizer offload and saved a verified 20-update
checkpoint. An activation-only offload alternative passed exact CUDA BF16
loss and all-gradient parity, but failed a 766 MiB allocation during the GPU
optimizer step. That alternative was reverted. The final run resumed the
verified 20-update CPU-offload checkpoint. Earlier uncheckpointed partial
updates were discarded; repeated metric segments do not count as completed
updates. No further memory tuning or automatic restart occurred.

The resumed EAGLE4 process reported 14,856.06 seconds, including candidate
fitting and export. Its initial 20-update probe and discarded work are separate.
DSpark and DFlash2 recorded 1,295.47 and 1,464.70 seconds. CPU state placement and shared GPU contention differ, so these times
do not establish comparative training speed. Executed training source:
`6eff3f39e456731429683cd45a345a17cdb3dcaf`.

To run EAGLE4's corrected extension in a separate output directory:

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True python benchmarks/train_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/new-extension \
  --warm-start /path/to/original-comparison --capture-source target \
  --steps 10000 --candidate-steps 1000 --memory-mib 6200 --reserve-mib 768 \
  --algorithm eagle4 --optimizer-cpu-offload
```

The offload flag changes optimizer placement for every selected arm. The recorded
DSpark and DFlash2 runs omit it; the EAGLE4 command selects only EAGLE4.
Use the same verified capture for every arm and the separate patched engine
for native serving. The completed reference and native measurements follow below.

New data and a new schedule change together. This extension alone cannot
isolate their effects. The same 64 evaluation prompts have already been
inspected; this is a diagnostic comparison, not a new untouched holdout.

### Extended-export reference results

All arms generated 4,096 tokens on the same 64 prompts, using up to 64 new
tokens and six draft proposals. Exact BF16 equality remains a required failed
gate. The FP32 target control keeps the proposal models in BF16 and passes
64/64 for every arm. Both target EOS IDs remain enabled.

| Verifier | Arm | Accepted / proposed | Acceptance | Committed tokens / round | Exact greedy prompts |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | DSpark | 1,505 / 7,448 | 20.207% | 3.110 | 56 / 64 |
| BF16 | DFlash2 | 1,413 / 7,659 | 18.449% | 3.003 | 56 / 64 |
| BF16 | EAGLE4 | 219 / 11,096 | 1.974% | 2.096 | 53 / 64 |
| FP32 | DSpark | 1,475 / 7,508 | 19.646% | 3.080 | 64 / 64 |
| FP32 | DFlash2 | 1,395 / 7,745 | 18.012% | 2.992 | 64 / 64 |
| FP32 | EAGLE4 | 227 / 11,065 | 2.052% | 2.101 | 64 / 64 |

The extension improves observed proposal acceptance relative to the original
checkpoints. EAGLE4 remains substantially below both baselines in this scope;
this recipe does not establish competitive draft quality. Data and schedule
changes remain confounded. The numerical control supports precision-path
sensitivity but does not explain every BF16 difference or certify lossless BF16
serving. Full-prefix timing is not optimized serving throughput.

The extended candidate set retains the dense argmax at 10,606 of 10,687
teacher-forced supervised positions (99.242%). Rescoring agrees at 10,513
positions (98.372%). This high root-state coverage does not establish recurrent
generation quality. The original raw-answer coverage uses different positions
and remains a separate historical measurement.

### Extended native serving and retained failure replay

**Fail:** native target-only and extended EAGLE4 each complete eight requests
and generate 256 tokens. Exact equality remains **7/8**. The actual extended
export loads its trained candidate factors and native kernels. Static tree,
CUSTOM attention, eager execution, context 512 and the 128 MiB KV cache stay
fixed. The complete strict-failure report returns exit 1.

The bounded original-export replay uses sample 3 and up to 16 tokens. It
reproduces the original first divergence at token index 13. The new eight-prompt,
32-token run reproduces the same divergence. Target-only logprobs place its
chosen token first and the speculative token second, separated by 0.125.
The speculative output reports a tie between its selected token and another
candidate. This is consistent with numerical sensitivity, but no native FP32
control establishes the complete cause. The upstream near-tie threshold is
not used to turn either strict failure into a pass.

No calibration, repeated isolated latency/throughput benchmark, native
DSpark/DFlash2 comparison or production qualification was performed. The
extension improves acceptance in this reference workload but does not meet
the required BF16/native correctness contract or show EAGLE4 quality parity
with the existing baselines.

## Further 10,000 updates with the GPU optimizer

The [GPU follow-up receipt](results/eagle4-minicpm-gpu-followup-20261030.json)
records this EAGLE4-only extension from 15,000 to 25,000 dense updates.
DSpark and DFlash2 retain their earlier 15,000-update measurements. They were
not retrained in this follow-up, so this is not a common 25,000-update comparison.

### Fixed inputs and training scope

The target, data revision, 1,024 training prompts, 64 diagnostic prompts,
maximum training length 256 and seed 29 remain fixed. Recovery regenerated
the deleted feature cache. All 1,088 feature tensors match the retained inputs.
Of these, 797 serialized files match directly; 291 match after reproducing
their historical serialization names. The new capture manifest records its
actual hashes. Training and test source identities and prompts remain disjoint.

The run loads the completed 15,000-update **dense export**, resets the optimizer
and starts another 10,000-update cosine schedule. All 239,872,512 dense
parameters train. A separate 1,000-update candidate stage follows and preserves
every dense tensor. Architecture, vocabulary, seven recurrent unrolls,
learning rate, objective and clipping remain unchanged.

Available GPU capacity changed after another task released its serving workload.
This task did not stop or change a shared service. FP32 masters and fused Adam
states now reside on the A100 GPU. The allocator cap is 12,000 MiB with a
2,048 MiB reserve. The earlier 6,200 MiB failures remain failures; they are not
relabelled as successful memory repairs.

A 20-update checkpoint passed before exact optimizer/RNG resume to 10,000
dense and 1,000 candidate updates. All 16 dense tensors changed. Candidate
fitting preserved the dense tensors, and final export reload matched exactly.
Peak allocated memory was 5,903.19 MiB. The resumed process reported
1,521.36 seconds, including candidate fitting and export; the initial probe
is separate. CPU optimizer placement and the shared workload differ from the
earlier run, so these times do not isolate an optimizer-placement effect.

### Diagnostic controls

The evaluator now supports `--algorithm eagle4` and
`--eagle4-head candidates|dense`. Its default candidate path remains unchanged.
Both modes share the target verifier, recurrent states, masks and six-proposal
chain. Dense scoring uses BF16 full-head matrix multiplication. Candidate
scoring uses FP32 selected-row accumulation before BF16 conversion. This control
therefore changes both selection and scoring numerics.

Before training, candidate-mode results reproduce the retained baseline exactly:
FP32 accepts 227/11,065 proposals (2.052%) with 64/64 exact outputs; BF16
accepts 219/11,096 (1.974%) with 53/64 exact outputs. Dense mode accepts
221/11,082 (1.994%) in FP32 and 214/11,109 (1.926%) in BF16.
Removing candidate scoring alone does not resolve the low acceptance.

Acceptance counts all proposals, including discarded suffixes. It is not
single-token prediction accuracy or generated-output correctness. In the
baseline FP32 candidate trace, the first proposal is accepted at 224/1,921
eligible prefixes (11.661%); the second is accepted at 3/222 (1.351%).
High teacher-forced candidate coverage does not establish recurrent quality.

Both checkpoints use the same 64 previously inspected prompts, 64 new tokens
and six proposals per round. FP32 requires strict 64/64 target-greedy equality.
BF16 keeps the same strict criterion. Paired prompt bootstrap intervals use
5,000 resamples and seed 29. These are diagnostic intervals on reused prompts,
not fresh-holdout evidence or method-superiority tests.

To evaluate an EAGLE4 export through its full head:

```sh
python benchmarks/evaluate_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/comparison \
  --algorithm eagle4 --eagle4-head dense --verifier-dtype float32 \
  --samples 64 --max-new-tokens 64 --memory-mib 12000 --reserve-mib 2048 \
  --result-dir evaluation-fp32-dense
```

### Completed 25,000-update results

All eight runs completed. Each generated 4,096 tokens on the same 64 diagnostic
prompts. BF16 runs retain exit 1 because strict target-greedy equality failed.

| Verifier / proposal head | Before accepted / proposed | After accepted / proposed | Acceptance before → after | Exact prompts before → after |
| --- | ---: | ---: | ---: | ---: |
| FP32 / candidates | 227 / 11,065 | 214 / 11,107 | 2.052% → 1.927% | 64/64 → 64/64 |
| FP32 / dense | 221 / 11,082 | 211 / 11,116 | 1.994% → 1.898% | 64/64 → 64/64 |
| BF16 / candidates | 219 / 11,096 | 217 / 11,102 | 1.974% → 1.955% | 53/64 → 53/64 |
| BF16 / dense | 214 / 11,109 | 219 / 11,096 | 1.926% → 1.974% | 53/64 → 53/64 |

**No observed improvement in the primary FP32 candidate comparison.** The
acceptance difference is −0.1248 percentage points. Its paired diagnostic 95%
interval is [−0.3548, +0.1030] percentage points. This includes zero and does
not establish a population regression. Dense scoring also shows no observed
improvement. These results close this bounded continuation recipe.

The new FP32 candidate trace accepts its first proposal at 208/1,927 eligible
prefixes (10.794%). The second accepts 6/206 (2.913%); the third accepts 0/5.
Teacher-forced candidate retention reaches 10,643/10,687 (99.588%). Candidate
rescoring agrees with dense argmax at 10,536/10,687 positions (98.587%). The
measured weakness remains dense/recurrent proposal quality in this workload.
These diagnostics do not identify its complete cause.

The steady dense interval covers 9,979 updates in 1,382.05 seconds:
7.220 updates/second, or 0.13850 seconds/update. It includes periodic checkpoint
work. At that rate, 10,000 updates take about 23.1 minutes. A linear 100,000-update
estimate is about 3.85 hours, excluding capture, candidate fitting and evaluation.
The latter estimate was not executed and predicts no quality improvement.
Native serving with the 25,000-update export, isolated serving performance,
new-seed replication and fresh-holdout quality remain **not-run**. The earlier
native 7/8 strict failure remains a failure.

### Official source and CUDA implementation audit

The follow-up audit pins EAGLE-4 release 0.1.4 at
[`3e20798`](https://github.com/eagle-hq/EAGLE-4/tree/3e20798bcce810486cd810ecf087936488106ae2)
and public TorchSpec at
[`c4fc3e0`](https://github.com/lightseekorg/TorchSpec/tree/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9).
This source inspection is separate from the executed Draftfit checks.

| Suspected defect | Evidence and scoped conclusion |
| --- | --- |
| Candidate-only training | Contradicted: all 239,872,512 dense parameters train and all 16 dense tensors change. Candidate fitting follows separately. |
| Wrong normalization, residual or candidate structure | The admitted dense architecture matches the inspected [SGLang model](https://github.com/eagle-hq/EAGLE-4/blob/3e20798bcce810486cd810ecf087936488106ae2/eagle_harness/sglang/models/llama_eagle4.py). This is source evidence, not engine parity. |
| Wrong token shift or recurrent feedback | Existing real tiny-target tests align training and generation logits at three depths. This establishes internal consistency, not full upstream runtime equivalence. |
| Candidate filtering causes the low rate | Full-vocabulary controls remain near 2%; removing candidate scoring does not resolve the measured weakness. Scoring numerics also differ. |
| Broken CUDA cross-entropy or backward | A real seven-depth MiniCPM batch passes the predeclared loss and all-gradient parity checks below. |
| Exact author training reproduction | Unverified. The official [new-target guide](https://github.com/eagle-hq/EAGLE-4/blob/3e20798bcce810486cd810ecf087936488106ae2/docs/new-model.md) explicitly lacks a reproducible revision of its inspected training bundle. |

The CUDA diagnostic compares the fused loss with standard PyTorch on identical
inputs. Direct FP32 and BF16 tests use vocabulary size 130,560, binary masks and
a non-unit upstream gradient. Both pass. The actual 25,000-update export then
runs the seven-depth training path on one retained 256-token training example.
No optimizer update runs. Fused loss is 2.51030135; reference loss is 2.51030183.
All 16 dense gradients are finite and nonzero. Maximum per-tensor relative L2
error is 0.4719%; maximum absolute error is 0.00012208. These pass the declared
1% and 0.01 limits. Peak allocation is 3,993.45 MiB.

The diagnostic first failed because its script used the invalid stage name
`dense`. Its log and result remain preserved. One manual correction to `draft`
produced the parity result. This harness failure is separate from training and
is not evidence of an algorithm defect.

Two training differences remain against the pinned public TorchSpec source.
Its [loss path](https://github.com/lightseekorg/TorchSpec/blob/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9/torchspec/models/eagle3.py)
normalizes by valid-token count. Draftfit's existing EAGLE loss normalizes by
full sequence positions, so supervised fractions affect gradient scale.
TorchSpec's [feature projection](https://github.com/lightseekorg/TorchSpec/blob/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9/torchspec/models/draft/llama3_eagle.py)
also accumulates in FP32 by default; this Draftfit recipe projects in BF16.
Their effects on acceptance have not been isolated. Neither difference is
proven to cause this result, and this audit does not certify the whole model.
No Qwen comparison or 100,000-update extension was started.
