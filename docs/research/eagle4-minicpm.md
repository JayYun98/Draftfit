# MiniCPM5-1B: DSpark, DFlash2 and EAGLE4

## Scope

**Follow-up:** the original cache uses raw answers, while MiniCPM generates
reasoning continuations. See [diagnosis and extension](#diagnosis-and-extension)
before interpreting the generation acceptance figures. The corrected 10,000-update
extension has not trained yet. Corrected capture reached 291/1,024 training
samples before a coordinated main FinBench GPU hold. The capture files are
verified and preserved; no automatic restart is active. See the
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

## Recipe selection

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

## Execution and results

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
Native sample 3 first differs at token 13; its diagnosis remains pending.

Upstream [test results](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/support-matrix.md#how-these-were-measured)
report exact agreement, near ties and FP32 controls separately. Its near-tie
rule uses a no-speculation logprob gap of at most 0.13. Larger gaps receive
additional FP32 checks. This policy differs from our strict token-equality gate.
The upstream's published BF16 shape differences provide a relevant diagnosis
pattern; they do not classify our unmeasured native mismatch.

The native probe now saves token logprobs and first-mismatch details, following
the [upstream diagnosis guidance](https://github.com/eagle-hq/EAGLE-4/blob/3b96ef6e1c0b2ef0c698fab74c8e383d8bf8053d/docs/new-model.md#check-generation-before-calibration).
This updated native probe has not run. Strict equality gates remain unchanged.

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

Corrected MiniCPM capture produced **291 training samples**. All begin with
`<think>`; they contain 51,016 supervised positions and 107 `</think>` tokens.
Every preserved prompt matches the original source, and all masks, feature
finite values and file hashes passed CPU recovery checks. A separate recovery
folder retains these records without modifying the interrupted output.

**Paused:** the main FinBench precision comparison requested temporary GPU use.
The task's capture and driver exited, GPU memory and the shared reservation were
returned, and the original service identities were preserved. Additional dense
updates completed: **zero**. GPU capture resume, remaining capture, additional
training, new generation and native gates remain unrun. An explicit coordination
release and fresh service/lease checks are required before restarting.

After that release, use a new output directory or the verified recovery directory:

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True python benchmarks/train_eagle4_comparison.py \
  --target /path/to/minicpm5-1b --output /path/to/new-extension \
  --warm-start /path/to/original-comparison --capture-source target \
  --steps 10000 --candidate-steps 1000 --memory-mib 6200 --reserve-mib 1024
```

This produces 15,000 cumulative dense updates per arm. New data and a new
schedule change together; this extension alone cannot isolate their effects.
Repeat the reference and native correctness gates before any performance claim.
The same 64 evaluation prompts have already been inspected; the extension is a
diagnostic comparison, not a new untouched holdout.
