# Recent target training recipes

These are experimental recipes, not trained artifacts or certified targets.

| Target | Immutable revision | DSpark | DFlash2 |
| --- | --- | --- | --- |
| openbmb/MiniCPM5-2B | f97400052a43d642bbc6e9975e2397e3ae6a6b52 | Official draft warm start | Official DSpark retarget or scratch draft |
| XingChen-AGI/Xing4.0-29B-A4B | baae3c3e813cad5f888f1f485cfff659c89076c5 | New draft | New YaRN draft |
| TaichuAI/ZDTaichu5.0-9B | bc125a9819a421755f4dcebe498eb81c139b65a3 | New draft or separate official DSpark warm-start candidate | New draft |
| Altworld/Hemmingway-1 | b987f1800da287b3016862948743a253fbf80553 | New draft | New draft |
| Agnes-AI/Agnes-3.0-Flash | 891ce4f9ffb89b22888aa7fcc2bb2f3618867684 | New draft | New draft |

The five config prefixes are `minicpm5-2b`, `xing4-29b-a4b`, `zdtaichu5-9b`,
`hemmingway-1`, `agnes3-flash`; append `-dspark.json` or `-dflash2.json`.
All ten files were constructed on the meta device and passed `target prepare`
against their pinned target metadata. This validates shapes, not original weights.

The new route is **text-only Transformers capture**, shared with the owned online
teacher. It does not add these models to vLLM/SGLang capture support. Qwen3.5
hybrid uses its actual Transformers decoder; Agnes and Taichu unwrap their text
decoder; Xing taps explicitly average its hyperconnection axis, matching its
final readout. Serving must implement the same Xing tap contract.

New drafts use full attention and full-dimensional scalar text RoPE. Hybrid
target mRoPE is not copied into the draft. These are independently trained
architectures, not conversions of target attention or existing MTP heads.
Xing drafts retain YaRN factor 64 through Transformers' rotary implementation.
New-draft depth/taps/mask IDs are experiment choices requiring tokenizer and
quality verification. The official MiniCPM draft retains its own values.

Taichu's official 3.21B `speculators` checkpoint cannot warm-start the original
full-attention new-draft recipe: it also contains embedding/head tensors,
mixed sliding attention and sample-from-anchor semantics. A separate official
config adapter and corrected DSpark candidate now exist in the
[Taichu/LFM preparation plan](RECENT_MODEL_NEXT_TARGETS.md); actual full-weight
load, capture and numerical equivalence remain unverified. Do not discard those
tensors or claim numerical equivalence.
Image/video training and MTP-head training are also outside this text route.

Official MiniCPM5 draft: `openbmb/MiniCPM5-2B-DSpark` at
`114a20fdbf53220712c7fbdd7dccddbf1dedebb4`.
Source: https://huggingface.co/openbmb/MiniCPM5-2B-DSpark
Xing source: https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B

## MiniCPM5 preparation

`configs/minicpm5-2b-dspark.json` adapts the official config: registered
architecture name, nested DSpark options, explicit RoPE theta. Its 5 layers,
block size 7, taps [1,10,20,30,39], mask 75982 and Markov/confidence heads
are preserved. All 62 official safetensors header keys/shapes match a meta-device
instance exactly, with no extra local tensors. This does not establish numerical
parity or actual weights loading. The official checkpoint omits target embeddings;
the trainer must obtain them from the pinned target.

Download the official checkpoint at the exact revision into a local snapshot
before warm start. Substitute that existing directory below:

```sh
draftfit target prepare openbmb/MiniCPM5-2B \
  --revision f97400052a43d642bbc6e9975e2397e3ae6a6b52 \
  --strategy dspark --teacher-backend transformers \
  --draft-config configs/minicpm5-2b-dspark.json \
  --draft-checkpoint /path/to/pinned-official-draft \
  --hidden-states ./features/minicpm5-dspark --output-dir ./runs/minicpm5-dspark

draftfit target prepare openbmb/MiniCPM5-2B \
  --revision f97400052a43d642bbc6e9975e2397e3ae6a6b52 \
  --strategy dflash2 --teacher-backend transformers \
  --draft-config configs/minicpm5-2b-dflash2.json \
  --hidden-states ./features/minicpm5-dflash2 --output-dir ./runs/minicpm5-dflash2
```

DFlash2 in this original recipe uses a fresh 3-layer draft, block 7, and mask
75982; it does not load DSpark weights. The family matrix below instead uses
the five-layer official DSpark backbone as an explicit alignment-retraining
initialization. These recipes have different taps, so keep their capture
directories separate.
Depth/taps are starting hypotheses, not optimized settings. Follow ENVIRONMENT.md
for feature capture; inspect the actual tokenizer's assistant mask before choosing
a renderer. A model card's tool support does not enable tool-call training here.

## Xing prerequisites

40 layers, hidden size 3584, 64 routed experts, MLA and `hc_mult=4` require
inspection of actual intermediate feature layout and reduction boundaries.
The HF adapter implements this reduction; vLLM remains unsupported here. Review pinned
custom model code and validate real capture before allocating extended training.
Do not pretend it is ordinary Llama or flatten hyperconnection states implicitly.

The DFlash2 implementation now supports default RoPE and YaRN. The source
target uses YaRN factor 64; the new draft preserves this scaling.
DSpark metadata preparation alone does not establish equivalent position semantics.

## Gates and observations (2026-09-30)

Local Python 3.12 / Torch 2.13.0 / Transformers 5.14.1:
- MiniCPM DSpark and generated DFlash2 metadata preparation passed.
- Official DSpark header vs owned model: 62/62 tensor keys and shapes matched.
- Fixed generic metadata initialization crashing on YaRN before fields were set;
  target project suite: 12 tests passed, including scaling-preservation regression.
- Both Xing methods now prepare and construct; YaRN rotary outputs match the
  Transformers reference at positions 0, 127, 128 and 4096 and config roundtrips.
- Actual tiny random Qwen3.5 hybrid teacher → capture → disk → common loader →
  DSpark and DFlash2 gradient/weight updates passed. This is not original-model evidence.
- MiniCPM official-style config normalization and exact synthetic checkpoint
  warm-start roundtrip passed; original 62-tensor checkpoint header also matches.
- Original MiniCPM DSpark weights now load on CPU: 62/62 tensors. With explicit
  anchor retarget, DFlash and DFlash2 receive 58/58 shared tensors exactly;
  DFlash2 initializes 23 additional tensors. Their HF-format exports reload
  with 58/58 and 81/81 tensor equality, respectively. These checks do not
  exercise a training update, tokenizer mask, target feature parity or serving.
- The pinned official `model.safetensors` SHA256 is
  `ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97`.
  The integrated CPU gate passed 399 tests (8 skipped); the wheel built and
  its isolated installed `draftfit family --help` command worked. This is
  packaging and logic evidence, not a GPU result.
- At this initial CPU phase, tokenizer masks, teacher feature parity, real
  training, resume and serving were not run. The later GPU gate below covered
  masks, feature capture and short training; teacher parity, exact resume and
  serving remain open.

Next run: fixed reviewed conversation split; 20-step integration smoke, optional
50–100 steps after successful update/resume/export checks. Compare MiniCPM official
draft vs additional training, and target-only vs each exported method on the same
held-out prompts. Loss change is training evidence, not a speedup claim.

## MiniCPM5 single-GPU family validation plan (2026-09-30)

This is a proposed run, not GPU evidence. Use one on-demand GPU and sequential
offline Transformers capture/training. The online teacher/trainer split needs a
different resource plan. The family runner, warm start, DSpark ablations and
metrics are integrated in this Draftfit branch. One command trains independent
drafts sequentially from shared features; it does not create one checkpoint
that serves every architecture.

### Before rental

1. Keep family jobs on one target revision and one pair of disjoint offline
   feature directories. Verify `family --plan`, checkpoint tensor transfer and
   export reload before renting.
2. Pin final target `openbmb/MiniCPM5-2B@f97400052a43d642bbc6e9975e2397e3ae6a6b52`,
   SFT target `openbmb/MiniCPM5-2B-SFT@3e0690c57f6ac772180498467c82e56170cc00ac`,
   and official draft `openbmb/MiniCPM5-2B-DSpark@114a20fdbf53220712c7fbdd7dccddbf1dedebb4`.
   Official draft weights load locally; target embedding injection and the
   trained export reload still need real checks. Final and SFT have the same tokenizer/config blobs but different
   weight files and chat templates; this is a controlled target-weight shift,
   not a measured acceptance regression.
3. Use one final-model chat template for both targets and record the identical
   input token IDs. The published templates omit `{% generation %}`, so the
   current `hf` assistant-mask parser cannot be assumed to work. Inspect the
   `qwen` renderer on structured MiniCPM examples or provide reviewed
   pretokenized `input_ids`/`loss_mask`; assert assistant-only nonempty masks
   before capture. Do not silently substitute a different renderer.
4. Prepare 120 train and 60 held-out, prompt-disjoint text tasks, balanced
   across general, math and code. Use the SFT target to generate reviewed
   training responses. Freeze prompt split, seed, template, max length (start
   at 512), BF16, taps `[1,10,20,30,39]`, block 7, full vocab 130560 and mask
   75982. Capture a DSpark superset sidecar with `target_last_hidden_states`
   once for SFT train/holdout; the aligned DFlash/DFlash2 jobs consume it as
   a checked subset. Existing 3-layer DFlash2 recipe has different taps and
   is a separate scratch baseline, not a family conversion seed.
5. Use the official MiniCPM DSpark checkpoint directly for DSpark baseline,
   same-strategy continual training and DSpark loss/head ablations. For
   DSpark-to-DFlash/DFlash2, set `training.retrain_anchor_alignment=true` on
   that stage. This copies shared weights, sets the destination method to
   `sample_from_anchor=false`, and retrains on the new label alignment. The
   default rejects this transfer. Copied weights are an initialization;
   zero-step outputs are not equivalent and a short run does not establish
   recovered acceptance. Verify the exact serving source/image supports each
   exported layout.

### One-command MiniCPM5 matrix

After producing the reviewed SFT train/holdout feature directories described
above, prepare one base run. Replace the snapshot path with the locally pinned
official DSpark directory. Both feature paths must have compatible capture
manifests, and each must be separate. The `target prepare` command creates
`runs/minicpm5-base/train.json`:

```sh
draftfit target prepare openbmb/MiniCPM5-2B-SFT \
  --revision 3e0690c57f6ac772180498467c82e56170cc00ac \
  --strategy dspark --teacher-backend transformers \
  --draft-checkpoint /path/to/pinned-official-dspark \
  --hidden-states ./features/minicpm5-sft-train \
  --output-dir ./runs/minicpm5-base \
  --set data.eval_hidden_states_path=./features/minicpm5-sft-holdout \
  --set training.eval_interval=5 --set training.save_interval=5 \
  --set training.max_steps=20
```

Save this manifest as `minicpm5-family.json` in the repository root and
replace every snapshot path with the same pinned directory. `official` trains
20 steps from the published weights; `continual` trains another 20 steps from
that stage's weights without inheriting its optimizer state. For optimizer/RNG
continuity, use `training.resume_from` on an individual run instead.

```json
{
  "base_config": "runs/minicpm5-base/train.json",
  "output_dir": "runs/minicpm5-family",
  "runs": [
    {"name": "official", "strategy": "dspark", "checkpoint": "/path/to/pinned-official-dspark"},
    {"name": "continual", "strategy": "dspark", "from": "official"},
    {"name": "head-only", "strategy": "dspark", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.dspark_head_only=true"]},
    {"name": "eal", "strategy": "dspark", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.dspark_acceptance_loss=eal", "training.dspark_acceptance_loss_alpha=0.1"]},
    {"name": "wtv", "strategy": "dspark", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.dspark_acceptance_loss=wtv", "training.dspark_acceptance_loss_alpha=0.1"]},
    {"name": "dflash-ce", "strategy": "dflash", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.retrain_anchor_alignment=true"]},
    {"name": "dflash-dpace", "strategy": "dflash", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.retrain_anchor_alignment=true", "training.loss_type=dpace", "training.dpace_alpha=0.5"]},
    {"name": "dflash2", "strategy": "dflash2", "checkpoint": "/path/to/pinned-official-dspark", "overrides": ["training.retrain_anchor_alignment=true"]}
  ]
}
```

```sh
draftfit family --manifest minicpm5-family.json --plan
draftfit family --manifest minicpm5-family.json
```

The plan validates target/layer/mask contracts and shared held-out features
without writing output. The run writes per-stage `draft.json`, `train.json`,
`metrics.jsonl`, checkpoints, and aggregate `results.json`. It stops on a
failed stage or missing held-out metrics. Inspect each stage's train/eval loss,
token accuracy, DSpark confidence or DFlash2 selector terms, weight changes,
and exported model separately. Proxy acceptance metrics do not replace actual
server acceptance, token parity, or throughput measurements.

| Requested path | Acceptance check | Current status |
| --- | --- | --- |
| One-command family + ablations | Five-layer config contracts, shared disjoint cache, per-arm checkpoint and held-out metrics | MiniCPM5 GPU: 7/8 original stages passed 5 steps; DFlash2 passed separately with `NO_SHARD`; repaired `FULL_SHARD` passed one GPU step/eval |
| DSpark-family conversion | Explicit anchor retarget, copied shared tensors, destination export reload | Official DSpark CPU load 62/62; DFlash/DFlash2 transfer 58/58; GPU update/eval passed; serving not run |
| Continual training | Weights-only handoff and separate full optimizer/RNG resume with step continuity | MiniCPM5 weights-only continuation passed 5 GPU steps; exact optimizer/RNG resume not run |
| Verification metrics | Finite training loss, held-out loss/accuracy and method terms; live token parity and acceptance | GPU train/eval metrics and four-source holdout below; live token parity/acceptance not run |

### MiniCPM5 GPU gate and mixed-source holdout (2026-10-01)

The one-GPU smoke used pinned `openbmb/MiniCPM5-2B-SFT@3e0690c57f6ac772180498467c82e56170cc00ac` and `openbmb/MiniCPM5-2B-DSpark@114a20fdbf53220712c7fbdd7dccddbf1dedebb4`. The family trained on 24 synthetic arithmetic dialogues and evaluated on eight disjoint synthetic dialogues. Each completed stage wrote a checkpoint, positive gradient norm, and held-out metrics after five optimizer steps. This establishes the training paths, not useful adaptation quality. DFlash2 with `FULL_SHARD` initially failed on mixed Tensor/DTensor inputs in `CandidateSelector.hidden_projection`; the separate `NO_SHARD` run completed five steps. Registering the model's selector call as an FSDP forward method resolved the failure: the original `FULL_SHARD` configuration then completed one GPU step, holdout evaluation, and checkpoint save. Its longer run remains untested.

| Family stage | Synthetic holdout avg accuracy | Synthetic holdout avg loss |
| --- | ---: | ---: |
| Official DSpark | 0.738 | 0.784 |
| Continual from DSpark | 0.726 | 0.785 |
| Head-only | 0.539 | 1.596 |
| EAL | 0.746 | 0.831 |
| WTV | 0.733 | 0.845 |
| DFlash CE retarget | 0.885 | 0.469 |
| DFlash DPACE retarget | 0.756 | 82.164 |
| DFlash2 retarget, `NO_SHARD` | 0.339 | 7.598 |

The data loader produced 160 complete assistant-ended dialogues from four public sources at the requested 25% each: PerfectBlend dedup, OpenCodeInstruct, APIGen-MT, and Nemotron Agentic. Its split contains 152 train and eight held-out dialogues, with two held-out dialogues per source. The eight held-out dialogues were captured on GPU with `max_length=4096`; at `256`, long agent/tool dialogues had no supervised tokens and would have silently biased the validation mix. The training remains the synthetic smoke above. Each evaluation run loaded the trained checkpoint, took one additional synthetic optimizer step, and then evaluated this separate real-data holdout. These numbers are for the post-step checkpoints.

The input was generated with `python scripts/prepare_data.py --dataset mix --sample-size 160 --ratios perfectblend-dedup=25,opencodeinstruct=25,apigen-mt=25,nemotron-agentic=25 --seed 47 --split-eval --output-path OUTPUT_DIR`. The complete-dialogue filter skipped incomplete/tool-ended rows and replenished the per-source counts. The resulting train/test SHA256 hashes are `8561f4689174df1b7a7adae22f8bfe6afd4d63662c7b278ece74e81a1e943148` and `1fa753e4265d53d38580f14d55ff36e073b3fdb052cbaa8cc3e3b78605c6ca70`.

| Trained checkpoint | Four-source holdout avg accuracy | Four-source holdout avg loss |
| --- | ---: | ---: |
| DSpark | 0.572 | 1.499 |
| DFlash CE | 0.133 | 6.096 |
| DFlash2, `NO_SHARD` | 0.029 | 17.935 |

The objectives and candidate selectors differ across families, so their loss values are not directly comparable. Eight examples are too few for a quality claim. CUDA speculative serving is still unverified; the local MLX attempt below failed exact token parity. Local run manifests, logs, results and model-only checkpoints are under `.planning/recent-models/gpu-run/download/`.

### Local MLX speed and exactness check (2026-10-01)

Apple M5 Pro, 64 GB, MLX 0.32.1, `mlx-dspark` `414309238e0f0f2a97315bcf0157e9f20fffebbc`. Target: pinned `openbmb/MiniCPM5-2B-SFT@3e0690c57f6ac772180498467c82e56170cc00ac` BF16. The pinned official DSpark and all eight five-step trained model-only checkpoints were exported to safetensors and loaded by the MLX runtime; the DSpark configs omit `draft_vocab_size` only when it equals the full `vocab_size` (no reduced-vocabulary map exists). Original checkpoints remain unchanged. Drafters were quantized to 4-bit at load, with fixed cap 3 and lookup drafting off. Each of three chat/code/math prompts had one unmeasured warmup and three greedy runs at `max_new_tokens=128`. Figures are median *end-to-end* generated tokens/s across nine runs per arm; the minimum and maximum show variation. Two separate sessions each measured the target-only baseline before and after the draft arms. Raw token IDs, timings, accepted lengths, scripts and local exports are in `.planning/recent-models/gpu-run/download/mlx-benchmark/`.

| Session / arm | Median tok/s (min–max) | Median accepted tokens/round | Exact target-token match |
| --- | ---: | ---: | --- |
| DSpark session, target only before / after | 50.1 (47.3–51.2) / 46.0 (45.7–47.0) | 1.00 | reference |
| Published DSpark | 108.0 (86.8–116.5) | 3.07 | fail |
| Trained DSpark | 97.3 (82.1–112.4) | 3.00 | fail |
| Continual DSpark | 98.0 (83.6–108.8) | 3.00 | fail |
| Head-only DSpark | 107.3 (81.7–112.6) | 3.07 | fail |
| EAL DSpark | 101.5 (81.6–112.8) | 2.93 | fail |
| WTV DSpark | 101.2 (84.0–115.6) | 3.00 | fail |
| DFlash session, target only before / after | 42.6 (41.3–43.7) / 49.4 (46.6–52.9) | 1.00 | reference |
| Converted DFlash CE | 46.2 (42.2–51.6) | 1.31 | fail |
| Converted DFlash DPACE | 52.0 (47.5–53.4) | 1.33 | fail |
| Converted DFlash2 | 36.0 (34.5–40.4) | 1.03 | fail |

Every draft arm first differed from the baseline at generated token 123 (chat), 85 (code), or 88 (math), independent of the draft checkpoint. Published DSpark cap 1/2/3 reproduced the same mismatch positions. At the chat/code boundary the incremental target logits had an exact BF16 tie for the two candidate tokens; math had a 0.125 logit margin. A full-prefix target forward and the MLX hidden-state tap agreed exactly on logits, while the incremental and batched verification paths selected different tokens. This points to numerical/path dependence in this runtime; it does **not** establish exact speculative equivalence. Some speculative runs also emitted 129–130 tokens against a 128-token cap. The observed throughput cannot be promoted as a lossless speedup, and these three public prompts are not a workload-quality benchmark. DFlash-family acceptance after five synthetic steps was low, especially DFlash2. Resolve token parity and the cap behavior, then repeat matched serving A/B on held-out requests before a speed claim.

#### BF16 parity diagnosis and correctness mode (2026-10-01)

The published DSpark `official-normalized/model.safetensors` is a symlink to the pinned official file; its weights were not numerically converted. A two-token replay at the math divergence produced greedy `[282,24548]` and fast speculative `[282,550]`. At the same prefix, one-row BF16 target verification scored tokens 24548/550 at `24.0/23.875`; two-row verification scored them at `24.0/24.0` and reversed the argmax. This isolates the first mismatch to target verification numerics, not the draft checkpoint conversion. Casting the target compute to FP32 made the first 128 tokens match on all three prompts, but that changes the numerical target and is not a BF16 parity fix.

In the local `mlx-dspark` checkout at `4143092` plus an uncommitted correction, `load_target(..., serial_verify=True)` / `generate --serial-verify` runs dense text verification one token at a time, matching the BF16 greedy path. A shared commit-boundary fix also limits speculative output to `max_new_tokens` and keeps the accepted cache prefix in sync. The [saved parity gate](../.planning/recent-models/gpu-run/download/mlx-benchmark/verify_exact.py) compared nine drafts × three prompts at 128 tokens: **27/27 exact token IDs and finish reasons; 27/27 stayed at 128 tokens**. Raw per-arm results are in `strict-parity-results.json` beside the script. One-run medians were about 54 tok/s target-only, 36–50 tok/s DSpark, 19–20 tok/s DFlash and 15 tok/s DFlash2. These are diagnostic timings, not a warmed serving benchmark. The default fast BF16 path still diverges on the two-token replay, so no exact fast-mode speedup is claimed. Serial verification currently supports dense mlx-lm text targets only; hybrid/VLM targets require a separate state-safe solution.

#### TorchSpec zero-step warm-start conversion and local MLX speed (2026-10-02)

At TorchSpec `bd64d93`, `tools/convert_to_torchspec.py` converted the pinned official MiniCPM5 DSpark HF safetensors into DFlash and DFlash2 training-init DCP checkpoints. `tools/convert_to_hf.py` exported both back to serving safetensors. The DFlash export includes an untrained `embed_tokens.weight` that MLX's `load_dflash` rejects and would replace with the target embedding at `bind`; an MLX copy removes only that tensor. Both final MLX artifacts preserve **58/58 shared tensors bitwise** from the official DSpark. The four DSpark Markov/confidence tensors are ignored. DFlash2 additionally has 23 freshly initialized selector/conv tensors; neither conversion performs any optimizer step. TorchSpec presents this as a warm start for continual training, not a quality-preserving direct replacement.

Apple M5 Pro/64 GB, same pinned MiniCPM5-SFT BF16 target and three chat/code/math prompts, 128 output tokens, 4-bit drafters, cap 3, lookup off. Each arm had a 16-token warmup and two measured runs per prompt; baseline ran before and after. The local `mlx-dspark` runtime is `4143092` plus the uncommitted serial verification and output-cap correction. Medians below are across six fast-mode runs per arm; serial-mode medians are across one run per prompt. Raw rows and the runner are in `.planning/recent-models/gpu-run/download/mlx-benchmark/torchspec-conversion/`.

| Arm | Fast tok/s median (range) | Tokens/round median | Exact target IDs | Serial tok/s, exact IDs |
| --- | ---: | ---: | --- | --- |
| Target only, before / after | 53.85 (53.76–54.18) / 53.62 (53.60–53.77) | 1.00 | reference | — |
| Published DSpark | 116.46 (97.21–128.72) | 2.98 | 0/6 | — |
| Platform five-step DFlash CE | 54.00 (50.60–55.43) | 1.31 | 0/6 | — |
| Platform five-step DFlash2 | 41.39 (40.44–43.40) | 1.03 | 0/6 | — |
| TorchSpec zero-step DFlash | 52.23 (49.09–54.37) | 1.25 | 0/6 | 18.78, 3/3 |
| TorchSpec zero-step DFlash2 | 51.08 (46.54–51.60) | 1.29 | 0/6 | 18.94, 3/3 |

All fast draft arms first differ at positions 123/85/88 on the same three prompts, consistent with the target verifier's existing BF16 batch-width discrepancy rather than changed copied weights. Their speed cannot be reported as lossless acceleration. TorchSpec's zero-step DFlash2 happened to outperform the platform's five-step DFlash2 in this small local run, but its new selector/conv weights are untrained; three prompts and one initialization do not establish a reliable conversion benefit. Neither converted arm matches the official DSpark's draft acceptance or speed. Training on a held-out-representative workload and a numerically exact fast verifier remain separate gates.

### One-GPU run order

| Gate | Work | Evidence / decision |
| --- | --- | --- |
| 0: rental smoke | Check CUDA/BF16, full checkpoint load, tokenizer/mask, one real feature batch, one optimizer step and peak VRAM. | Stop and fix environment if any step fails. |
| 1: drift baseline | On the same 60 held-out prompt IDs, serve target-only and official DSpark with final target, then with SFT target; greedy, max 128 tokens, same server settings. | Exact target-only/speculative token IDs, actual accepted tokens per verification, position-wise acceptance and tokens/s. Call it drift only if the paired acceptance drop is observed. |
| 2: one-command family | From the same official DSpark checkpoint, run standard, head-only, EAL and WTV DSpark arms and anchor-retarget DFlash CE/DPACE and DFlash2 arms. Use equal budgets and save/evaluate every five steps. Continue a DSpark arm from its earlier trained checkpoint. | One manifest/result tree; finite loss/gradient, changed trainable weights, frozen backbone for head-only, held-out loss/token accuracy/confidence/selector terms, checkpoint and export reload. `simulated_acc_len` remains a proxy. |
| 3: conversion | Check official DSpark-to-DFlash/DFlash2 shared tensors bitwise, new selector initialization and complete reload; check DSpark checkpoint handoff and DFlash-to-DSpark reverse transfer separately. | Architecture, block, taps, vocab, mask and anchor metadata match. Omitted retarget option is rejected; copied backbone is not zero-step quality parity. |
| 4: continual/resume | Warm start the official DSpark on SFT features for 10 steps, resume the *same run* to step 20, and compare with an uninterrupted 20-step control from the same checkpoint/seed. If held-out acceptance actually improves, extend only the promising arm to 50 or 100 steps. | Distinguish weights-only warm start from optimizer/RNG resume; verify step continuity, finite loss and held-out acceptance. Short runs establish wiring, not recovery by themselves. |
| 5: real serving | Reload the exact exports. Compare target-only, official, continually adapted, converted DSpark, DFlash and DFlash2 on held-out requests, where the pinned server supports each layout. | Exact greedy token IDs/EOS, accepted length, position-wise acceptance, three warmed throughput repetitions, latency and peak VRAM. Mark unsupported or failing layouts `not-run`/`fail`; no speed claim from training metrics. |

If final-to-SFT shows no paired acceptance drop, use the run only as a
functional transfer test. A Midtrain target is a separate, optional drift
probe with its own pinned revision and baseline; do not relabel SFT as drift.
The family jobs all use SFT features; the final target appears only in the
drift baseline. Use distinct output directories for each run and retain
revision/config/data digests, raw metrics, export hashes and server version.

For rental planning, the 2026-09-30 read-only Vast on-demand search found a
single RTX A6000 48 GiB offer at about **$0.433/hour including 120 GiB disk**
(offer 21004754, availability and price can change). A six-hour ceiling at
that rate is about $2.60 before transfer fees; allow setup, benchmark, backup
and destruction time inside the ceiling. Requery offers and account balance
before creation. Storage keeps billing while an instance exists, including
when stopped. Rent only after the integrated CPU and renderer gates pass; use
the returned instance ID, not this offer ID, for cleanup.
