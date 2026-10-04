# Next DSpark targets: Taichu and LFM2.5

Status on 2026-10-02: Taichu text DFlash2 conversion and tiny CPU forward/export checks pass; tiny native LFM conv/MoE capture checks pass. Both pinned LFM DSpark drafts loaded 62/62 real tensors on CPU, and LFM2.6B→DFlash2 copied 58 shared tensors. The two LFM JSON files in `configs/` are metadata candidates, and the Taichu DSpark JSON mirrors the official geometry; use the pinned official checkpoint directory as the config and weight source for a real warm start. No original-target feature parity, Taichu full official-weight load, GPU update, exact serving parity, or throughput pass is recorded here.

## Pinned sources and compatibility

| Target → official draft | Pinned target / draft commits | Observed draft contract | Current platform boundary |
| --- | --- | --- | --- |
| [ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B) → [ZDTaichu5.0-9B-DSpark](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B-DSpark) | `bc125a9819a421755f4dcebe498eb81c139b65a3` / `af3e79072af64c5ae2e421783c7bcdb40073f597` | Official [config](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B-DSpark/blob/af3e79072af64c5ae2e421783c7bcdb40073f597/config.json): `speculators` layout, 5 layers, block 8, taps `[1,8,15,22,29]`, mask 248077, four sliding plus one full draft attention layer, `sample_from_anchor=true`. HTTP-range read of `model.safetensors` header: 62 tensors, including copied target embedding and LM head. | The loader unfolds the official config, rejects incompatible geometry, and checks both copied tensors against the loaded target before training. `configs/zdtaichu5-9b-dspark.json` now mirrors official geometry, but full-weight load and text-only numerical parity remain unverified. |
| [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) → [LFM2.5-8B-A1B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark) | `5dd22602c2e9f6a097b1de4c4efe0658b605015c` / `5b285c827912834665b1915f171897e49ff0f388` | Official [config](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark/blob/main/config.json): `Lfm2DSparkDraftModel`, 5 layers, block 9, taps `[2,6,10,14,18]`, mask 125017. HTTP-range header: 62 tensors, 327,707,521 BF16 parameters, no embedding/head copy. Target is 24-layer conv/full-attention MoE. | The alias maps to the registered DSpark backbone with interleaved RoPE. All 62 local tensor names/shapes match the official header; a tiny saved LFM-format checkpoint loads without reinitialization. Tiny native MoE capture passes; original-target capture/serving proof is outstanding. |
| [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) → [LFM2.5-2.6B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) | `654f9463ce32b05d0429d76fe1f580b27d4c1ac0` / `458cedab07d0f7b2b05700c77e1aa463d43d6f04` | Official [config](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark/blob/main/config.json): `Lfm2DSparkDraftModel`, 5 layers, block 9, taps `[2,9,17,21,27]`, mask 125017. HTTP-range header: 62 tensors, 327,707,521 BF16 parameters, no embedding/head copy. Target is 30-layer conv/full-attention dense LFM2. | Same adapter and 62/62 header shape match. Tiny native dense capture passes; original-target capture/serving proof is outstanding. |

The header counts above exclude safetensors `__metadata__`; Taichu's header verifies key names and shapes, not weight values or model behavior. Both pinned LFM draft weights loaded into the local CPU model without missing, unexpected or mismatched tensors; source-runtime numerical parity remains unverified. Target parameter counts from the Hugging Face model API are about 9.79B, 8.47B total/1B active, and 2.70B respectively. The LFM candidate JSONs also declare interleaved RoPE, but the official checkpoint directory remains the preferred source of both config and weights.

`draftfit target inspect` can read the pinned targets without downloading weights:

```sh
draftfit target inspect LiquidAI/LFM2.5-8B-A1B --revision 5dd22602c2e9f6a097b1de4c4efe0658b605015c
draftfit target inspect LiquidAI/LFM2.5-2.6B --revision 654f9463ce32b05d0429d76fe1f580b27d4c1ac0
draftfit target inspect TaichuAI/ZDTaichu5.0-9B --revision bc125a9819a421755f4dcebe498eb81c139b65a3
```

## Official warm start contract

Download each *pinned* official draft snapshot to a local directory with its original `config.json` and `model.safetensors`. Use the corresponding pinned target revision in the table. Supply that directory as `--draft-checkpoint` and **omit `--draft-config`**; the loader reads both official files directly. The candidate JSON files are for metadata preparation, not substitutes for pinned official config and weights. The loader rejects a mismatched explicit config before loading weights. The Taichu training build also compares the two copied matrices with the target embedding/head row by row and aborts on a mismatch. LFM drafts do not contain these matrices and use target weights normally.

For an already validated train/holdout feature pair, the preparation command is:

```sh
draftfit target prepare "$TARGET" --revision "$TARGET_SHA" --strategy dspark \
  --teacher-backend transformers --draft-checkpoint "$PINNED_DRAFT_DIR" \
  --hidden-states "features/$NAME-train" --output-dir "runs/$NAME-official" \
  --set "data.eval_hidden_states_path=features/$NAME-holdout" \
  --set training.max_steps=1 --set training.save_interval=1 \
  --set training.eval_interval=1
draftfit train -c "runs/$NAME-official/train.json" --plan
```

This is a prepared command, not a verified full-weight run. Tiny LFM conv/MoE capture passes, but original-target capture and numerical parity still need a real model run. Taichu text capture and numerical parity also remain unverified on the original target. A local snapshot path keeps the draft revision fixed; a mutable repo ID does not.

### Numerical acceptance gate before the first update

Use one fixed text token sequence and the same BF16 target/draft revisions in official SGLang or vLLM and in Draftfit. Record the target's five tapped hidden tensors, the draft's projector output and each decoder-layer output, its final hidden states, target-head base logits, and Markov-corrected logits at the same anchor and candidate positions. Compare shapes, maximum and mean absolute error, relative error away from zero, top-1 token agreement, and finite values. Check the Taichu copied embedding/head equality gate first. For LFM, confirm adjacent-pair RoPE against its official `rope_is_neox_style=false` runtime. Record reference engine commit, attention backend, input IDs, position IDs, mask, dtype, and tolerance with the result. Any shape, position, or anchor mismatch blocks training; establish numeric tolerance from the actual BF16 reference run, rather than assuming an arbitrary threshold. No such parity result exists yet.

## Short-run template after the capture gate

For each target, first implement/validate its text-only feature boundary. LFM's conv/MoE layer outputs and rollback/replay need a dedicated teacher and serving check. For Taichu, inspect its remote modeling code before enabling `--trust-remote-code` and verify Qwen3.5 4D position handling, template mask, 5 tapped tensors and final norm. Its native template has no `{% generation %}` block; compare the `qwen3.5` renderer's token IDs and assistant mask with native output. LFM's official template does include generation blocks; use `hf` as the initial renderer candidate and verify its assistant mask. Use separate train and holdout conversations from the validated data loader; do not let one conversation cross the split. A capture manifest must pin the same target commit, tokenizer and taps in both feature directories.

The following official-draft capture example is a **template, not a command claimed to work today**. Set `TARGET`, `REVISION`, `DRAFT_DIR`, `TRAIN_JSONL`, `HOLDOUT_JSONL`, `NAME`, and `CHAT_TEMPLATE` separately for one target, using the table's SHA and pinned official draft directory. Set `CHAT_TEMPLATE=qwen3.5` for Taichu or `CHAT_TEMPLATE=hf` for LFM only after validating actual token IDs and supervised masks. The 1-step limit tests wiring, not draft quality:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --target-backend transformers --target-model-path "$TARGET" --target-revision "$REVISION" \
  --strategy dspark --draft-model-config "$DRAFT_DIR/config.json" --data-path "$TRAIN_JSONL" \
  --chat-template "$CHAT_TEMPLATE" --output-path "features/$NAME-train" \
  --max-length 128 --num-samples 2 --batch-size 1 --torch-dtype bfloat16
python -m torch.distributed.run --standalone --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --target-backend transformers --target-model-path "$TARGET" --target-revision "$REVISION" \
  --strategy dspark --draft-model-config "$DRAFT_DIR/config.json" --data-path "$HOLDOUT_JSONL" \
  --chat-template "$CHAT_TEMPLATE" --output-path "features/$NAME-holdout" \
  --max-length 128 --num-samples 2 --batch-size 1 --torch-dtype bfloat16
draftfit target prepare "$TARGET" --revision "$REVISION" --strategy dspark \
  --teacher-backend transformers --draft-checkpoint "$DRAFT_DIR" \
  --hidden-states "features/$NAME-train" --output-dir "runs/$NAME-base" \
  --set "data.eval_hidden_states_path=features/$NAME-holdout" \
  --set training.max_steps=1 --set training.save_interval=1 --set training.eval_interval=1
draftfit train -c "runs/$NAME-base/train.json" --plan
```

`target prepare` is metadata-only and rejects `model.trust_remote_code=true`. Taichu may require an explicitly reviewed `--trust-remote-code` during capture and a corresponding explicit opt-in in the generated `train.json`. LFM now passes the native tiny decoder capture check; validate its original target and tokenizer before training.

After real train/holdout features pass, a small **sequential** family manifest can be saved alongside the run directory (the paths below are relative to the manifest):

```json
{
  "base_config": "runs/MODEL-base/train.json",
  "output_dir": "runs/MODEL-family",
  "runs": [
    {"name": "official-dspark", "strategy": "dspark"},
    {"name": "continual", "strategy": "dspark", "from": "official-dspark"},
    {"name": "head-only", "strategy": "dspark", "from": "official-dspark", "overrides": ["training.dspark_head_only=true"]},
    {"name": "dflash-retarget", "strategy": "dflash", "from": "official-dspark", "overrides": ["training.retrain_anchor_alignment=true"]}
  ]
}
```

`draftfit family --manifest PATH --plan` validates the feature contract first; `draftfit family --manifest PATH` then launches one stage at a time and requires held-out loss/accuracy in `results.json`. Here `continual` is a weights-only handoff. Resume of optimizer/RNG state uses the individual training `resume_from` path instead. The first stage takes the official checkpoint from the base config; the remaining stages use its trained checkpoint. Full-weight load and DFlash-family save/reload still need a real run. Converted weights require anchor-alignment retraining before quality is assessed.

For either LFM model or Taichu text, a `dflash2-retarget` row with `"strategy":"dflash2"`, `"from":"official-dspark"`, and `"overrides":["training.retrain_anchor_alignment=true"]` passes metadata conversion. Taichu selects `TaichuDFlash2DraftModel` for its restricted text MRoPE and mixed sliding/full layout. Its tiny CPU RoPE, masks, backward, reload, and export schema pass; all three targets still need original-weight feature, training, and serving checks. The existing `configs/zdtaichu5-9b-dflash2.json` is a separate three-layer scratch recipe, not the five-layer official retarget config.

## One H200 / 140 GB schedule

Approximate BF16 *weights only* from official API parameters: Taichu target 19.59 GB + official draft 6.42 GB; LFM 8B target 16.94 GB + draft 0.66 GB; LFM 2.6B target 5.39 GB + draft 0.66 GB. Full AdamW training also needs draft gradients (~2 bytes/parameter) and two FP32 optimizer moments (~8 bytes/parameter): a Taichu draft adds ~32.1 GB beyond its weights, each LFM draft adds ~3.28 GB beyond its weights. Activations, teacher feature capture, CUDA allocator, checkpoint saves, and non-FP32 optimizer details are extra and unmeasured. These are capacity bounds, **not** measured peak VRAM.

`family` is sequential by implementation. Start with the already prepared MiniCPM5 smoke; inspect actual `torch.cuda.max_memory_allocated`, finite loss, gradient and saved weight change. Then run one new target's capture, release its teacher, and train its family stages. Keep all jobs serial until individual measured peaks leave a clear margin; multiple concurrent processes on one H200 can reduce throughput or OOM despite the weight estimate. CPU metadata/dataset work can run concurrently with the GPU job. Do not run training commands from synthetic capture evidence; validate original target features and masks first. Only measured peak memory and time can justify concurrent GPU training.

## Verification gates

| Gate | Status here | Required evidence |
| --- | --- | --- |
| Official repo ID, pinned config, key header | pass | HF API SHA/config and HTTP 206 range header; no full weight download |
| Candidate Taichu/LFM JSON contract | metadata pass | Exact dims/taps/mask and `target prepare` pass for Taichu DSpark/DFlash2 and both LFM DSpark candidates; `train --plan` still needs real feature manifests |
| Target/tokenizer mask and offline capture | tiny LFM conv/MoE capture pass; original targets not-run | Real model forward, assistant-only nonempty mask, tapped shapes/finite values, matching train/holdout capture manifests |
| Official checkpoint conversion | LFM drafts load 62/62 actual tensors; LFM2.6B→DFlash2 copies 58 shared tensors; all three DSpark→DFlash/DFlash2 configs pass; Taichu tiny DFlash2 CPU/export passes | Taichu full-weight load, original-target feature parity, correct anchor/attention semantics, trained-artifact save/reload |
| One-step train, continual, family metrics | not-run | Finite loss/grad, changed weights, held-out `eval/avg_loss` and `eval/avg_acc`, checkpoint reload and step continuity |
| Exact serving and throughput | not-run | Exact target-only greedy tokens, accepted lengths, repeated same-prompt throughput and state rollback/replay |

Before renting for these targets, run a real-weight text forward and assistant-mask check for Taichu and LFM. Tiny native LFM dense/MoE conv-block capture passes, but a metadata or tiny-model match does not clear the original-target parity gate. The LFM taps include conv blocks; replacing them with attention-only taps would change the official draft contract. Taichu DSpark→DFlash2 now passes text-only CPU config and mask checks. Keep the first GPU run serial and measure peak VRAM before considering concurrent training.

The existing [MiniCPM5 experiment](RECENT_MODEL_EXPERIMENTS.md) remains the nearest real-GPU reference; these three models require their own captured features and proof.
