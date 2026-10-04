# Recent target training branch

This branch adds experiments for adapting an existing DSpark draft to a new
target and training independent DSpark, DFlash and DFlash2 drafts from shared
features. It does **not** produce one weight file that serves as every method.
The [family workflow](DFLASH_FAMILY.md) explains new-draft, conversion,
continual and resume paths and shows one-GPU and multiple-GPU layouts.

## Delivered paths

| Path | Entry point | Boundary |
| --- | --- | --- |
| Family and ablation matrix | `draftfit family --manifest family.json --plan`, then `draftfit family --manifest family.json` | Compatible offline train/held-out caches; independent optimizer and checkpoint per arm. |
| Existing DSpark to DFlash or DFlash2 | `model.draft_checkpoint_path` with `training.retrain_anchor_alignment=true` when the source samples from anchor | Compatible weights transfer; method-specific heads initialize separately and training is required. |
| Continual from existing weights | `model.draft_checkpoint_path` | New optimizer and step schedule; `training.resume_from` instead resumes the same interrupted run with optimizer state. |
| Raw-text initialization | `draftfit pretrain`, then `draftfit expand-aux-layers` and verifier-feature distillation | Single-device Qwen3-shaped methods; hard-token metrics are not verifier acceptance. |
| Domain validation | `scripts/prepare_data.py` and `draftfit benchmark` | Filters empty/tool-final conversations, fills requested quotas from valid rows, and reports prompt-domain timing; exact serving checks remain separate. |

See [the MiniCPM5 experiment record](RECENT_MODEL_EXPERIMENTS.md) for pinned
targets, manifests and training gates, [the consolidated report](reports/minicpm5-family-2026-10-03.html)
for measured speed and parity limits, and [the Hugging Face release record](specs/minicpm5_hf_release.md)
for four public model repositories and nine revision hashes. The model cards
start with Draftfit's purpose, show the common v4 speed/domain table, and explain
when to investigate DSpark, DFlash CE, DFlash DPACE or DFlash2.

## Evidence and remaining gates

The MiniCPM5 v4 family was trained on separate 1,000-row stages and tested on a
53-conversation four-source holdout. On the measured A100 v4 setup, the official
DSpark reached 354.69 tok/s, trained DSpark 347.94, DFlash CE 294.77 and DPACE
304.85; DFlash2 had no compatible A100 serving path. MLX measured all four
draft methods on the same holdout with a different throughput definition. The
53-prompt fast BF16 parity and 12-prompt A100 checks expose correctness gaps;
these exports are research artifacts rather than deployment-certified drafts.

Additional Taichu/LFM/Xing/Agnes/Hemmingway recipes are **candidate configs**.
Their real-weight capture, full training and exact serving have not all run.
DFlow, Carryover, DScale and DPara are not registered training/serving features
in this branch; a separate documentation/research worktree discusses them.

The public weights inherit upstream attribution and include APIGen-MT training
samples. Its noncommercial and generator-provenance limits are documented in
each Hugging Face model card; downstream rights and commercial use are not
cleared. Local `.planning/` experiment data, private prompts, checkpoints,
credentials and feature caches are excluded from Git.

## Integration note

The branch was developed on an older, independent Git history. The current
`main` contains later package and documentation cleanup. Integration into a
`dev` branch based on current `main` must port the feature diff while preserving
those cleanup changes, then rerun CPU and package checks. Keep `main` unchanged
until the user requests its later batch merge.
