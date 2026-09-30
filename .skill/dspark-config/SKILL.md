---
name: dspark-config
description: Select target metadata, draft settings and a compatible runtime for DSpark, DFlash, DFlash2 and other Draftfit methods.
---

# Prepare targets and configuration

Read [environments](../../docs/MODEL_ENVIRONMENTS.md),
[target extension](../../docs/TARGET_EXTENSION.md),
[support](../../docs/PUBLIC_SUPPORT.md) and [uv setup](../../docs/ENVIRONMENT.md).
The maintained SpecForge-derived engine is included; do not install upstream separately.

Use the full procedure for initial setup or target/backend changes. For
learning-rate, step or path edits, validate affected fields without unnecessary
reinspection, new projects or runtime rebuilds.

1. Establish target path or HF ID/exact revision, draft method, data/features
   and GPU requirements. Preserve the user's chosen model.
2. Use `draftfit target inspect` for config/tokenizer metadata and weight keys.
   Distinguish model-card claims from JSON; review remote code before execution.
3. Identify depth, hidden width, vocab, embeddings/head, attention, MoE,
   linear/conv/recurrent state, template and assistant masks. Depth/tap
   recommendations are hypotheses, not proven optima.
4. Reuse environments if the backend supports execution/capture/serving/state.
   Otherwise identify missing backend work. Config does not implement Python
   or kernels. Distinguish stock SGLang from capture builds.
5. Check `draftfit algorithms` and `draftfit target prepare --help`; prepare a
   new output directory with individual `--set` fields and dedicated data/draft
   arguments. Preserve files and preparation's remote-code safeguards.
6. Check paths, layers, runtime and GPU placement with
   `draftfit train -c PATH --plan`. Online needs Mooncake/capture servers;
   offline requires matching target/tokenizer/revision/taps. Preparation does
   not validate actual features.

Report environment, config paths, changed fields/reasons and remaining checks.
Keep uncertain template/feature/state gates pending. Model names alone do not
prove dtype/kernel support. See the [training skill](../dspark-train-validate/SKILL.md).
