# Joint family training with a resident target

`draftfit family --joint` trains compatible DSpark, DFlash and DFlash2 arms
with one shared backbone, separate method-specific parameters and one optimizer.
Each input batch is read once. With online capture, the frozen Transformers
target runs once per microbatch and all arms consume the resulting features.
An exported arm is a separate serving model; sharing training parameters does
not create one universal serving checkpoint.

![Joint training shares target features and a backbone while preserving separate family losses and exports.](_static/diagrams/joint-lifecycle.svg)

## Choose the execution mode

| Mode | Command | Parameters and input |
| --- | --- | --- |
| Independent arms | `draftfit family --manifest family.json` | Separate models and optimizers; compatible offline features |
| Concurrent independent arms | Add `--jobs N` | Separate workers; concurrency is not shared-backbone learning |
| Joint offline | Add `--joint` | Shared backbone and batch reader; compatible offline features |
| Joint online | `--joint` plus `training.joint_online_capture=true` | Resident frozen target; tokenized train and holdout rows |
| Unattended online | `draftfit unattended --config controller.json` | Joint online plus bounded policy, verification, recovery and export |

See [the family guide](DFLASH_FAMILY.md) for weight conversion and independent
runs, or [the unattended guide](UNATTENDED_NATIVE.md) for policy-driven runs.
The joint runner is a single-device path; it is not the multi-GPU disaggregated
producer/consumer service.

## Manifest and data

Create `family.json` beside your typed `base.json`:

```json
{
  "base_config": "base.json",
  "output_dir": "joint-output",
  "runs": [
    {"name": "dspark", "strategy": "dspark"},
    {"name": "ce", "strategy": "dflash"},
    {"name": "selector", "strategy": "dflash2"}
  ]
}
```

Start with a compatible target/draft configuration from the
[workflow](PUBLIC_WORKFLOW.md) and [target recipes](RECENT_MODEL_EXPERIMENTS.md).
For online joint training, replace offline feature paths with
`data.prompts_path` and `data.eval_data_path`, and set
`model.target_backend="transformers"`. These files contain **tokenized complete
sequences**, not raw conversations or unanswered prompts. Prepare responses and
tokenize them with the pinned target tokenizer before starting this runner.

Each JSONL row needs equal-length `input_ids` and binary `loss_mask` arrays.
A supported contiguous `attention_mask` is optional; a `domain` label is used
by verification and policy. Train and holdout sequences must be disjoint and
fit `data.max_length`. The first `joint_verification.max_prompts` holdout rows
form the fixed decoding workload, so include every required domain there.

The following is a **training-section fragment** to merge into `base.json`;
it does not supply the target, draft geometry, mask token or dataset paths:

```json
{
  "training": {
    "strategy": "dspark",
    "joint_online_capture": true,
    "joint_feature_queue_bytes": 268435456,
    "joint_feature_queue_batches": 4,
    "batch_size": 1,
    "accumulation_steps": 4,
    "num_epochs": 10,
    "max_steps": 1000,
    "total_steps": 1000,
    "eval_interval": 100,
    "save_interval": 100,
    "log_interval": 10,
    "fsdp_sharding": "NO_SHARD",
    "attention_backend": "sdpa",
    "joint_verification": {
      "interval": 100,
      "max_prompts": 4,
      "max_new_tokens": 16,
      "draft_tokens": 4,
      "trials": 1
    }
  }
}
```

The queue must accommodate the accumulation window and feature bytes. The data
and epoch count must provide the requested optimizer updates:
`floor(floor(rows / batch_size) * num_epochs / accumulation_steps)` is the
capacity check, not `rows == steps`. Preflight also validates accumulation
boundaries. Increasing a step limit does not generate more data.

## Scratch, continual and resume

- **Scratch:** omit `model.draft_checkpoint_path`.
- **Continual initialization:** set that field to a compatible existing local
  draft. This is a weights-only start with a new optimizer and schedule.
  Anchor-aligned DSpark conversion requires
  `training.retrain_anchor_alignment=true` on DFlash/DFlash2 arms.
- **Per-arm seeds:** allowed only when shared backbone tensors and required
  layouts agree. Divergent backbones or mixed scratch/seed initialization cannot
  silently be merged into one backbone.
- **Exact continuation:** use the joint checkpoint with `--resume-from` and a
  new output directory. Preserve the data, architecture and scheduler contract.

```sh
draftfit family --manifest family.json --joint --plan
draftfit family --manifest family.json --joint --stop-at-step 100
# family-resume.json uses a new output_dir and the same training contract.
draftfit family --manifest family-resume.json --joint \
  --resume-from joint-output/joint-step100.pt
```

`--stop-at-step` is an operational boundary; it preserves the configured full
scheduler horizon. The checkpoint contains shared and private weights,
optimizer state, RNG, counters and the committed input cursor. Per-arm exports
are useful for inference but do not replace that exact-resume checkpoint.

## Compatibility and verification

All arms must agree on the target revision, tokenization, feature taps, hidden
width, backbone geometry and relevant attention/alignment settings. Current
joint training requires the same tap layout; automatic union-of-layers capture
is not implemented. DSpark also needs final normalized target hidden states.

Native verification supports the implemented Llama/Qwen2/Qwen3 and
Gemma4Unified text adapters; capture support alone does not imply verification
support for a hybrid architecture. Draft attention must use a supported eager
or SDPA path. See [queue and verification behavior](SHARED_TARGET_FEATURE_QUEUE.md)
and [support boundaries](PUBLIC_SUPPORT.md).
