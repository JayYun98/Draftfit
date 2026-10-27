# DSpark / DFlash family: starting and continuing a draft

Draftfit supports **DSpark, DFlash and DFlash2** with three distinct execution
modes. Feature reuse, shared-backbone learning and checkpoint conversion have
different contracts:

| Mode | Command | Shared state |
| --- | --- | --- |
| Independent matrix | `draftfit family --manifest family.json` | Compatible offline inputs; separate optimizers and drafts |
| Concurrent independent matrix | Add `--jobs N` | Separate workers; no shared optimizer/backbone |
| Joint family | Add `--joint` | One batch reader, shared backbone, private method parameters and one optimizer |
| Unattended joint online | `draftfit unattended --config controller.json` | Resident target, bounded feature queue, verification, policy and durable exports |

Use [joint online setup](JOINT_ONLINE.md) for scratch or existing-draft joint
initialization, [queue and verification](SHARED_TARGET_FEATURE_QUEUE.md) for
memory/recovery behavior, and [unattended execution](UNATTENDED_NATIVE.md) for
automatic bounded phases. Each family is exported separately; shared training
does not make one interchangeable serving checkpoint.

The offline diagrams below describe independent runs. Their scheduling and
validation limitations do not describe the colocated joint runner.

## No existing draft

![Independent family training from frozen-target features through export and serving checks.](_static/diagrams/family-scratch.svg)

Use `draftfit target prepare` with `--strategy dspark`, `--strategy dflash` or
`--strategy dflash2` and a suitable draft config, then `draftfit train -c ...`.
For a matrix, `draftfit family
--manifest family.json --plan` validates the shared offline sources before
`draftfit family --manifest family.json` runs stages in order. A DSpark superset
cache with `target_last_hidden_states` can also supply DFlash/DFlash2 when the
target, layers, tokenizer and mask contracts match. A different tap layout needs
separate capture. The family runner requires disjoint train/held-out sources and
`training.eval_interval`; it stops on a failed stage or missing checkpoint or
held-out loss/accuracy. It does not perform export or serving validation.

## Raw-text pretraining

[Speculators RFC #1127](https://github.com/vllm-project/speculators/issues/1127)
proposes an inexpensive first stage before verifier distillation. Draftfit now
offers a **single-device** version for Qwen3-shaped DSpark, DFlash and DFlash2:
it loads only the frozen target embedding and LM head, packs local JSONL `text`
rows with EOS separators, and trains from hard next-token labels. It does not
run the target decoder or create verifier hidden-state features. Use a separate
held-out file; up to 32 full sequences are checked by default.

```sh
draftfit pretrain --target /path/to/pinned-target \
  --draft-config configs/minicpm5-2b-dflash2.json --strategy dflash2 \
  --corpus /path/to/train-text.jsonl --validation /path/to/heldout-text.jsonl \
  --output ./pretrained-one-tap --steps 100 --sequence-length 256
draftfit expand-aux-layers --checkpoint ./pretrained-one-tap \
  --target-layer-ids 0 10 21 31 41 --output ./pretrained-five-taps
draftfit target prepare /path/to/pinned-target --local-only --strategy dflash2 \
  --hidden-states /path/to/matching-on-policy-features \
  --draft-checkpoint ./pretrained-five-taps --output-dir ./distill-run
draftfit train -c ./distill-run/train.json
```

The pretrain output is an ordinary draft checkpoint with one auxiliary tap,
layer 0. Expansion copies that projection into the layer-0 slot and zeroes
the new columns, preserving the embedding-only function mathematically. A wider
matrix multiplication can still differ by floating-point rounding; bitwise
output parity is not promised. The
subsequent `draftfit train` run loads weights only; it starts a new optimizer.
Use the same target weights/tokenizer, block size, anchor alignment, attention
layout and expanded tap list as the feature capture. The expanded list must
include tap 0; recapture features if an existing cache omits it. `summary.json` and
`metrics.jsonl` record loss, accuracy, gradient norm and position-wise hard-token
diagnostics. DSpark's hard-label pretraining uses point-mass TV and confidence
targets; these are corpus proxies, not verifier acceptance measurements.

This path has no multi-GPU sharding, optimizer resume or target embedding
scaling for Gemma/Granite. A held-out distillation A/B and exact serving check
are still needed to establish quality or speed. TorchSpec's
[conversion tool](https://github.com/lightseekorg/TorchSpec/blob/main/tools/convert_to_torchspec.py)
also treats shared tensor transfer as a warm start. Its token/position-aware
training metrics informed our DFlash-family position-wise CE and accuracy
telemetry; neither converter promises zero-step quality parity.

## Existing draft: identify the source first

![Existing-draft initialization, checked family conversion and exact resume routes.](_static/diagrams/family-continual.svg)

| Source | Implemented route | Boundary |
| --- | --- | --- |
| Same-method DSpark, DFlash or DFlash2 | `model.draft_checkpoint_path` / `--draft-checkpoint`; weights-only warm start | Source draft config and required tensor shapes must match. New optimizer, schedule and step count. |
| DSpark with `sample_from_anchor=true`, including the official MiniCPM5 draft | Same-method DSpark warm start; or explicit DSpark → DFlash/DFlash2 initialization | Cross-method path requires `training.retrain_anchor_alignment=true`, sets destination `sample_from_anchor=false`, and needs new training. Copied weights do not imply zero-step prediction or acceptance parity. |
| Non-anchor DSpark, DFlash or DFlash2 | Checked conversion among these three methods | Full compatible shared layout, vocabulary, taps, mask and feature contract are required. DSpark heads or DFlash2 convolution/selector tensors are newly initialized when absent. No generic checkpoint conversion guarantee. |
| Same interrupted training run | `training.resume_from` | Restores training state; cannot be combined with `model.draft_checkpoint_path`. Independent stage handoffs use weights only; joint exact resume uses `--joint --resume-from`. |
| EAGLE3, PEagle, Domino, target MTP head or unrelated checkpoint | No DSpark/DFlash family handoff | A matching method-specific workflow or new draft is needed. Taichu's published `speculators` checkpoint is not supported by the new-draft recipes. |

The family manifest accepts an external `checkpoint` or `from` an earlier arm,
never both in one arm. Cross-method handoff requires compatible architecture,
block size, hidden width, layer count, full vocabulary where DFlash2 needs it,
attention layout, target taps, mask ID and relevant alignment settings. Invalid
combinations fail during planning or loading. Explicit draft configs need to
match the chosen method; changing `training.strategy` alone does not convert a
draft. Refer to [the workflow guide](PUBLIC_WORKFLOW.md#fine-tune-weights-versus-resume-a-run)
for the individual-run commands.

### Published `speculators` checkpoints

Recognized nested Qwen3-compatible DSpark and DFlash configurations can be used
as warm starts. Generic nested sources must declare
`model.draft_capture_layer_indexing`, or provide equivalent explicit source
metadata. `block_output` means zero-based decoder block outputs;
`embedding_inclusive` means source ID `i > 0` maps to decoder block `i - 1`.
Embedding slot zero is unsupported by this capture path. Missing or conflicting
conventions fail before loading weights; repository names and method names do
not identify the producer's capture boundary. Saved native configurations retain
original IDs, the declaration and resolved physical block IDs so reload does not
apply the conversion twice. A declaration enables a controlled experiment; it
does not prove that an undocumented publisher used that convention.

For example, the verified wkk Qwen3-0.6B DSpark producer uses `block_output`.
A different publisher must be checked separately. Keep its resolved producer
configuration or feature metadata with the checkpoint. The stored format alone
cannot establish capture or zero-step quality parity.

The loader resolves the target depth from the actual target
configuration, preserves the source taps, label alignment and attention layout,
and requires compatible tensor shapes. If the published artifact also contains
frozen target embedding or LM-head copies, every present copy must match the
selected target exactly after dtype promotion. Reduced-vocabulary or unknown
layouts and unexplained checkpoint keys remain errors.

The recognized anchor-relative sliding-window format uses the published context
rule, rather than silently substituting query-relative local attention. Compatible DFlash2 conversions retain this mask contract in training and native
verification; their convolution and selector parameters are newly initialized
when absent from the seed. Its
semantics come from the pinned [Speculators attention implementation](https://github.com/vllm-project/speculators/blob/47eee66a19cb5f967b215575d866934250522ad1/src/speculators/models/dflash/attention.py).
`pure_draft_prefix_len` affects the upstream Domino branch; it does not shift
plain DFlash labels. Support for this format does not imply support for every
Speculators model or its serving runtime.

DSpark generation follows its stored `sample_from_anchor` alignment. When true,
hidden slot zero predicts the first token after the committed anchor. When
false, slot zero predicts the anchor and generation starts at slot one. Both
paths emit the same number of speculative proposals and retain the Markov head.
Verify the exact exported weights with the intended serving implementation;
correct training loss alone cannot establish proposal alignment or speed.

## Training and validation by GPU count

### One GPU: independent offline capture, train, then serve

![Single-GPU offline capture, independent draft training and separate serving evaluation.](_static/diagrams/family-offline.svg)

The offline path reuses one GPU by capturing features before training. Train
and held-out files must be disjoint and use the same target revision, tokenizer,
tap layers and mask contract. The `family` runner executes its arms in order;
held-out loss and token accuracy check training, while the exported draft needs
separate token/state parity and throughput checks.

### Multiple GPUs: separate capture and trainer pools

![Disaggregated target capture and multi-rank draft training with held-out evaluation.](_static/diagrams/family-disaggregated.svg)

In a disaggregated online producer/consumer run, the target producer and draft trainer use
separate GPU resources; the selected deployment config determines rank and
device placement. The current online path records training metrics and
checkpoints, but does not wire a held-out evaluation source into that run. Use
offline held-out features for that gate. Offline multi-GPU evaluation reduces
counts across ranks before reporting ratios. See the [disaggregated training guide](basic_usage/disaggregated_training.md)
and the source-tree architecture reference, `draftfit/runtime/ARCHITECTURE.md`.

For either layout, compare the exact exported artifact with target-only and
the official draft on the same prompts and serving settings. Record token IDs,
accepted length, latency, throughput and peak memory; a lower training loss is
not a serving speed result. The MiniCPM5 outcomes and missing gates are in the
[experiment record](RECENT_MODEL_EXPERIMENTS.md).

## Status of the broader family proposal

| Claim | Status |
| --- | --- |
| One offline cache and sequential DSpark/DFlash/DFlash2 runs, per-arm checkpoints and held-out metrics | Implemented in [`draftfit/family.py`](../draftfit/family.py), subject to matching feature contracts. |
| Same-method warm start, guarded cross-method weight handoff, DSpark head-only/EAL/WTV and DFlash DPACE experiment settings | Implemented in the current code; the [MiniCPM5 record](RECENT_MODEL_EXPERIMENTS.md#gates-and-observations-2026-09-30) separates CPU checks from pending real training and serving. |
| Shared-backbone joint training from scratch or a compatible seed | Implemented with `--joint`; separate method-specific parameters and exports. Matching feature and geometry contracts are required. |
| Bounded feature queue, milestone verification and unattended policy | Implemented for the supported native path; see [limits](UNATTENDED_NATIVE.md). Production-speed selection and automatic architecture/head search are not supported. |
| One checkpoint directly reusable as every serving method | Not supported. Shared training does not remove method-specific serving behavior. |
| DFlow, Carryover, DScale and DPara | Not additional trainable families in this joint/unattended interface. Research helpers do not establish end-to-end integration or serving certification. |
| Universal model support or speedup | Unverified. Exact target revision, capture backend, draft and serving backend must pass real artifact reload, token/state checks and workload A/B measurement. |

The implementation boundaries are defined by
[`convert_family_draft_config` and `warm_start_draft_model`](../draftfit/training/model_loading.py),
[`prepare_project`](../draftfit/target_project.py) and the
[`family` CLI](../draftfit/cli.py). Original-weight transfer checks and later bounded joint-training checks have
different scopes. Neither proves a quality-preserving conversion or speedup.
DFlash2 CUDA evidence applies only to its recorded target/runtime;
see [model validation](MODEL_VALIDATION.md).
