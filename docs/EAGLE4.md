# Official EAGLE4 checkpoint adaptation

Draftfit supports official EAGLE4 checkpoints as training inputs.
“Official” describes the source checkpoint; Draftfit is an independent integration.
The original model license and attribution still apply to derived weights.

## Choose the training stage

| Stage | Updated parameters | Preserved parameters |
| --- | --- | --- |
| `training.eagle4_stage: draft` | Recurrent draft block, feature projection, normalization, and dense output head | Target embedding and learned low-rank candidate factors |
| `training.eagle4_stage: candidates` | Low-rank candidate factors | Embedding and dense draft body/head |

Full-draft adaptation is the `draft` stage. It is not head-only training.
Keep `training.eagle4_initialize_candidates: false` when adapting an official checkpoint.
Enabling that option explicitly rebuilds candidate factors with SVD; it is not a routine loading step.

## Preserve the source architecture

Download and pin the target and its compatible official draft before training.
For example, [EAGLE-Corp/Qwen3.8-27B-EAGLE4](https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4)
provides an EAGLE4 draft for Qwen3.8-27B.
A draft for another target is not automatically compatible.

```sh
draftfit target prepare /path/to/pinned-target --local-only \
  --strategy eagle4 --draft-checkpoint /path/to/official-eagle4 \
  --hidden-states /path/to/train-features --output-dir ./eagle4-run
```

The checkpoint supplies its draft configuration when no explicit draft config is given.
Keep its tap order, vocabulary, rotary settings, sliding window, and candidate-head dimensions.
Draftfit maps official `layers.0.*` weights to its recurrent `midlayer.*` names during loading.
It restores nonpersistent rotary and vocabulary buffers after loading.
The draft retains its own rotary dtype policy; blanket conversion to FP32 is not a parity fix.
The target's FP32 rotary buffers must not be rounded by a blanket target dtype cast.

`model.draft_checkpoint_path` loads model weights and starts a new optimizer and schedule.
`training.resume_from` restores the same run, including optimizer, schedule, step, and data state.
Do not combine these two initialization routes.

## Native-logit features

The ordinary saved-feature path projects final hidden states through the target head.
For quantized targets, that projection can differ from the actual inference runtime.
Use `target_repr: native_logits` artifacts when exact native teacher distributions are required.
They enter training directly, without an external target-head projection.

Each saved sample contains these tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `input_ids` | `[N]` | Exact original token sequence |
| `loss_mask` | `[N]` | Binary mask of supervised **output** positions |
| `aux_hidden_state` | `[N, 3H]` | Three ordered post-block target taps |
| `native_logits` | `[N-start, V]` | Finite, detached native full-vocabulary logits |
| `native_logit_start` | Scalar int64 | Original position of the first stored logit; `1 <= start < N` |

A logit at original position `j` supervises output position `j-1` after the input shift.
The trainer applies this mapping once. Existing online `logits` batches are already shifted and retain their previous contract.
The mask must exclude uncovered positions and the final output position.
Invalid shapes, missing supervision coverage, nonfinite logits, and incompatible vocabulary sizes fail before loss computation.

An external capture adapter can publish the artifacts with:

```python
from draftfit.algorithms.eagle4.native_logits import save_native_logits_features

save_native_logits_features(
    "./native-features", samples,
    target_model="/path/to/pinned-target",
    target_revision="pinned-revision",
    target_hidden_size=5120,
    target_vocab_size=248320,
    capture_layers=[3, 31, 59],
    capture_backend="external-native-runtime",
    capture_settings={"runtime_revision": "pinned-build", "logit_dtype": "bfloat16"},
    target_quantization=quantization_identity,
)
```

Here `samples` contains the tensors above. The dimensions are specific to the example target, not universal defaults.
Supply the actual runtime settings and quantization identity from your capture adapter.
The writer publishes a capture manifest only after valid samples are saved.
Use a new directory and separate train/holdout captures.
The writer checks artifacts; it does not run an inference backend or establish numerical capture parity.
Compare native logits, labels, tap positions, and token history against the actual target before a large run.
Store only the supervised suffix to bound memory.

## Precision

Default `eagle4_precision: torch` uses the existing PyTorch implementation.
An explicit `eagle4_precision: vllm` in the **draft model configuration** enables native arithmetic helpers.
These cover fused projections, native RMSNorm, and rotary operation ordering.
They require a compatible CUDA/vLLM installation; this is not a CPU or generic backend switch.
Keep the same setting in the saved draft configuration for training and reload comparisons.

Teacher probabilities normalize in FP32 before loss computation.
Vocabularies above the fused kernel limit use the differentiable FP32 loss path.
Native BF16 logits must retain their original values before normalization.
A projected or higher-precision recomputation can change argmax decisions at ties.

## Warmup–stable–decay scheduling

The [example configuration](../examples/configs/eagle4-official-continual.yaml) uses full-draft adaptation with a fixed 3,000-update horizon.
Set these training fields:

```yaml
learning_rate: 0.000003
lr_scheduler: wsd
warmup_ratio: 0.05
lr_decay_ratio: 0.20
min_lr_ratio: 0.10
lr_decay_style: linear
```

This allocates 150 warmup updates, 2,250 stable updates, and 600 decay updates.
The learning rate rises from `3e-7` to `3e-6`, stays at peak, then returns to `3e-7`.
`lr_decay_style: cosine` selects a half-cosine decay instead.
Existing `cosine` and `constant` configurations retain their previous behavior.

The optimizer owns scheduler advancement: one scheduler step per optimizer update.
Do not advance it again after gradient accumulation or an outer training step.
Exact resume validates the saved horizon, phase lengths, floor, and decay style.
Changing the horizon at every evaluation changes the experiment and is rejected during exact resume.

These values are a reproducible starting recipe, not a generally optimal learning rate.
Compare changes from the same initial weights, data order, supervision, and optimizer settings.

## Evaluate, select, and export

1. Check zero-update checkpoint reload and forward parity before training.
2. Train and save checkpoints at fixed milestones.
3. Compare dense and candidate proposal heads separately when diagnosing acceptance.
4. Select a checkpoint using a predefined screening set.
5. Evaluate the fixed selection on an untouched confirmation set.
6. Export and reload the selected weights before publication.

`benchmarks/evaluate_eagle4_comparison.py` accepts `--eagle4-head candidates` or `--eagle4-head dense`.
Candidate-mode replay and dense-head replay measure different proposal paths.
Loss and teacher-forced token accuracy are diagnostics; they are not rollout acceptance.
Known-trajectory replay does not measure live serving throughput.
Report the actual exported update, even when a later checkpoint completed more updates.

Use `draftfit export --help` and the [workflow guide](PUBLIC_WORKFLOW.md) for export commands.
A self-contained HF export needs the frozen target embedding when the training checkpoint omits it.
SGLang export deliberately shares that embedding with the target instead.
HF export retains Draftfit `midlayer.*` tensor names for reload and further training.
An official engine expecting `layers.0.*` needs an explicit key conversion and runtime validation.
Check preserved candidate factors and exported logits after reload.
Measure live target-only and target-plus-draft execution separately before claiming a speed gain.

## Regression checks

```sh
python -m unittest tests.test_modeling.test_eagle4 tests.test_modeling.test_eagle4_precision \
  tests.test_eagle4_native_logits tests.test_optimizer.test_wsd \
  tests.test_eagle4_comparison tests.test_eagle4_export
python -m tests.public_cpu
```

These checks cover loading, alignment, scheduling, loss arithmetic, and tiny-model export contracts.
They do not qualify a new native capture build, real-weight GPU training, or a live serving engine.
