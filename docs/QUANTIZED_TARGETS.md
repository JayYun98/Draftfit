# Train a draft for a quantized target

A draft trained against a floating target can see different features after target quantization.
Capture new features from the deployed quantized target, then train the draft against those features.
You can initialize from an existing compatible draft or create a new draft.
The target remains frozen. The draft remains floating point.

This workflow reuses the registered algorithms, losses, checkpoint reader, and trainer.
It does not quantize draft weights or train the target itself.
A target model name alone does not establish backend support.

## Choose the target load

| `model.target_quantization` | Behavior |
| --- | --- |
| `auto` (default) | Read the checkpoint's `quantization_config`. Preserve its packed runtime load. Dense checkpoints remain dense. |
| `bitsandbytes-4bit` | Load a floating Transformers decoder in NormalFloat4 (NF4), using the requested compute dtype. |
| `bitsandbytes-8bit` | Load a floating Transformers decoder through LLM.int8. |

Explicit recipes require `model.target_backend=transformers` and a floating source checkpoint.
Use `auto` for an already quantized checkpoint. Requantization is rejected.
The loader places packed modules directly on one device. It does not call `.to()` afterward.

Provision the target engine first. In a compatible Linux runtime, add the optional dependency without replacing the engine:

```sh
uv pip install --python /opt/runtime/bin/python --no-deps 'bitsandbytes==0.50.2'
```

The `quantization` package extra declares the same optional library.
Its CUDA and hardware requirements still apply.
Transformers capture preserves the decoder layouts already registered on `dev`.
Each architecture and quantization combination still requires its own runtime validation.
Experimental vLLM capture accepts prequantized checkpoints through `auto`, within its existing pinned runtime and architecture limits.
The vLLM engine must support the checkpoint format; no dequantization fallback is provided.
SGLang keeps its existing checkpoint and `model.sglang_quantization` behavior.
Its custom capture build remains required.

**Embeddings and the output head must remain floating point.**
The trainer loads these frozen tensors from the same target checkpoint.
Packed embeddings or output heads are unsupported and fail during loading.
Do not substitute a head from another target to bypass this limit.
Existing explicit Prism/GGUF packed readout adapters retain their separate contracts.
Native NVFP4 execution, hybrid targets, and untested quantization formats remain separate validation gates.

## Prepare either initialization

The examples use a local floating target and quantize its decoder during capture.
Replace each path with your actual target, data, and draft.
For a prequantized target, select its checkpoint and use `--target-quantization auto` throughout.
Use an immutable `--revision COMMIT_SHA` for remote targets.

To adapt an existing draft:

```sh
python -m draftfit.cli target prepare /models/target --local-only \
  --strategy dspark --teacher-backend transformers \
  --target-quantization bitsandbytes-4bit \
  --draft-checkpoint /models/existing-draft \
  --hidden-states ./features/quantized --output-dir ./adapted-draft \
  --set model.torch_dtype=bfloat16 --set training.attention_backend=sdpa
```

To create a new draft, omit `--draft-checkpoint`:

```sh
python -m draftfit.cli target prepare /models/target --local-only \
  --strategy dflash2 --teacher-backend transformers \
  --target-quantization bitsandbytes-4bit \
  --hidden-states ./features/quantized-fresh --output-dir ./fresh-draft \
  --set model.torch_dtype=bfloat16 --set training.attention_backend=sdpa
```

Use `--draft-config` when the selected algorithm requires explicit architecture metadata.
Weights-only initialization resets optimizer, scheduler, counters, and data position.
`training.resume_from` continues the same run and cannot be combined with a draft checkpoint.
Keep tokenizer, vocabulary mapping, capture layers, and draft architecture compatible.

## Capture and train

Capture features into a fresh directory, using the project's generated `draft.json`:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m draftfit.offline_capture.prepare \
  --target-backend transformers --target-quantization bitsandbytes-4bit \
  --target-model-path /models/target \
  --draft-model-config ./adapted-draft/draft.json --strategy dspark \
  --data-path ./data/train.jsonl --chat-template qwen \
  --output-path ./features/quantized --max-length 1024 \
  --tp-size 1 --batch-size 1 --torch-dtype bfloat16
python -m draftfit.cli train -c ./adapted-draft/train.json --plan
python -m draftfit.cli train -c ./adapted-draft/train.json
```

Change the project, strategy, and feature paths together for the fresh example.
Use a chat template supported by the target tokenizer.
These large-model commands are usage templates, not completed quality experiments.

The capture manifest records the target quantization recipe and checkpoint metadata.
Training checks both train and evaluation feature directories before model assembly.
Missing manifests, different target quantization, target revisions, capture backends,
compute dtypes, or Transformers attention settings fail explicitly.
Dense legacy caches remain readable for dense runs.
Do not copy a manifest onto old floating-target features.

For managed online capture, use `--train-data` instead of `--hidden-states` during preparation.
The launch plan passes the quantization recipe to the owned Transformers teacher service.
The existing two-GPU, Mooncake, and service supervision requirements remain unchanged.

## Check the result

Training checkpoints include target quantization in resume provenance.
Exported draft configuration carries `draftfit_target_quantization` as deployment metadata.
It does not set the draft's standard `quantization_config` or pack draft tensors.

Use the existing export and serving workflow with the exact quantized target runtime.
Compare the original and adapted drafts on held-out prompts against that same target.
Report loss, acceptance, strict token correctness, and measured speed separately.
Do not infer better acceptance or speed from a successful optimizer update.
See the [spec and evidence](specs/quantized-target-adaptation.md).

Real NF4/INT8 GPU execution and serving remain unverified for this change.
The bounded GPU smoke stopped before model construction because its isolated runtime lacked Transformers.
Dependency setup then failed because the instance disk was full.
See the [verification record](reports/quantized-target-adaptation.md) for completed CPU checks and open gates.

References: [Transformers bitsandbytes loading](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
[vLLM hidden-state extraction](https://docs.vllm.ai/en/v0.22.1/features/speculative_decoding/extract_hidden_states/).
