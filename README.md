# Draftfit

**Your workload. Your draft.**

Source release: **0.1.0** · [Changelog](CHANGELOG.md) · [Support boundaries](docs/PUBLIC_SUPPORT.md)

Train and fine-tune speculative decoding drafts for your target model and your
own data. Draftfit brings conversation preparation, draft training, checkpoint
recovery, export and workload evaluation into one workflow.

Built for people running personal agents, teams serving repeated task patterns,
and researchers comparing draft methods. Your target stays frozen; you customize
the smaller draft that proposes tokens for it to verify. Better acceptance can
help, but speedup must be measured against target-only inference on your workload.

## How it works

1. **Prepare your data.** Normalize conversations and keep a separate evaluation set.
2. **Choose your target and method.** Generate an editable configuration or start from an existing draft checkpoint.
3. **Train.** Consume saved target features, or stream them from a live inference server to separate training workers.
4. **Export and measure.** Reload the draft and compare correctness and throughput with target-only inference.

The target stays frozen. The inference backend runs its architecture; this platform trains the draft. Dense, MoE, hybrid and recurrent targets need compatible capture and serving implementations—not another target implementation in the trainer. A generated recipe is not a compatibility guarantee.

## Setup

Use a Linux GPU environment matching your target backend. Start with the [runtime profiles](docs/RUNTIME_PROFILES.md) and their capture requirements; installing this package does not patch an inference server.

```sh
git clone https://github.com/JayYun98/dspark-train-platform.git
cd dspark-train-platform

# In an activated, provisioned training environment, preserve backend versions:
uv pip install --python "${VIRTUAL_ENV:?Activate the runtime environment}/bin/python" --no-deps --no-build-isolation -e .
draftfit algorithms
```

`--no-deps` assumes the required dependencies are already installed. Environment
creation, installation and builds use [uv profiles](docs/ENVIRONMENT.md). Activate
the selected environment before running the CLI examples below. For CPU-only
preparation and contributor checks, use the hashed CPU profile in that guide.

## Quick start

Convert your OpenAI `messages` or ShareGPT conversations into training JSONL,
with a deterministic held-out split:

```sh
draftfit data prepare --input ./my-conversations.jsonl \
  --output ./data/train.jsonl --split-eval --eval-output ./data/holdout.jsonl
```

The converter validates text conversations locally and refuses to overwrite files.
Splitting requires at least two distinct prompt contexts. Repeated contexts stay
in the same split; unsupported semantic fields are rejected rather than dropped.

Inspect a downloaded target, or use a Hugging Face model ID with an exact revision:

```sh
draftfit target inspect /path/to/target --local-only
draftfit target inspect ORG/MODEL --revision EXACT_COMMIT
```

Prepare a DSpark run from conversation JSONL. Each training row contains a `conversations` array with `role` and `content` fields, including an assistant response. The chat template must produce the correct assistant loss mask.

```sh
draftfit target prepare /path/to/target --local-only \
  --strategy dspark --train-data /path/to/train.jsonl \
  --output-dir ./my-draft \
  --set model.draft_num_hidden_layers=5

draftfit train -c ./my-draft/train.json --plan
draftfit train -c ./my-draft/train.json
```

The generated online configuration assigns the live target to **GPU 0** and draft training to **GPU 1**. SGLang is the default teacher; select `--teacher-backend transformers` or `--teacher-backend vllm` for the experimental platform-owned teacher services. All three use the same Mooncake transport and draft trainer, not a TorchSpec/AngelSpec subprocess. Review the plan and backend environment before launch. Preparation refuses to overwrite an existing project.

### Fine-tune an existing draft

Provide its checkpoint during preparation so its existing architecture is used instead of generic defaults:

```sh
draftfit target prepare /path/to/target --local-only \
  --strategy dspark --train-data /path/to/train.jsonl \
  --draft-checkpoint /path/to/draft-export \
  --output-dir ./my-finetune
draftfit train -c ./my-finetune/train.json
```

This is a weights-only warm start with a new optimizer and schedule. To continue an interrupted run instead, set `training.resume_from=/path/to/training-checkpoint`. Do not combine warm start and resume. Optimizer resume requires the same trainer world size.

### Train from saved features

Replace `--train-data` with `--hidden-states /path/to/features` to create an offline run. Caches must match the target revision, tokenizer, capture layers and draft method. See [the workflow guide](docs/PUBLIC_WORKFLOW.md) for capture requirements and overrides.

Offline teachers can use **Hugging Face Transformers**, **vLLM**, or SGLang;
all feed the same PyTorch draft trainer. The new Transformers/vLLM adapters are
experimental, single-device text paths for Llama/Qwen2/Qwen3—not universal model
or serving support. See [teacher setup](docs/ENVIRONMENT.md#offline-teachers).
Native HF/vLLM online capture now uses the same producer/channel/consumer flow;
bounded Qwen3 GPU capture, training and managed CLI runs have passed. HF BF16
feature parity passed; vLLM BF16 cross-engine parity failed, while FP32 parity
with deterministic training passed. These results do not certify serving,
speedup or other targets. Use the [native teacher runbook](docs/OWNED_GPU_ENVIRONMENT.md)
and [recorded limits](docs/MODEL_VALIDATION.md#native-teacher-two-gpu-functional-evidence-2026-09-14).
The existing bounded SGLang/Ling validation remains separate evidence.

### Export

```sh
draftfit export --to hf \
  --checkpoint /path/to/completed-checkpoint \
  --draft-config ./my-draft/draft.json \
  --output-dir ./exports/my-draft
```

Check `draftfit export --help` for method-specific embedding or vocabulary inputs. Export success is separate from serving compatibility: reload the exact artifact and compare it against target-only inference before deployment.

## Draft methods and target support

Run `draftfit algorithms` for available methods and feature contracts. DSpark,
DFlash, DFlash2, EAGLE3, PEagle and Domino have integrated training paths; PEagle
is streaming-only and Domino needs an explicit compatible draft configuration.

### DFlash2

Select `--strategy dflash2` during preparation. The implementation includes grouped
convolution and candidate-selector training, adapted from the pinned TorchSpec
implementation with z-lab/dflash provenance—not a DFlash
alias. Tune `training.dflash2_selector_loss_alpha` and the generated draft config.
It requires the full target vocabulary and homogeneous full or sliding attention;
USP attention and vocabulary pruning are unsupported.

Use `export --to hf` for fine-tuning reload, or `export --to sglang` for the serving
layout (requires `input_embedding_scale=1.0`). CPU tensor updates and export-schema
checks do not certify an inference backend. Bounded real-teacher CUDA training,
resume and export have been exercised; validate the exact runtime and live
serving artifact before deploying a combination.

Compatibility is a combination of **target revision + draft method + capture backend + serving backend + runtime**, not a list of architecture names. The [support matrix](docs/PUBLIC_SUPPORT.md) and [model validation matrix](docs/MODEL_VALIDATION.md) distinguish real-weight tests from synthetic checks and recipes. Bounded Ling results are not universal support or a speedup promise. Native Ling replay on RTX5090 remains disabled following a state-parity failure.

## Customization and evaluation

Benchmark your held-out text conversations against an already-running server:

```sh
draftfit benchmark --model /path/to/target \
  --data-path ./data/holdout.jsonl --num-prompts 100 \
  --max-new-tokens 128 --concurrency 1 \
  --base-url http://127.0.0.1:30000 --output-json ./baseline.json
```

Run again against the same target with your exported draft enabled, writing a
different output file. Keep prompts, generation settings, concurrency and hardware
fixed. The benchmark preserves conversation context and removes the final assistant
reference answer; it sends your prompts to the configured server. Use only a
trusted endpoint. It does not start a server or certify token/state correctness.
This benchmark accepts text-only conversations; tool schemas and tool-call
messages are rejected until their rendering contract is supported.

Output JSON paths must be new and their parent directory must exist. The command
reserves the path before inference and refuses to overwrite prior measurements.
Handled inference/write failures remove its incomplete report; an abrupt process
kill may leave an empty or partial file, which is not a completed measurement.

Edit `train.json` and `draft.json` to tune draft depth, feature taps, block size, supported attention options and training settings. Check finite updates and checkpoint recovery with a short run before scaling the budget. Choose settings using held-out results rather than architecture alone.

For deployment, check token correctness, cache/state behavior and measured throughput separately. A falling loss or longer acceptance length alone does not establish a speedup.

## Development and provenance

For the personal-agent workflow and the boundary between existing training tools
and future trace-driven updates, see [Personal inference optimization](docs/PERSONAL_INFERENCE.md).
Reviewed conversation exports work today; automatic OTel/OpenCodex ingestion and
continual deployment are not yet implemented.

The distribution and primary command are `draftfit`; the Python
package is `draftfit`. The implementation lives in that package:
application composition, draft
algorithms, training, feature transport and teacher services are maintained here.
The legacy `speculative_train_platform`, `dspark` and `specforge` imports resolve
to the same implementation. The former `speculative-train-platform`, `dspark`
and `specforge` commands remain aliases, not separately installed training frameworks.

DSpark names one supported draft algorithm, not the platform. Existing algorithm
settings and checkpoint identifiers remain unchanged. Use a fresh environment
when migrating from `speculative-train-platform`, `dspark-train-platform` or upstream SpecForge; the old and
new distributions must not be installed together because compatibility files overlap.
The repository URL retains its existing name.

The training engine is shipped in this distribution, not installed from upstream
SpecForge. This remains a SpecForge-derived project with preserved attribution,
not a claim that every component was written from scratch.
Custom SGLang and Mooncake remain explicit external runtimes; see
[code ownership and module boundaries](docs/CODE_OWNERSHIP.md).

The platform builds on SpecForge's training engine and adapts selected ideas and implementations from [TorchSpec](https://github.com/lightseekorg/TorchSpec), [AngelSpec](https://github.com/Tencent/AngelSpec), and DSpark-related projects. Upstream licenses and attribution remain intact. See [third-party notices](THIRDPARTY_NOTICES.md) and [source attribution](docs/SOURCE_ATTRIBUTION.md).

- [Detailed workflow](docs/PUBLIC_WORKFLOW.md)
- [Add a target through configuration](docs/TARGET_EXTENSION.md)
- [Compare personal-workload measurements](docs/PERFORMANCE_GATE.md)
- [Production release gates](docs/RELEASE_GATES.md)
- [Runtime profiles](docs/RUNTIME_PROFILES.md)
- [Model environment selection and agent skills](docs/MODEL_ENVIRONMENTS.md)
- [Contributor and release checks](docs/PUBLIC_RELEASE.md)
- [License](LICENSE)
