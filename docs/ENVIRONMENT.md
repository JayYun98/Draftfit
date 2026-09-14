# Environment management with uv

Use uv (0.9.18 or newer); contributor Python is 3.12. The owned distribution
contains the modified SpecForge engine. Do not install upstream SpecForge beside
it: both would own the same import namespace. See [code ownership](CODE_OWNERSHIP.md).

## CPU preparation and contributor checks

From this checkout:

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-cpu.lock \
  --torch-backend cpu --require-hashes
uv run --no-project --python .venv/bin/python python -m tests.public_cpu
uv build --python .venv/bin/python --no-build-isolation
uv pip install --python .venv/bin/python --no-deps \
  dist/dspark_train_platform-0.2.0-py3-none-any.whl
uv run --no-project --python .venv/bin/python python -I scripts/check_public_install.py
```

Use the current wheel version if it changes. `requirements-cpu.in` and its
generated hash lock are the CPU profile, not a GPU profile. Regenerate explicitly:

```sh
uv pip compile requirements-cpu.in --python-version 3.12 --universal \
  --torch-backend cpu --default-index https://pypi.org/simple \
  --generate-hashes -o requirements-cpu.lock
```

`uv pip sync` plus a hash lock is the selected environment workflow. There is no
universal `uv.lock` or supported bare `uv sync` command: CPU and custom CUDA
backends intentionally have different dependency sets. `uv run --no-project`
uses the explicit interpreter without trying to resolve/synchronize the project
and replace a backend. Never bypass a hash mismatch. See [installation evidence](PUBLIC_RELEASE.md).

## GPU: preserve the tested custom runtime

Provision the backend from the relevant [runtime profile](RUNTIME_PROFILES.md)
or pinned custom image first. Inside that image, use its actual Python path:

```sh
# Replace /opt/runtime/bin/python with the interpreter supplied by the image.
uv pip install --python /opt/runtime/bin/python --no-deps --no-build-isolation -e .
uv run --no-project --python /opt/runtime/bin/python python -m specforge.cli algorithms
```

`--no-deps` is intentional only for an already provisioned runtime, not a complete
installation recipe. The custom SGLang capture build and Mooncake remain external
runtime dependencies. Installing the platform must not silently replace the
patched server with PyPI SGLang. The optional `sglang` extra selects stock 0.5.14
on Linux only and does NOT provide the Ling capture patch or certify DFlash2 serving.

For a new custom backend build, record its source commit, patch digest, image or
wheel SHA256 and dependency inventory; use that immutable artifact in the runtime
profile. Keep the inference server's build separate from the owned training wheel.
Do not invent an artifact URL or label an unbuilt recipe as a published build.

## Offline teachers

SGLang is not required by the shared draft trainer. Select the engine that
generates features, then train from its saved files. Here “Transformers” means
Hugging Face Transformers with PyTorch, not TensorFlow; vLLM is an inference
engine, not a gradient-training backend.

| Teacher | Current boundary | Validation |
| --- | --- | --- |
| SGLang | Existing pinned capture profiles, including bounded Ling support | See model validation records |
| Transformers | Single-device Llama/Qwen2/Qwen3 text decoders; eager/SDPA | Actual tiny CPU capture → disk → loader → DSpark/DFlash2 update |
| vLLM | Experimental native `extract_hidden_states` + `ExampleHiddenStatesConnector`; same dense families, TP1/PP1, single process | Boundary tests only; real GPU parity pending |

Neither new adapter implements live server-to-Mooncake streaming yet. Setting
`model.target_backend=transformers` or `vllm` is accepted for offline runs and
rejected for online runs. Both reuse the existing offline algorithm contracts;
PEagle remains streaming-only. Export/serving support is a separate check.

Use the CPU profile above for local Transformers tests, or a provisioned GPU
Transformers environment. For vLLM, provision a separate compatible vLLM runtime
first (source API reviewed at **0.22.1**, not a certified image), then install the
platform without replacing that runtime. Do not install vLLM into the locked CPU
profile or into the patched Ling environment. Record the actual version, model
revision, image digest and dependency inventory before GPU validation.

Given an existing compatible draft config and reviewed conversation dataset:

```sh
# Run from this checkout in the selected, activated teacher environment.
uv run --no-project python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/prepare_hidden_states.py \
  --target-backend transformers \
  --target-model-path /path/to/target \
  --draft-model-config /path/to/draft.json --strategy dspark \
  --data-path ./data/train.jsonl --chat-template qwen \
  --output-path ./features/transformers --max-length 2048 \
  --tp-size 1 --batch-size 1 --torch-dtype bfloat16
```

For CPU capture set `SPECFORGE_DEVICE=cpu` and `--torch-dtype float32`. For vLLM
use `--target-backend vllm`, a new output directory and a CUDA runtime; optionally
set `--vllm-gpu-memory-utilization 0.4`. Keep one process and TP1. For remote model
IDs pass `--target-revision EXACT_COMMIT`; config, tokenizer and weights share it.
`--dry-run` resolves the contract without loading weights.

The adapters capture post-block, pre-final-norm auxiliary states. The final
target state is normalized separately; vLLM's last capture slot is **not** already
normalized. Unsupported architectures, invalid taps, token mismatches and
non-finite tensors fail rather than falling back to a different teacher. vLLM
requires safetensors target weights for loading the final norm. Intermediate
connector files are private temporary files, not a scalable streaming transport.

Create the training run from those files:

```sh
dspark target prepare /path/to/target --local-only --strategy dspark \
  --hidden-states ./features/transformers --output-dir ./offline-draft \
  --draft-config /path/to/draft.json \
  --set model.target_backend=transformers --set data.max_length=2048
dspark train -c ./offline-draft/train.json --plan
dspark train -c ./offline-draft/train.json
```

For vLLM change both feature path and backend. Keep the draft config, revision,
template and capture layers identical. Use a new output directory when changing
teacher settings; do not mix feature files from different producers.

Before advertising a new backend as validated, compare real HF/vLLM features on
identical token IDs (including last-layer taps and padding), train/resume/export,
and test the exported draft in a compatible serving engine. Tiny CPU tests do
not establish GPU correctness, hybrid-state replay or inference speedup.

Reference: [vLLM native extraction](https://github.com/vllm-project/vllm/blob/v0.22.1/docs/features/speculative_decoding/extract_hidden_states.md).

## Maintenance rules

- Use uv for environment creation, dependency synchronization, builds and installs.
- Keep historical experiment commands unchanged as evidence, not onboarding advice.
- Preserve upstream attribution in absorbed source; namespace compatibility is not
  a dependency on a separately installed upstream package.
- Refactoring metadata/preflight code does not invalidate unrelated recorded GPU
  results. Repeat only the affected runtime gates when execution behavior changes.
