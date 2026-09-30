# Customize a speculative draft, without reimplementing the target

Use `draftfit` for the product CLI. Convert local OpenAI/ShareGPT data with `draftfit data prepare
--input data.jsonl --output train.jsonl --split-eval --eval-output holdout.jsonl`.
Splits are approximate five percent, grouping identical prompt contexts to avoid
leakage. Unsupported semantic fields fail explicitly instead of being dropped.

For DFlash2 select `--strategy dflash2` in target preparation. Its draft includes
grouped convolution and a candidate selector; use full vocabulary, eager/SDPA/flex
attention, and homogeneous full or sliding layers. Adjust
`training.dflash2_selector_loss_alpha` (positive and finite). HF export is for
weights-only fine-tuning reload; SGLang export converts to its serving schema and
requires unit input embedding scale. Actual backend/GPU certification is separate.

Measure held-out text conversations with `draftfit benchmark --model /path/to/target
--data-path holdout.jsonl --output-json result.json`. This contacts the specified
server; use a trusted endpoint. Tool schemas/calls are rejected by this text-only
benchmark. Compare identical prompts and settings in target-only and draft-enabled
runs; do not infer speedup from acceptance alone.

See [CPU installation/release checks](PUBLIC_RELEASE.md) and
[supported versus unverified combinations](PUBLIC_SUPPORT.md).

The target model stays frozen. Users choose a supported target, capture its
features, and train or fine-tune a draft through configuration. A recipe is a
starting point, **not evidence that its model/runtime combination has passed GPU
validation**. Ling's completed long training is separate from this platform's
short functional tests; repeating it is not an onboarding requirement.

## Install and obtain editable recipes

Build a wheel using the [uv environment guide](ENVIRONMENT.md).
In an **already provisioned, compatible training environment**, install that wheel
with `uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps /absolute/path/to/draftfit-<version>-py3-none-any.whl`.
`--no-deps` does not provision missing dependencies or make an arbitrary machine
compatible. The package metadata's general SGLang/Torch pins are not the verified
Ling runtime: do not let a routine dependency install replace the Ling image's
backend. See [runtime profiles](RUNTIME_PROFILES.md) and the matching patch README
in the source checkout before GPU execution. Model/backend patches are not
silently applied by installing the wheel.

Recipe access itself uses only Python's standard library and works without a
GPU, Torch, SGLang, a source checkout, or a network connection:

```sh
python -m draftfit.assets list
python -m draftfit.assets export ./my-draft-project
cd ./my-draft-project
```

Export requires a **new** destination; it never overwrites customized files.
It preserves `configs/...` and `examples/configs/...`, so example draft paths
resolve when commands run from this project directory. Both installed wheels
and editable checkouts use the same source recipes. Runtime profile YAML files
are lists of dotted overrides, not standalone training configs.

## Inspect, then choose a recipe

With the CLI dependencies available, inspect a downloaded model directory first:

```sh
draftfit target inspect /absolute/path/to/target --local-only
```

For a Hugging Face repository, use `draftfit target inspect ORG/MODEL
--revision EXACT_COMMIT`. Inspection reads metadata rather than model weights;
architecture classification and suggested taps are not real feature/state
parity checks. Pin the actual weights/tokenizer used for capture to that same
revision. Review the tokenizer's chat template, embedding/head names, feature
tap positions, and all required validation gates before spending GPU time.

Generate an editable project from target metadata without executing its model code:

```sh
draftfit algorithms
draftfit target prepare /absolute/path/to/target --local-only \
  --strategy dspark --hidden-states /absolute/path/to/features \
  --output-dir ./my-custom-draft \
  --set model.draft_num_hidden_layers=5
draftfit train -c ./my-custom-draft/train.json --plan
```

This writes `train.json`, `draft.json`, `inspection.json`, and a manifest into a
new directory, refusing overwrites. Remote targets are resolved to an immutable
revision used by owned runtime loaders. External servers still need an independent
revision check. Generated layer taps are suggestions: explicitly override
`model.target_layer_ids=[...]` after inspecting the target's feature semantics.

Replace `--hidden-states` with `--train-data /absolute/path/to/data.jsonl` for the
managed GPU-0 capture / GPU-1 training topology. This requires the matching patched
capture backend and transport; preparation does not start either process.
Raw JSONL uses a `conversations` array per row (not a top-level `messages` key).
Preparation streams every row, rejects empty/malformed data and missing assistant
responses, and rejects known HF templates without generation spans. This is
structural validation; actual tokenization masks still require the CPU renderer
test on the chosen tokenizer. Preformatted or multimodal datasets need a
separate supported workflow, not this text project generator.
EAGLE3 online runs additionally require `--set model.vocab_mapping_path=...`.
The generic `hf` renderer requires native assistant `{% generation %}` spans in
the tokenizer template; otherwise select a registered renderer or supply correctly
masked pretokenized data. Template presence alone is not mask validation.

Use `--draft-checkpoint /absolute/path/to/export` for weights-only fine-tuning,
or `--draft-config /absolute/path/to/draft.json` for an explicit draft structure.
Existing draft dimensions are preserved unless explicitly overridden. Domino
requires an explicit draft config; the catalog describes each algorithm's actual
options, not blanket model support. No target architecture is copied into the draft.

Alternatively choose a matching file under `examples/configs/`. Do not use another model's
hidden size, vocabulary, layer indices, or draft JSON unchanged. For an
unrepresented target, the `target scaffold` command can consume saved inspection
JSON; inspect its output and required gates rather than assuming support from
successful config generation.

## Customize and preflight

For example, `examples/configs/qwen3-8b-dspark-offline.yaml` references
`configs/qwen3-8b-dspark.json`. Edit the selected run file's target and data paths,
output directory, training options, and the referenced draft config together.
The offline example requires **previously captured features**; planning it does
not create them. Online recipes require a compatible capture server and transport.

```sh
draftfit train -c examples/configs/qwen3-8b-dspark-offline.yaml --plan
```

Plan validation does not load real weights or prove a kernel/backend is usable.
Start a new output directory with a short real run before increasing the budget:

```sh
draftfit train -c examples/configs/qwen3-8b-dspark-offline.yaml \
  training.max_steps=5 training.save_interval=5 \
  output_dir=./outputs/first-smoke
```

Customize learning rate, epochs, sequence length, and applicable draft depth,
block size, taps, and anchor count. These options are algorithm-specific: simply
changing `training.strategy` while leaving an incompatible draft config or
feature cache is not a conversion. Regenerate features when their contract
changes, and use a distinct output directory for each experiment.

## Fine-tune weights versus resume a run

- `model.draft_checkpoint_path=/absolute/path/to/draft` is a **weights-only warm
  start**. Use the matching draft architecture/config. It starts a new optimizer,
  schedule, step counter, and data progression; incompatible weights must fail.
- `training.resume_from=/absolute/path/to/checkpoint` restores a **training run**.
  Preserve its optimizer/checkpoint payload and required configuration. A draft
  export alone is not a full checkpoint.
- Do not set both options. Cross-world-size optimizer resharding is unsupported;
  a complete infrastructure-loss recovery is a separate gate from restarting
  one trainer while its feature server is still alive.

## Export and verify the actual exported artifact

Use the completed checkpoint directory and the exact draft JSON used to train:

```sh
draftfit export --to hf \
  --checkpoint /absolute/path/to/completed-checkpoint \
  --draft-config /absolute/path/to/draft-config.json \
  --output-dir ./exports/my-draft
```

Depending on the algorithm, export may additionally require the target's frozen
embedding (`--embedding-source`, `--embedding-key`) or a vocabulary mapping.
Consult `draftfit export --help`; a successful file conversion is not an
inference compatibility claim. Reload this exact export in its supported
serving backend and compare target tokens/state on held-out prompts. Report
functional correctness separately from speculative acceptance and speedup.

## Benchmark your workload

Against an already-running compatible server:

```sh
draftfit benchmark --model /path/to/target \
  --data-path ./data/holdout.jsonl --num-prompts 100 \
  --max-new-tokens 128 --concurrency 1 \
  --base-url http://127.0.0.1:30000 --output-json ./baseline.json
```

Repeat with the exported draft enabled, writing a different report. Keep prompts,
generation settings, concurrency and hardware fixed. The command preserves
conversation context and removes the final assistant reference; it sends prompts
to the supplied server, so use a trusted endpoint. It neither launches a server
nor certifies token/state correctness. Tool schemas and tool-call messages are
rejected by this text-only benchmark.

The output path must be new and its parent must exist. Handled failures remove
an incomplete report; abrupt termination may leave a partial file that is not a
completed measurement. Compare results using the [performance guide](PERFORMANCE_GATE.md).

## Honest release boundaries

The existing bounded H200 evidence covers Ling online capture on GPU 0 plus
draft training on GPU 1, a short online/offline tensor comparison, native Ling
state replay, and trainer checkpoint recovery. It does not validate every model,
sequence length, cache mode, concurrent workload, or exported artifact. A new
target needs real capture, short training, export reload, and serving/state
checks before being called supported. Metadata-only fixtures remain structural
tests. No GPU rental or cloud deployment is performed by the recipe exporter.

Contributors can reproduce the backend-free packaging gate with:

```sh
python -m unittest tests.test_packaging.test_assets -v
```

It builds a clean wheel without downloading dependencies, imports its assets in
an isolated interpreter, verifies recipe bytes, and checks overwrite refusal.
