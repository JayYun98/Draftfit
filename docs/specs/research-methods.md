# Selected native research methods

## Scope

The later [latest-method integration](../research/latest-speculative-methods.md) adds
NgramRL, SpecMem, ValuSpec-on-CAST, RIFT and ARR reference execution, plus
TreeWY numerical, MoE-SpAc policy and query-layout diagnostics. Its separate
support matrix and recipes state the exact implementation boundaries below.

Integrate additive ALR/IRA target-rollout distillation, Osprey backbone conversion,
CAST tree decoding, and Redline calibration on the existing Draftfit interfaces.
The selected source is the read-only September integration worktree's dirty
changes, adapted to development base `53370e2`. The implementation does not port
its queue, controller, milestone, automatic batching or external engine recipes.
Existing development training and GPU execution paths retain their contracts.

| Method | Entry point | Output and boundary |
| --- | --- | --- |
| ALR/IRA | `research` preparation, then training strategy `alr_ira` | Target-generated rollout cache; sparse KD over standard DFlash tensors |
| Osprey | `research` conversion, then training strategy `osprey` | Registered Qwen3/Llama draft with its own aligned embedding |
| CAST | `research` recipe or `draftfit.research.cast.decode.cast_generate` | Batch-one reference tree verification with corrected sampling |
| Redline | `research` recipe using `draftfit.benchmarks.redline.select` | Configuration selection from paired correctness and measured cost |

The registry and source catalog remain importable without loading Torch. Recipe
configuration rejects extra keys and every unselected method. Planning loads no
models, creates no output directory and does not certify model compatibility.
Execution requires a fresh output directory. Preparation/conversion stages in a
sibling temporary directory; failed work publishes no partial destination.

## Recipe interface

The shared handler is available through `python -m draftfit.research` and the
main CLI's `research` dispatch:

```sh
python -m draftfit.research --sources
python -m draftfit.research -c recipe.json --plan
python -m draftfit.research -c recipe.json runtime.device=cpu
```

Recipes may be JSON or YAML. Dotted overrides are revalidated. Their separate
schema does not change existing train recipes. Source metadata lists the original
four projects plus the latest-method ledger; inherited licenses are retained in
`draftfit/research/NOTICE.md` and relevant source headers.

## ALR/IRA preparation and training

```json
{
  "method": "alr_ira",
  "model": {"target_model_path": "local-target", "torch_dtype": "float32"},
  "data": {"input_path": "reviewed-token-rows.jsonl", "max_samples": 32},
  "training": {
    "block_size": 16,
    "target_layer_ids": [1, 3, 5],
    "anchors_per_sample": 2,
    "in_rollout_anchors": true,
    "max_length": 2048
  },
  "runtime": {"device": "cpu", "seed": 42},
  "output_dir": "alr-features"
}
```

Input JSONL rows contain integer `input_ids` and an equally sized binary
`loss_mask` from reviewed assistant spans. A primary anchor precedes a supervised
corpus token. The producer greedily generates up to `block_size - 1` target
continuations, then captures post-block/pre-final-norm taps and final normalized
states in a final target forward. It does not mutate the input corpus. IRA may
choose a secondary anchor inside that same rollout and reuse identical computed
features without further target forwards. Stop IDs can shorten a rollout.

This is a full-prefix reference producer for dense Llama/Qwen2/Qwen3 targets.
It intentionally requires more forwards than a cached implementation. It records
a capture manifest and per-block preparation receipt. The local fallback revision
hashes target configuration, not weights; callers must pin and preserve actual
target weights when stronger provenance is required. An explicit revision is
checked against the training recipe, not independently authenticated.

Train a standard DFlash architecture with `training.strategy=alr_ira`, the
prepared `data.hidden_states_path`, matching target path/revision, taps, mask ID
and block size. Generic offline teacher capture cannot substitute for this cache.
Training validates its manifest, uses eager/SDPA full attention, and requires
`training.loss_type=dflash`. The sparse objective harvests teacher top-k from the
state immediately preceding each supervised rollout token. Root and padding slots
have zero weight; draft attention cannot read target features at or after its
anchor. Existing optional loss decay weights remain active.

Only these training fields are added:

| Field | Default | Meaning |
| --- | --- | --- |
| `alr_topk` | 8 | Positive target support size, clamped to vocabulary size |
| `alr_hard_label_mix` | 0.0 | Finite fraction in [0, 1] of teacher argmax cross entropy |
| `alr_tail_bucket` | true | Match omitted target probability mass as one tail bucket |

The checkpoint persists resolved `alr_ira_*` objective keys. Existing strategies
ignore the three ALR training fields. Full optimizer/RNG resume must retain the
same strategy and objective; optional DFlash-to-ALR initialization transfers only
compatible draft weights. This implementation does not claim full paper training
reproduction, quality improvement or throughput improvement.

## Osprey conversion and training

```json
{
  "method": "osprey",
  "model": {
    "source_checkpoint": "local-source-backbone",
    "num_layers": 2,
    "target_hidden_size": 1024,
    "target_layer_ids": [1, 3, 5],
    "target_tokenizer": "local-target"
  },
  "output_dir": "osprey-draft"
}
```

The converter reads local HF checkpoints (single or indexed safetensors/PyTorch
weights), retains the requested source prefix, and expands Q/K/V input to accept
projected target features beside token embeddings. At initialization, the copied
token half and zero target half preserve the retained source backbone's function.
The deepest target projection starts with a partial identity so the zero target
Q/K/V half can receive gradients immediately. Preservation is relative to the
retained prefix; reducing layer count does not preserve the full source model.

Supported sources are bias-free, dense Qwen3 with default RoPE and Llama with
default or Llama3 RoPE. Active sliding attention is rejected before conversion;
only full-attention source layers are supported. Exactly three unique ascending
target taps are required.
Optional tokenizer alignment maps exact tokens first, otherwise averages the
source-token embedding/head rows for the target token's decoded text. Unmapped
rows remain zero. It rewrites vocabulary buffers and target special token IDs in
model and generation config. Alignment changes source logits by design.

Use `model.draft_model_config` and `model.draft_checkpoint_path` pointing to the
converted directory, `model.load_target_embedding=false`, and
`training.strategy=osprey`. Target capture reuses the EAGLE3 feature layout and
TTT objective, with Osprey strategy labels. New vocab mappings are rejected;
alignment is a conversion operation. The frozen converted embedding is retained
in every draft checkpoint and HF reload. The architecture names are
`OspreyQwen3DraftModel` and `OspreyLlamaDraftModel`.

The shared target-head loader follows the configured training dtype, supports
indexed or a single unsharded local checkpoint, and checks the exact head shape
before copying. Direct conversion, registered training and HF reload checks use
tiny local models; they do not establish pretrained-model quality.

The CPU EAGLE3 objective uses the dense differentiable reference. CUDA retains
the existing fused Triton loss. SDPA conversion, training and HF reload have CPU
coverage. FlashAttention, flex attention, distributed training, full-state resume
and a production Osprey serving backend require separate execution evidence.

## CAST reference decoding

A CAST recipe supplies `model.target_model_path`, `model.draft_checkpoint_path`,
`data.input_path`, `output_dir`, optional `generation` and `speculation` sections.
Input rows supply `input_ids` or a text `prompt`. `generation.max_new_tokens` is
an exact output cap; `stop_token_ids` stops on the first emitted stop token.
`temperature=0` uses greedy verification. Positive temperature uses speculative
accept/reject with the target residual correction. Default proposal temperature
is 1.0. `speculation.budget` bounds nonroot tree nodes; `branch_k` is the maximum
per-depth proposal width and is clamped to vocabulary size.

The reference uses an unmodified DFlash/DFlash2 draft, ancestor-only target
verification masks and accepted-path KV gathering. It supports dense full
attention Llama/Qwen2/Qwen3 text targets. Model/draft dimensions, vocabulary and
RoPE must match. This batch-one reference is separate from production serving.
Its reported timings are diagnostic and do not establish a speedup.

## Redline recipe

```json
{
  "method": "redline",
  "data": {"input_path": "paired-calibration.json"},
  "calibration": {"alpha": 0.1, "delta": 0.1, "reference": 0},
  "output_dir": "redline-result"
}
```

The input JSON contains binary `correct[configuration][prompt]`, positive `cost`
and optional unique `names`. All configurations use the same prompts. By default
higher cost values are better (for example throughput); set `lower_is_better`
for latency. The runner lazily calls the benchmark-owned kernel and persists
selected/reference names, risk, p-values, validity, costs and sample count.
Dataset independence, representative calibration and an unchanged configuration
family are prerequisites the kernel cannot establish. No live model or speed
measurement is performed by this recipe.

## Verification and remaining evidence

The focused suites use tiny random local models and offline input. Commands from
the integration checkout, using an existing CPU environment:

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 SPECFORGE_DEVICE=cpu \
python -m unittest tests.test_research_methods tests.test_research_core \
  tests.test_research_providers tests.test_research_device \
  tests.test_algorithms.test_builtin_providers tests.test_config.test_schema -v
```

Acceptance checks cover actual rollout forward counts/IRA feature reuse, sparse
KD versus dense CE, causal predecessor alignment, future-context isolation,
registered cache-to-optimizer updates, Osprey Qwen3/Llama logit preservation,
TTT target-half gradients, tokenizer remapping/special IDs, complete checkpoint
HF reload, CAST greedy token parity/caps/EOS and corrected sampling frequency,
Redline recipe persistence, strict validation and unchanged unrelated resume
contracts. Use the pinned environment in [ENVIRONMENT](../ENVIRONMENT.md). CPU
checks do not substitute for the locked Linux accelerator environment.

`tests.test_research_device` is a three-test handoff for the parent-owned GPU
check. It defaults to CPU and selects CUDA only with
`DRAFTFIT_RESEARCH_TEST_DEVICE=cuda`; optionally set
`DRAFTFIT_RESEARCH_TEST_DTYPE=bfloat16`. The CUDA run prints peak allocated bytes.
It creates only tiny random models and downloads nothing. CPU passes do not
establish CUDA, fused loss, serving, distributed resume, pretrained-model quality
or production speed correctness. No GPU actions or rentals were performed by
this integration worker. Dated execution receipts and exact hook snippets live
in the worker's local planning records.
