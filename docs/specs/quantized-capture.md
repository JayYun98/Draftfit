# Additive quantized teacher capture

## Problem and scope

Local Prism Bonsai packs and their PQ2_0 GGUF companion require the target's
quantized basis and recurrent capture behavior. Reading rotated weights as an
ordinary dense head changes the teacher. Add explicit local adapters while
preserving the existing HF, SGLang, Transformers and vLLM paths.

This work ports only the capture/readout files from the read-only dirty
`recent-target-dev` snapshot at `e4b3f59597b2ff1c01d096d5f601a9708f9711e6`, onto
integration base `53370e2a91a5888ba44eea0caa2fd5edb0711ce2`, with additional
input, fingerprint and precision checks. Source digests and worker receipts are
in `.planning/training-feature-integration-20261028/quantized-*` (local records,
not release artifacts). Queue, ACK, control, policy, schema and CLI ownership
remain with the integration owners; exact call sites are in `quantized-hooks.md`.

Included paths:

- Local MLX Qwen3.5 text/hybrid capture and reviewed Prism Hadamard packs.
- Revision/tokenizer/content-bound pretokenized offline preparation.
- Bonsai GGUF v3 metadata, PQ2_0 readout and explicit native C ABI capture adapter.
- Frozen Torch embedding/readout, embedding reconstruction and tiny CPU checks.

Arbitrary GGUF quantizations, Orca/Saluki IQ2 variants, remote model Python,
vision/MTP capture, MLX online services, generic llama.cpp compatibility,
speculative state rollback, production serving and performance claims are outside
this adapter's supported boundary.

## Contract and implementation

### MLX offline producer

`draftfit.offline_capture.mlx.OfflineMLXCapture` accepts local `qwen3_5`,
`qwen3_5_text`, or `prism_hadamard_qwen35` directories through `from_pretrained`.
MLX/MLX-LM are optional external dependencies; Prism packs additionally require
MLX-VLM. No repository Python loader or `auto_map` is executed. Unsupported
architectures, custom loaders, invalid geometry and mismatched cache/layer counts
fail explicitly. This implementation has not been executed against MLX in the
current CPU environment; version numbers alone do not establish compatibility.

Taps are unique, ascending post-block/pre-final-norm indices. Final normalized
hidden states are separate. Binary masks must match integer `[batch, sequence]`
IDs within the vocabulary. Padding cannot be supervised and each row must select
a nonempty contiguous span. Each row runs with fresh recurrent/KV state, then
zero padding is restored in the original positions. Captures and temporary layer
proxies share a lock. Outputs cross into Torch as FP32 with shape/finite checks.

Prism schema 1/2 metadata must declare positive geometry, unique typed module
records, untied readout, supported Hadamard blocks and explicit signs. Schema 2
requires the reviewed `mlx-vlm-qwen3_5` namespace, grouped GDN activation layout,
2-bit/128-group affine quantization and the declared component set. Text tensor
loading is strict; only the known vision namespace is omitted. The adapted
packed/FWHT implementation retains the Apple MIT notice in
`draftfit/offline_capture/PRISM_RUNTIME_LICENSE`.

### Preparation and content identity

`python -m draftfit.offline_capture.prepare_mlx` is a separate offline producer,
not an additional `target_backend` for the managed online service. Input JSONL
starts with a provenance object:

```json
{"kind":"provenance","target_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tokenizer_sha256":"<64-hex digest of tokenizer.json>"}
{"id":"train:reviewed-0001","input_ids":[1,2,3,4],"loss_mask":[0,0,1,1]}
```

IDs above illustrate the format, not actual target tokens. Use reviewed target
IDs and explicit assistant masks. Revisions must be immutable 40-character lower
case commit identifiers; local Hub snapshot directory names must match. For an
ordinary local directory the revision is caller-supplied provenance, not proof
that its weights came from that Hub commit.

The existing capture manifest records SHA256 for config, tokenizer, draft config,
dataset and every local safetensors shard. Its manifest hash changes when any
weight changes under the same revision. Inputs are rechecked before and after
capture. Hashing reads complete weight files, including in dry-run; it does not
instantiate weights. Keep the inputs immutable during the run. This is not an
atomic filesystem snapshot or remote authenticity verification.

Duplicate IDs, out-of-range/bool tokens, bad taps, mismatched target/draft geometry,
missing weights, empty data, unsupported masks and input drift fail. The producer
never truncates a sequence or guesses a loss mask. It stages all bounded samples,
then publishes a new directory containing `NNNNNN.ckpt`, `capture_manifest.json`
and `capture_complete.json`. Existing destinations are rejected. Capture failure
or detected input drift publishes no partial feature directory. Captured fields
are `input_ids`, `loss_mask`, `hidden_states`, `target_last_hidden_states`; they
use the existing DSpark superset contract. Training-side binding remains the
existing reader/integration owner's responsibility.

### GGUF and Torch readout

`draftfit.modeling.target.quantized_parts.load_target_parts` recognizes only an
explicit existing local `.gguf` file or the reviewed local Prism pack marker.
Other targets delegate to `TargetEmbeddingsAndHead.from_pretrained` with existing
arguments. Canonical embedding/head keys are required for quantized paths.
`embedding_only=True` and `export_embedding` avoid materializing the dense head.

`gguf_parts` has a bounded little-endian GGUF v3 parser. It rejects duplicate or
unknown mandatory Prism metadata, invalid spans/alignment/storage, missing signs,
splits, ties and unsupported geometry. The GGUF profile is specifically Bonsai
Qwen35: hidden 5120, vocabulary 248320, depth 64, Hadamard block 1024, explicit
sign widths `[5120, 6144, 17408]`, grouped GDN and PQ2_0 embedding/output matrices.
A validated readout descriptor does not prove that every backbone tensor is a
loadable native target. `load_target_config` reads this geometry directly and
does not feed the GGUF file into HF JSON loading.

PQ2_0 uses an FP16 scale plus 32 bytes per 128 values: low-to-high adjacent 2-bit
codes decode as `(q - 1) * scale`. The MLX pack uses its separate 2-bit affine
layout. Embeddings apply inverse signed Hadamard. Heads transform activations
before the dequantized rotated readout. FP16 rounding and stored buffers survive
trainer dtype/device casts; autocast cannot silently change the head's FP16
projection. Gradients flow to draft inputs while the readout stays frozen.
Non-finite scales and FP16 dequantization overflow fail. This rounding contract
is not a claim of native F32, Metal or CUDA parity.

Full Bonsai readout storage is about 338 MB packed embedding plus 2.54 GB FP16
head, before activations and other model memory. Head decoding uses 512-row
chunks. Standalone parsing/readout does not verify file SHA; its caller owns
immutable target identity. Embedding export reconstructs only an embedding and
does not certify a serving backend or turn a packed teacher into an HF model.

## Native bridge prerequisites

`benchmarks/native/prism_capture_bridge.cpp` expects Prism's extended llama API,
with reference revision `e3de715222b2667c54a267441159a5729f11155e`. A stock
libllama is insufficient. Required headers include `include/llama.h`,
`src/llama-ext.h`, and `ggml/include/ggml-backend.h`. The linked runtime must expose
capture taps, per-layer inputs, normalized nextn output, and full hybrid memory
clear (`llama_set_capture_layers`, `llama_get_embeddings_capture_ith`,
`llama_set_embeddings_layer_inp`, `llama_set_embeddings_nextn`,
`llama_memory_clear`), plus the referenced model/context APIs and ggml backend
loader. Build libllama/ggml and this shared bridge from a separately reviewed
source tree with matching compiler/ABI and loader paths. None is bundled or
automatically installed by this Python adapter.

Record the actual source revision, local patch digest, build flags, headers,
bridge/library SHA256 and dependency paths. The bridge's revision function returns
a compiled constant; it cannot authenticate the linked libllama. The Python
wrapper checks that marker, exact target SHA256 and geometry, records the bridge
SHA256, and optionally enforces `expected_bridge_sha256`. It rechecks file contents
around delayed native initialization. These checks do not replace build provenance.

`OfflinePrismGGUFCapture.from_pretrained(path, bridge_path=..., expected_sha256=...,
device="cpu", torch_dtype=torch.float32, context_size=1024, chunk_size=32)` is the
explicit API. Call `set_capture_layers` before `capture`, and `close` at teardown.
Context must be 64..8192, chunk 1..512; reserve rows retain the source's floor of
64. Each unpadded row clears attention and recurrent state. Native outputs must
be finite before and after output dtype conversion. `return_logits=True` attaches
native FP32 logits at `batch.native_logits`; use these for native verification,
rather than substituting the approximate Torch head.

CPU mode disables model GPU layers and context offload, but runtime backend
discovery may still probe installed plugins. To guarantee a CPU-only native run,
provision a runtime without GPU backends/plugins. Accelerator selections in the
adapter are unverified. This is resident full-prefix reference capture, not a
KV-cached production speculative serving engine. Native logits can consume large
host memory; bound sample lengths and batch sizes accordingly.

## Usage

Use the provisioned Python 3.12 environment described in [ENVIRONMENT](../ENVIRONMENT.md).
From the repository root, the following dry-run needs no MLX import but does need
local config, tokenizer and safetensors files:

```sh
uv run --no-project --python /path/to/python3.12 python -m draftfit.offline_capture.prepare_mlx \
  --target /path/to/immutable/local-target --target-revision EXACT_40_HEX_COMMIT \
  --draft-config /path/to/draft.json --data /path/to/reviewed-tokenized-train.jsonl \
  --output /path/to/new/train-features --max-length 512 --num-samples 100 --dry-run
```

Expected dry-run output includes the manifest, bounded sample count, estimated
feature bytes and `weights_loaded: false`; no feature directory is created.
Remove `--dry-run` only in the provisioned MLX runtime. Use separate reviewed
holdout data/output. The real MLX command remains unrun in this integration.
The parent integrates provider, export, schema and joint teacher call sites;
the generic offline loader and managed teacher backend list are unchanged here.

## Acceptance and evidence

Criteria were set before hardening. Original packed-readout regressions reproduce
negative-index acceptance, FP16 overflow, autocast drift and invalid metadata
acceptance; the fixes retain passing independent matrix/codec references.

| Requirement | Check | Current result / limit |
| --- | --- | --- |
| Exact tiny PQ2/affine decode, forward/inverse basis, frozen storage and input gradients | `tests.test_gguf_parts`, `tests.test_quantized_target_parts`, `tests.test_quantized_capture_boundaries` | Pass on actual tiny CPU files/tensors; GGUF fixtures reduce only vocab under an explicit test override |
| Real optimizer update through frozen packed head and checkpoint reload | `PackedReadoutBoundaries.test_real_cpu_optimizer_step_through_frozen_packed_head` | Pass on CPU; this is not a real native teacher or pretrained draft quality run |
| Fingerprints, geometry, no silent fallback, no overwrite | `tests.test_gguf_target_geometry`, actual MLX preparation dry-run CLI and boundary tests | Pass on CPU; staging failure/drift checks inject a producer and prove persistence behavior only |
| C ABI validation and wrapper thread/error/identity behavior | `tests.test_prism_capture_api` | Pass with compiled fake llama API / injected wrapper library; no native numerical/runtime compatibility claim |
| Native bridge source syntax with real available extended headers | C++17 `-fsyntax-only`, header digests in worker receipt | Pass; no native link/load/model run, and this local header snapshot has no separately verified runtime checkout revision |
| Existing HF loading/capture and manifest behavior | target loader, Transformers capture and manifest regressions | Pass on CPU; tiny real Transformers captures remain separate from Prism/MLX evidence |
| Real MLX model loading/capture | `tests.test_mlx_capture` | Not-run: MLX/MLX-LM/MLX-VLM absent; 5 explicit skips |
| Real pinned Prism link/load, target taps/final norm/reset/chunk parity and native readout parity | Separate target/runtime gate | Not-run: no reviewed built runtime/model run in this task |
| Actual accelerator capture/training/resume/export/serving and rollback/speed A/B | Parent's separately authorized gates | Not-run: no GPU activity or paid action by this worker |
| Wheel license/import integration | Parent package-data hook and installed-wheel check | Pending integration owner; source notice retained, no wheel success claimed here |

Portable focused command (no MLX runtime tests):

```sh
SPECFORGE_DEVICE=cpu CUDA_VISIBLE_DEVICES='' \
uv run --no-project --python /path/to/python3.12 python -m unittest \
  tests.test_gguf_parts tests.test_gguf_target_geometry \
  tests.test_quantized_target_parts tests.test_quantized_capture_boundaries \
  tests.test_prism_capture_api -v
```

Runtime parity, full target training and serving must be executed against the
exact target/build artifacts before promoting this adapter beyond its bounded
experimental scope. No quality improvement, native numerical parity or speedup
is inferred from CPU contracts or mocks.

Final focused receipt: 68 tests run, 63 passed, 5 explicit MLX skips; no failures.
Pinned pre-commit hooks were not run because pre-commit and its formatters are
absent from this provisioned CPU environment. Source files passed whitespace
and syntax checks. No GPU or paid action was performed.
