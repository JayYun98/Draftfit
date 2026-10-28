# Latest speculative decoding methods

## Problem and scope

Apply the twelve research notes selected by the user, starting from development
base `1a296d35d1eb8253cd67fb2ae0cd5d9d8d55a0b9` and integrated onto
`3885e6264fe2db6deeaaa4fca57ea79139ca1ea0` before publication.
Reuse the platform's strict research
recipes, registered draft models, training contracts and verification paths.
This work must produce executable integration and observed behavior, with source
availability and runtime limits recorded for every method.

The initial portable integration targets existing-weight inference: NgramRL
rollout memory, SpecMem live user memory and ValuSpec candidate valuation.
Primary-source reviews determine the appropriate RIFT, TreeWY, query-layout,
replay and MoE-offload paths. Diagnostic or unrelated learning papers do not
automatically become new draft architectures.

## Contract

- All new behavior is opt-in through typed configuration. Existing recipes,
  trainer checkpoints, joint policies and exports retain their defaults.
- Candidate memory changes proposals only. The target verifies committed tokens;
  sampling must use the actual proposal distribution and residual correction.
- Cache reads are snapshots of previously verified output. Writes occur after
  complete verification, with explicit model/tokenizer and request scope.
- Memory is bounded. Prompt history, user stores and prefetch work have declared
  limits. No reference runner executes tools.
- A model-free proposal reference may use full-prefix forwards. Its diagnostic
  timing cannot establish CUDA overlap or production-engine speedup.
- New outputs use the existing fresh-directory and atomic-publication contract.

## Acceptance and evidence

| Requirement | Observable acceptance | Status |
| --- | --- | --- |
| Native memory proposals | Real tiny target generation consumes memory and matches target greedy tokens, including misses, rejection, EOS and token caps | passed: CPU fixtures |
| Sampling correctness | Proposal support and residual correction match a known categorical target distribution; rejected proposals cannot leak into output | passed: 6,000 categorical trials and exact greedy checks |
| Stateful isolation | Same-prompt history ages out, users remain isolated, verified write-back changes later proposals, bounded eviction survives state reload | passed: CPU fixtures; capped SpecMem output is not stored by default |
| ValuSpec integration | Candidate scoring and calibrated filtering feed the actual CAST verifier; retained trees have valid ancestry and greedy output remains exact | passed: tiny real CAST, calibration/evaluation split guard and empty-tree fallback |
| RIFT reference | Physically gathered per-head draft KVs are smaller; absolute RoPE positions remain logical; target output is exact | passed: tiny AR model, budgets 2 and full history |
| ARR reference | Replay matches greedy; fixed-point roots retire and replacements run; incomplete revisions preserve one final answer per root | passed: actual tiny target and scheduler boundaries |
| TreeWY reference | All outputs and every accepted state match independent recurrence with non-square state and branching siblings | passed: CPU; 2e-6 absolute tolerance |
| Config and CLI | Strict recipe planning/execution, atomic output, packaged examples/docs/license | passed: eight CLI fixtures; wheel/sdist and asset export |
| Existing paths | Affected ALR/IRA, Osprey and CAST regressions remain valid | passed within the CPU environment below |
| Pretrained GPU references | Pinned <=1B target; exact greedy output, real memory continuation, physical cache and numerical execution | bounded Qwen3-0.6B gate passed; see GPU follow-up below; ARR quality failed on the toy arithmetic question |
| Production model/engine gates | Exact deployment runtime/model artifacts and workload A/B | not-run |

## Coverage of the twelve selected notes

These are bounded implementations, not twelve paper reproductions. No training
registry entry or serving backend is added merely because a paper uses tokens,
trees or speculative decoding. Source metadata is available with `--sources`.

| Note / paper | Applied entry point | Implemented boundary and remaining work |
| --- | --- | --- |
| [NgramRL, 2610.32881v1](https://arxiv.org/abs/2610.32881v1) | `method: ngramrl` | Per-prompt bounded rollout memory, longest n-gram lookup, actual proposal sampling and exact correction. Synchronous reference only; causal fan-out, CPU/GPU overlap and verl/vLLM rollout-worker integration remain unimplemented. |
| [SpecMem, 2610.31840v1](https://arxiv.org/abs/2610.31840v1) | `method: specmem` | User-local cosine retrieval/LRU, explicit population seeds, schema fallback and verified-output write-back. Supplied embeddings; no MiniLM reproduction, tool execution, tool schema validation or side-effect speculation. |
| [ValuSpec, 2610.31338v1](https://arxiv.org/abs/2610.31338v1) | `method: cast`, `valuation` | Frozen AR conditional leaf-branch scores, target-greedy calibration and ancestry-safe filtering before target packing. Leaf union and linear quantile are explicit local conventions; greedy only. |
| [RIFT, 2610.31098v1](https://arxiv.org/abs/2610.31098v1) | `method: rift` | Real eager attention, GQA grouping, important/recent selection and physical AR draft KV gather. Recomputes committed history every round. No amortized/adaptive refresh, cache offload, compression-aware KD or DFlash cache mutation. |
| [ARR, 2610.28294v1](https://arxiv.org/abs/2610.28294v1) | `method: arr` | Stationary revision, one-way greedy replay, EOS-inclusive fixed points, root retirement/replacement and final voting under a hard processed-position budget. Serial reference; not paper batching or cached-FLOP accounting. |
| [TreeWY, 2608.20961v2](https://arxiv.org/abs/2608.20961v2) | `method: treewy` | Pinned Apache-2.0 author single-tree solver plus owned validation/commit adapter. Numerical GDN experiment only; convolution/attention-KV commit, Triton and vLLM scheduling remain external. |
| [MoE-SpAc, 2603.09983v2](https://arxiv.org/abs/2603.09983v2) | `method: moe_spac` | Paper-v2 utility transitions and exhaustive constrained threshold search on profiled routing traces. No expert execution/transfer. The public preview predates and differs from v2. |
| [Phase-scoped query layouts, 2610.03228v4](https://arxiv.org/abs/2610.03228v4) | `method: query_layout` | PQ/QB floating-point attention, per-query masks, GQA and phase admission diagnostic. Official HeteroSpec repository returned 404 on 2026-10-28; GGML lowering/graph/state/token parity is not implemented. |
| [Acceptance position, 2610.30644v1](https://arxiv.org/abs/2610.30644v1) | `diagnostics.record_acceptance` | Actual attempted-proposal position bins, distribution overlap, argmax agreement and realized acceptance. Rejected suffixes are censored; this is not the full MATH-500/paired-position study or a position-decay conclusion. |
| [Apple Silicon evaluation, 2610.31479v1](https://arxiv.org/abs/2610.31479v1) | `diagnostics.compare_ar`, `warmup`, `repetitions` | Same-runtime, same-output greedy AR pairing and full-generation repeated timings including proposal work. No MLX/Metal port or Apple paper reproduction; CPU smoke ratios are diagnostic only. |
| [Multi-token graph search, 2610.31439v1](https://arxiv.org/abs/2610.31439v1) | No executable integration | Separate scratch-trained graph-search/tokenization experiment; no verified author release or compatible draft checkpoint. A faithful addition needs its own dataset, tokenizer, model and training objective. Existing DFlash/DSpark configs cannot select it. |
| [Speculative diffusion tradeoffs, 2610.31819v1](https://arxiv.org/abs/2610.31819v1) | No executable integration | Multi-step diffusion effort and explicitly lossy verification need a compatible diffusion engine, proposal-law accounting and quality evaluation. Existing exact CAST/reference verification has no leniency switch. No dedicated author release was verified in the original paper. |

## Config, architecture and stage combinations

All recipes use the existing `StrictConfigModel`, discriminated method schema,
dotted overrides, read-only planning and fresh-directory publication. Training
configs/checkpoints/exports have no new fields. There is no new default mode.

| Config choice | Model or data requirement | Stage |
| --- | --- | --- |
| `ngramrl` | Dense Llama/Qwen2/Qwen3 target; repeated token prompts with monotonic `step` | Verified generation -> bounded per-prompt rollout history |
| `specmem` | Same targets; `user_id`, query embedding, optional schema token draft | Retrieval -> exact verification -> completion-gated user write-back |
| `cast + valuation` | Standard DFlash/DFlash2 + dense target + token-compatible frozen AR scorer | Disjoint calibration -> draft tree -> filter/compact -> CAST target verification |
| `rift` | Independent dense AR draft and target with identical token-ID mappings | Full draft-history recompute -> attention selection -> compressed proposal KV -> exact target verification |
| `arr` | Dense target, explicit EOS IDs, stationary question/solution prompt | Independent roots -> greedy replay refinement -> retirement/replacement -> one vote/root |
| `treewy` | GDN Q/K/V, log decay, gates, parent IDs and initial state | Numerical solve -> accepted GDN state reconstruction |
| `moe_spac` | Layer/window router counts, resident IDs, free bytes and measured cost/ratio profiles | Utility update -> all thresholds -> feasible prefetch decision or CPU fallback |
| `query_layout` | Q/K/V and query-specific additive masks | Phase admission -> PQ/QB reference comparison |

This is an explicit set of supported combinations. RIFT's AR cache contract
does not apply to feature-conditioned DFlash noise blocks. TreeWY's GDN state
does not turn dense CAST KV gathering into a hybrid serving implementation.

## Reproducible usage

Run from the repository root in a provisioned environment; see
[ENVIRONMENT.md](../ENVIRONMENT.md). Model recipes need real local checkpoints
and reviewed inputs. Replace the example paths and EOS IDs with the selected
target's values. Remote targets/scorers/drafts should set their revision fields.

```sh
python -m draftfit.research --sources
python -m draftfit.research -c examples/configs/research/ngramrl.yaml --plan
python -m draftfit.research -c examples/configs/research/rift.yaml cache.context_budget=256
python -m draftfit.research -c examples/configs/research/valuspec-cast.yaml
```

The same handler is available as `draftfit research`. Numerical examples need
no model download and include small illustrative inputs:

```sh
python -m draftfit.research -c examples/configs/research/treewy.yaml
python -m draftfit.research -c examples/configs/research/moe-spac.yaml
python -m draftfit.research -c examples/configs/research/query-layout.yaml
```

For a complete executable CPU fixture with actual tiny Transformers models,
tokenizers and DFlash weights, run:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/research/run_latest_smoke.py \
  --output outputs/latest-methods-smoke
```

This command executes eight recipes through the actual CLI, checks target token
identity, and preserves recipes, per-method logs, result JSON, memory snapshots,
runtime versions and source digests. The output directory must be new.

### Memory inputs and state

NgramRL JSONL rows require `input_ids` and nonnegative monotonic training `step`.
An optional `prompt_id` scopes the token prompt; histories are never shared
across different token prompts. Multiple rows in one step are allowed. Tables
come only from earlier verified rows and expire by configured training steps.

```json
{"input_ids":[1,2,3],"prompt_id":"task-a","step":0}
```

SpecMem rows supply precomputed finite embeddings from one consistent embedding
space. User history never becomes another user's shared seed. Population seed
files are explicit JSON lists of `{query_embedding, token_ids}` entries.

```json
{"input_ids":[1,2,3],"user_id":"user-a","query_embedding":[1,0],"schema_draft_tokens":[4,5]}
```

Default `memory.writeback_mode=eos` stores only target-verified outputs ending
in configured `generation.stop_token_ids`. Capped, unfinished outputs are not
stored. Explicit `verified_sequence` is a general token-memory adaptation which
also permits capped verified sequences; it does not establish complete tool calls.
Neither mode executes tools or certifies argument schemas/side-effect safety.

Resume only the memory with `memory.state_path=previous-output/memory.json` and
a fresh `output_dir`. Method, target-config/tokenizer identity and memory policy
must match. Prompt/user/rollout/token/vector limits remain enforced after reload.
This is not optimizer/RNG or RL-worker resume. Local identity hashes config and
available tokenizer bytes, not mutable model weights; preserve real weight
artifacts separately for benchmark provenance.

### ValuSpec scoring and calibration

The frozen scorer evaluates actual conditional log probabilities from the full
committed prefix, including CAST's pending root, along every root-to-leaf path.
Passing leaves retain their complete ancestor union; parent indices/masks are
rebuilt. An empty retained tree performs a valid root-only target step.

Thresholds are recalibrated per execution on disjoint target-greedy traces.
Artifacts record target/scorer revisions, vocabulary digest, input hashes,
calibration scores/count, maximum depth, branch policy and linear quantile.
Identical vocabulary size alone is insufficient: tokenizer vocabularies and
special IDs must match. No CAST marginal-score shortcut or stochastic filter
is exposed. Full-generation timing includes the first scorer pass; the legacy
CAST `ms_per_token` clock retains its separately labeled historical scope.

### RIFT, ARR and numerical boundaries

RIFT preserves logical absolute positions while physically gathering active
per-head K/V slots. Each round rebuilds a recoverable full-history view from
committed tokens, then selects context and grows in-flight proposal slots.
Reported active bytes exclude that reconstruction peak and temporary attention
buffers. The first token comes from dense prefill logits. The budget covers
context; in-flight draft tokens are reported separately. No claim of lower peak
GPU memory, amortized refresh or serving speed follows from this reference.

ARR charges every full-prefix target forward before execution, including
discarded/recomputed positions. Initial-root admission reserves the worst-case
uncached call cost. Revisions use a fixed template containing only question and
current solution. A truncated revision preserves the prior complete answer;
identity includes EOS. `last_line` and `token_sequence` are explicit local vote
extractors, not a math-answer accuracy evaluator or the author's prompt.

TreeWY accepts a connected parent-before-child tree with one lifted root. Null
accepted node preserves the initial GDN state; negative indices are rejected.
The adapter accepts finite nonpositive log decays and gates in [0,1], including
numerical endpoints. It does not commit convolution or attention state.
MoE policy timing fields are modeled from supplied profiles, not inference
measurements; query-layout agreement cannot certify quantized endpoint parity.

## Execution receipt: 2026-10-28

- Base: merged `origin/dev` at `1a296d35d1eb8253cd67fb2ae0cd5d9d8d55a0b9`.
- Branch: `codex/latest-spec-methods-dev`; evidence covers its working changes.
- Local environment: Python 3.12.8, Torch 2.13.0, Transformers 5.14.1, CPU.
- The locked CPU sync **failed** on the Torch 2.11.0 macOS wheel hash; it was
  not bypassed. The behavioral environment above is outside the pinned profile,
  so this work does not certify Torch 2.11.0 / Transformers 5.12.1 compatibility.
- New tests plus research core and wheel/sdist/asset checks: **31 passed**.
  The other affected research method/provider/review checks: **21 passed** in
  the preceding regression run; their exercised behavior did not change later.
- Eight final real CLI fixtures passed in
  `outputs/latest-methods-smoke-final-20261028/summary.json`. Five exercise actual
  tiny model inference; three exercise numerical/trace inputs. These are
  functional fixtures, not pretrained-quality or production-speed evidence.
- Review corrected capped SpecMem write-back and missing packaged-document
  wiring; focused checks were rerun after both corrections.
- The initial CPU delivery did not run pretrained/GPU checks; the subsequent
  bounded GPU execution is recorded below. MLX/Metal, hybrid state, expert
  offload, external-engine serving and throughput/quality A/B remain not-run.
  Training/resume/export gates are
  not applicable to these inference/diagnostic additions; existing training
  algorithms were covered only by the bounded CPU regressions.

Reproduce the checked regression groups in that local environment with:

```sh
python -m unittest tests.research.test_latest_research tests.research.test_latest_kv_tree \
  tests.research.test_latest_control tests.test_research_core tests.test_packaging.test_assets -v
python -m unittest tests.test_research_methods tests.test_research_providers \
  tests.test_research_review_regressions -v
```

### Pretrained GPU follow-up on existing gare2e capacity

The user's <=1B request ran on an existing A100 SXM4 80GB,
using Qwen/Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`:
596,049,920 parameters, FP32, eager attention. The existing environment was
Python 3.12.3, Torch 2.13.0, Transformers 5.17.0 and Pydantic 2.14.1; no
packages, serving settings or instance lifecycle were changed by this gate.
This remains an alternate runtime, not certification of the locked profile.

| Gate | Observed result | Limit |
| --- | --- | --- |
| NgramRL and state continuation | All 14 output tokens equal target AR; target forwards 14 cold, 2 warm and 2 after reload | Repeated synthetic prompt; synchronous reference |
| SpecMem and state continuation | Alice/Bob/Alice sources schema/schema/user; verified EOS write-back; reload retrieves Alice's history; output equals AR | Supplied two-dimensional embedding fixture and oracle schema seed; no real tools |
| RIFT | Output equals AR; accepted 12 proposals in 3 target forwards; active context KV 3,670,016 bytes versus 6,651,904--8,945,664 before gather | Independent copy of the same 0.6B model; full history recomputed; no peak-memory saving claim |
| ValuSpec/CAST | Calibrated scorer and 13 real verifier rounds equal target AR | Untrained 17,830,144-parameter DFlash; all 7 candidates removed each round; 1 calibration sample; fallback correctness only |
| ARR | Two complete roots reach EOS-inclusive fixed points and vote; 240 processed target positions within 12,000 budget | Both vote `1` for `1+1`: toy-answer quality fails despite correct scheduler execution |
| TreeWY | CUDA output and committed state agree with the same-input CPU reference; max state error 5.96e-8 | GDN numerical solver only |
| Query layout | CUDA PQ/QB max absolute error 0 | Floating-point diagnostic only |
| MoE-SpAc | Policy trace executes on CPU | No GPU expert/offload execution |

The source-only driver is `scripts/research/run_latest_pretrained_gpu.py`.
It uses separate processes, fresh outputs, a 5.5GiB allocator limit and a
7GiB headroom check before every case. Initial free VRAM was 8,032MiB;
maximum observed allocator reservation was 4.6796875GiB. Ten functional
cases plus preparation exited successfully. The first preparation failure
(Transformers chat-template return type) is preserved; explicitly rendering
text before tokenization corrected that harness issue.

Raw configurations, outputs, logs, resource snapshots and source hashes are
under `outputs/latest-methods-gpu-gare2e-20261028/`; 88 artifact digests and
257 executed source files were verified after retrieval. During the run,
the pre-existing vLLM PID disappeared and a different vLLM process appeared.
The cause is unverified; this driver contains no service-control actions.
Shared GPU load therefore changed: raw timings are retained as diagnostics,
with no stable-load, production-speed or quality-preservation claim.

To repeat after independently confirming existing GPU headroom:

```sh
PYTHONPATH=. python scripts/research/run_latest_pretrained_gpu.py \
  --model-path /path/to/pinned/Qwen3-0.6B/snapshot \
  --output outputs/new-pretrained-gpu-run --max-gib 5.5 --min-free-gib 7
```

TreeWY hybrid serving, MoE expert offload and GGML layouts additionally
require independently built compatible engines. These numerical recipes
cannot execute those gates. Broader held-out accuracy and stable-workload
performance remain required independently of the bounded checks above.

### Latest development integration before publication

The branch was rebased onto `origin/dev` at
`3885e6264fe2db6deeaaa4fca57ea79139ca1ea0` (PR #4). The shared model loader and
packaging changes integrate without conflicts. On this integrated source, all
**52 affected CPU regressions** passed, including the real wheel/sdist and
installed-asset checks. All **eight CLI fixtures** also passed, with results in
`outputs/latest-methods-smoke-pr-dev-corrected-20261028/summary.json`.
The first CLI invocation omitted the documented `PYTHONPATH=.` and failed before
loading the package; rerunning with that setting passed all eight cases.

Repository-pinned pre-commit hooks passed for the changed files. The pinned
TreeWY author source is excluded only from source-mutating format hooks so its
original bytes and license remain intact; syntax and safety checks still apply.
CPU checks use the alternate runtime recorded above. The GPU receipt remains
tied to the earlier `1a296d3`-based execution, before integration and formatting;
no new GPU run or locked-profile certification is claimed for this revision.

### Non-oracle effect follow-up

The initial GPU receipt established behavior; its two timing repetitions and
changing shared load did not establish workload efficacy. A subsequent probe
uses the integrated `89fe422` source plus the source-only
`scripts/research/run_latest_effects.py`, whose executed SHA256 is
`4f4b960f542063e78f987c426ddc96fb4e5ef2b43c4a326fda47a46b719a6dcf`.
The model, FP32 eager runtime and existing A100 allocation are unchanged.

Before evaluation, a single real count request fills causal memory. The current
baseline's output is never supplied to a candidate, and there are no schema
seeds. SpecMem uses explicit 128-bin lexical word-count embeddings, not a learned
semantic encoder. Every measurement restores the same prior history, performs
one warmup, then runs three candidate/AR pairs with alternating order. Lookup,
generation and write-back are timed; model loading, tokenization, state restore
and the prior priming request are excluded. All **51 measured pairs** have exact
greedy token equality. All three workers exited 0.

| Probe | Observed effect | Limit |
| --- | --- | --- |
| NgramRL repeated request | 14→2 target forwards; median 452.5→82.3 ms | One real prior identical prompt; no forward reduction for cold/paraphrased/changed/new prompts |
| SpecMem repeated request | 14→1 forwards; median 452.3→51.2 ms | Same user's real EOS-complete history; no oracle seed |
| SpecMem paraphrase | 10→8 forwards; median 339.7→269.3 ms | Lexical retrieval accepts only 2 tokens; changed numbers, new text and another user have no forward reduction |
| RIFT context budget 16 vs full history | Active context 3,670,016 bytes vs 4,816,896–9,175,040 before gather; 16-budget timing is 1.23–1.83 times AR latency | Three prompts, independent copy of the same 0.6B draft; full history reconstructed; whole-worker peak about 4.82 GB in both cache modes |
| ARR four new arithmetic questions | AR and ARR both have the correct final number in all four; both fail the requested bare-integer exact response in all four | Strict full-response exact match was predefined; last-number extraction is a post-hoc diagnostic and does not replace that failed metric. No accuracy gain |

The ARR root temperature is 0.7, with at most three roots, two active roots and
6,000 processed target positions per question. The original wrong `1+1` result
remains recorded. Four toy questions are not a held-out math benchmark.

Maximum observed allocator reservation was 4.60546875 GiB, below the 5.5 GiB
limit. Sampled external process identities stayed constant during the paired
measurements; later observations found the external server absent and then a
replacement process while health was unavailable. The cause is unverified,
and the probe performs no service-control actions. Shared GPU activity is not
isolated, so these synthetic reference
latencies do not establish stable-load or production speedup.

The [public aggregate receipt](results/latest-methods-effects-20261028.json)
contains all individual timings, counters, exact responses, source/workload
hashes and scoring rules. Raw logs and configurations remain local under
`outputs/latest-methods-effects-20261028/`; all 11 result/log digests are retained.
ValuSpec still has no trained-draft efficacy evidence; TreeWY/query layout remain
numerical and MoE-SpAc remains policy-only. No new training or serving claim is
made by this follow-up.

To repeat on independently confirmed spare capacity with a fresh output path:

```sh
PYTHONPATH=. python scripts/research/run_latest_effects.py \
  --model-path /path/to/pinned/Qwen3-0.6B/snapshot \
  --output outputs/new-latest-methods-effects --repetitions 3 \
  --max-gib 5.5 --min-free-gib 7
```

## Primary references

- [NgramRL](https://arxiv.org/html/2610.32881v1), sections 3.2--3.4.
- [SpecMem](https://arxiv.org/html/2610.31840v1), sections 3.3--3.4.
- [ValuSpec](https://arxiv.org/html/2610.31338v1).
- [RIFT](https://arxiv.org/html/2610.31098v1).
- [TreeWY revision 2](https://arxiv.org/html/2608.20961v2).

Native implementations from published algorithms must be distinguished from
adopted official source. A baseline repository is not the newer paper's release.

## Evidence and file layout

The GPU runs predate this documentation and test reorganization.
The recorded source hashes identify the executed implementations.
Algorithm modules and experiment scripts retain their original paths and bytes.
The public receipt retains its original content.
Current test commands use the `tests.research` package.

See the [research index](README.md) for recipes, results, and execution limits.
