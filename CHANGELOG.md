# Change history

This file groups related work by user-visible outcome. It is not an exhaustive
commit log and intentionally does not invent release dates, release numbers, or
one-to-one commit mappings.

## Unreleased — private DSpark Train Platform baseline

### Product boundary

- Reframed the fork as a configuration-first platform for a user-selected,
  frozen target model and a separately trained speculative draft.
- Kept target execution in the existing backend boundary; no target-architecture
  port or competing target runtime was added.
- Retained the `specforge` Python namespace and distribution name for local
  compatibility. Package publication remains disabled while the private
  repository identity is reviewed.

### Target-aware customization

- Added weight-free target inspection with separate observed facts,
  architecture-aware recommendations, and explicit unknowns.
- Added target revision handling and provenance checks so metadata, tokenizer,
  weights, capture, and serving do not silently mix revisions.
- Added editable target preparation for DSpark, DFlash, EAGLE3, PEagle, and
  Domino, including algorithm-specific validation and explicit unsupported
  combinations.
- Preserved existing draft dimensions unless the user overrides them and made
  Domino require an explicit draft configuration.

### Data, parity, and artifacts

- Hardened conversation JSONL preflight, assistant-response checks, renderer
  requirements, and overwrite refusal before generated files are written.
- Added capture provenance manifests and offline/online feature-contract checks.
- Added token, state, replay, and serving-acceptance validation commands.
- Added reproducible artifact metadata for target, draft, data, and validation
  inputs.

### Training and recovery

- Kept one typed `specforge train` entry point for offline and online/
  disaggregated workflows.
- Added or retained launch planning for local and producer/consumer layouts,
  including FSDP2 and same-world checkpoint/reload coverage.
- Documented the distinction between weights-only warm start and full training
  resume; incompatible weights and warm-start/resume conflicts fail explicitly.
- Kept cross-world-size optimizer resharding and cold-host infrastructure-loss
  recovery outside the supported boundary.

### Packaging and release checks

- Added dependency-free recipe listing/export for installed wheels and
  editable checkouts.
- Added the hashed CPU lock workflow, installed-wheel CLI checks, and
  source-distribution-to-wheel payload checks.
- Kept CPU locks/tests separate from runtime wheel assets and did not make wheel
  installation replace an existing GPU backend.
- Restricted privileged GPU workflow behavior to the upstream repository; the
  private repository’s CI is configured but manually disabled.

### Bounded runtime evidence

- Documented the pinned H100 Ling native ReplaySSM functional route, including
  exact committed-state and target-only token comparisons under its tested
  single-request/static constraints.
- Recorded the SM120/RTX5090 native ReplaySSM state-parity failure and kept that
  path disabled rather than broadening the support claim.
- Kept functional correctness, acceptance, quality, and speedup as separate
  outcomes; no all-model or production-quality claim is made.

### Upstream lineage and attribution

- Retained upstream implementation notices and copyright headers.
- Recorded borrowed design boundaries and pinned source references in
  [THIRDPARTY_NOTICES.md](THIRDPARTY_NOTICES.md).

### Explicitly not part of this history

- No package publication or public distribution under the inherited upstream
  identity.
- No blanket support claim for every example configuration or model family.
- No DFlash2, DFly, DFlare, MTP, multimodal training, cache-enabled capture,
  arbitrary backend pairing, or unsupported serving/state path.
