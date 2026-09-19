# Changelog

User-visible changes are recorded here. Package publication is a separate,
explicitly approved operation; a Git version tag does not imply a PyPI release.

## [0.1.0] — 2026-09-19

Initial Draftfit source release. **Your workload. Your draft.**

### Included

- Conversation preparation with deterministic train/holdout separation,
  target inspection and editable training configurations.
- Integrated DSpark, DFlash, DFlash2, EAGLE3, PEagle and Domino training paths,
  subject to their documented feature and backend constraints.
- Offline features and live teacher capture through a shared training pipeline;
  bounded SGLang support and experimental native Transformers/vLLM teachers.
- Weights-only fine-tuning, same-world-size checkpoint resume, restricted
  tensor/primitive checkpoint loading, and method-specific export.
- Held-out workload benchmarking, provenance checks and validation commands.
- Canonical `draftfit` package and CLI, with compatibility aliases for previous
  platform names. A fresh environment is required when migrating distributions.
- uv environment profiles, hashed CPU dependencies, wheel/sdist checks and
  installed-package onboarding checks.

### Validation and limits

See [model validation](docs/MODEL_VALIDATION.md) for exact historical GPU revisions
and [release gates](docs/RELEASE_GATES.md) for supported boundaries. vLLM BF16
feature parity and the documented RTX5090 native replay path remain failed.
Renaming and versioning do not constitute a new GPU validation campaign.

No universal architecture certification, guaranteed speedup, cross-world optimizer
resharding, automatic trace ingestion or continual deployment is claimed.
Blog and LinkedIn materials remain unpublished drafts.

### Provenance

This release includes a SpecForge-derived engine and attributed adaptations.
Original notices remain intact; see [third-party notices](THIRDPARTY_NOTICES.md).
The main branch starts with one consolidated release commit. Pre-release history
is preserved on backup branches as described in [repository history](docs/REPOSITORY_HISTORY.md).

### Version policy

`version.txt` is the package version source of truth. Tag releases as `vMAJOR.MINOR.PATCH`
only after the documented checks pass for the stated scope. During `0.x`, document
breaking interface changes in a minor release and compatible fixes in a patch release.
