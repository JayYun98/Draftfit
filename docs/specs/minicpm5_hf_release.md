# MiniCPM5 Draftfit model release

## Problem / scope

Publish the experimentally trained MiniCPM5-SFT draft family as a visible Draftfit validation set, with the measured speed table at the top of each Hugging Face model card. Four repositories cover DSpark, DFlash CE, DFlash DPACE and DFlash2. Their `main` revisions carry the v4 artifacts measured on the same 53-prompt domain holdout; `v5` revisions carry later continual checkpoints. DSpark also has a `v3-best` predecessor revision. Neither the target-only run nor OpenBMB's official draft is our model.

The card opens with what Draftfit does: common-cache family/ablation training, conversion from an existing DSpark checkpoint with anchor retraining, and weights-only continual training. The family outputs are separate weights rather than one universal serving weight.

## Contract / implementation

- Publish `config.json`, `model.safetensors`, the model card, a hash/provenance manifest, source terms, upstream license text and the prompt-free evaluation summary. Do not publish source conversations, feature caches, original runtime checkpoints, logs, credentials or target weights.
- Pin target `openbmb/MiniCPM5-2B-SFT@3e0690c57f6ac772180498467c82e56170cc00ac` and initial draft `openbmb/MiniCPM5-2B-DSpark@114a20fdbf53220712c7fbdd7dccddbf1dedebb4`.
- Separate A100 run-wide output throughput from MLX request-level median speed. Label v3/v5 numbers by their own holdout; do not imply they measured the v4 domain table.
- Label DFlash2 A100 serving as not measured; state the observed v4 token-parity limits. Do not claim production correctness or speed superiority over the official draft.
- The v4/v5 training data include APIGen-MT-5k, marked CC BY-NC 4.0, whose card also describes a GPT-4 provenance restriction. The user requested public visibility with these source terms disclosed. Per-row generator provenance and downstream rights remain unresolved; staging, publication, and CPU reload do not clear that issue.

## Acceptance and evidence

| Requirement | Pass criterion | Status | Evidence |
| --- | --- | --- | --- |
| Artifact integrity | All nine selected checkpoints export and reload; v4 DFlash2 matches the prior GPU export hash | pass | Local staged export manifests; DFlash2 SHA-256 `a328a9b13a2a9b176687f289eeffb6f98c20bad92039d343417d01afe9438316` |
| Card presentation | First paragraph promotes Draftfit, followed by the same v4 speed and four-domain table on every card | pass | Nine locally parsed staged cards |
| Artifact selection guidance | All nine public cards explain DSpark, CE, DPACE and DFlash2 design roles, when to try each and measured limits without claiming unobserved quality gains | pass | `hf-release-2026-10-04/card_update_receipt.json`; anonymous README hash audit on 2026-10-04 |
| Private-data boundary | Staged repo contents contain only weights/config/card/hash manifest; no prompt text or local paths | pass | Local file inventory and card/config preflight |
| Hugging Face artifact staging | Four model repositories exist with nine selected revisions; remote weight LFS SHA-256/size and downloaded small-file hashes match local exports | pass | `hf-release-2026-10-04/upload_receipt.json`; authenticated nine-revision audit on 2026-10-04 |
| Hugging Face public visibility | All four repositories and nine revisions are accessible without authentication; downloaded small-file hashes and remote weight hashes match local exports | pass | `hf-release-2026-10-04/public_audit.json`; anonymous nine-revision audit on 2026-10-04 |
| APIGen downstream rights | Resolve source restrictions and selected-row generator provenance for uses beyond the disclosed research/noncommercial scope | unresolved | Dataset card and `SOURCE_TERMS.md`; no commercial-use claim |
| Serving certification | Each uploaded revision reloads and passes target token/state parity and matched A100 A/B | not-run | Prior bounded measurements have explicit parity failures or unsupported DFlash2 runtime |

## Usage

Download `jayyun98/MiniCPM5-2B-SFT-Draftfit-DSpark`, `...-DFlash-CE`, `...-DFlash-DPACE`, or `...-DFlash2`; select `--revision v5` for the later stage or DSpark `--revision v3-best` for its earlier highest A100 observation. The draft requires the pinned SFT target and its tokenizer separately. Use a backend supporting that exact draft architecture; the public Draftfit repository documents the workflow. The experiment-specific code/configs are still being prepared for source release.

## Results / limitations

Local staging passed export and CPU reload for nine variants. Four repositories and all nine revisions were uploaded and made public at the user's request. Anonymous access confirmed every weight SHA-256 and size, and downloaded small files matched local SHA-256. A100 v4 measured DSpark/CE/DPACE, MLX v4 measured all four. v5 DFlash2 has no serving speed; v5 DPACE A100 retains console values but not raw run JSON. MLX minority domains have six requests each and separate model sessions. APIGen downstream rights and production serving remain unverified.

All nine public README revisions now include a use-case matrix: DSpark as the same-method adaptation/control path, DFlash CE as the token-loss baseline, DPACE as the prefix-weighted objective experiment, and DFlash2 as the grouped-convolution/candidate-selector experiment. The text separates design intent from measured throughput and warns that v5 is not automatically a better default. Only README files were uploaded in this follow-up; anonymous downloads matched local hashes, and remote weight SHA-256 values stayed unchanged.
