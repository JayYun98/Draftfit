# Opt-in runtime validation tools

These tools verify real backend behavior; they are not alternate training CLIs.
Use the pinned backend, explicit artifact paths and fresh output directories.

| Tool | Purpose |
| --- | --- |
| `run_owned_teacher_gpu.py` | Native Transformers/vLLM feature parity and draft train/resume/export gate; see the [runtime runbook](../../docs/OWNED_GPU_ENVIRONMENT.md) |
| `check_ling_online_features.py` | Compare actual Ling online/offline tensors, separately from manifest-only checks |
| `run_dflash_chat_serving_gate.py` | Compare an overfit sample against a running compatible SGLang chat server; not a workload speed benchmark |
| `fixtures/` | Small synthetic conversation/config inputs for the native teacher gate, not personal data |

Run each Python entry point with `--help` in its provisioned environment.
Prefer `draftfit export --to sglang` for supported normal exports. Existing
configs need not be modified by the normalizer unless their loader requires it.
Serving, feature parity and throughput are separate checks; see the
[support boundaries](../../docs/PUBLIC_SUPPORT.md).

Legacy export conversion belongs to the installed export package:
`python -m draftfit.export.normalize_legacy --config COPY/config.json --block-size N`.
This modifies the supplied copy in place for the specific SGLang loader and
supports DFlash/Domino/DSpark, not DFlash2. It does not certify serving.

Historical one-sample campaign instructions and unused shell orchestration
helpers are not maintained public workflows. These tools never authorize a
GPU rental or certify an untested model/backend combination.
