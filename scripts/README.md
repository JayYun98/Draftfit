# Maintained source-checkout tools

Start with `draftfit data prepare`, `draftfit target prepare`, `draftfit train`,
`draftfit export` and `draftfit benchmark`. Scripts below provide capabilities
not replaced by that user-facing workflow. They run from a source checkout in
the corresponding [uv environment](../docs/ENVIRONMENT.md), not as wheel assets.

| Tool | Maintained use |
| --- | --- |
| `prepare_data.py` | Download/normalize supported dataset presets; unlike the CLI's local conversation conversion |
| `expand_reasoning_conversations.py` | Expand reasoning conversations before target-response regeneration |
| `regenerate_train_data.py` | Generate target responses through an operator-provided server |
| `validate_regenerated_data.py` | Validate regenerated conversation records |
| `conversation_validation.py` | Shared validation helper used by the data scripts; not a CLI |
| `prepare_hidden_states.py` | Offline teacher feature capture for the trainer |
| `apply_sglang_spec_capture_patch.sh` | Apply a compatible pinned SGLang capture patch; also used by runtime preflight and Docker builds |
| `check_public_install.py` | CI's isolated installed-wheel onboarding check |
| `validate_ling_replay.py` | Explicit opt-in Ling token/state replay verification in its pinned runtime |
| [`gates/`](gates/README.md) | Real-backend feature, train/resume and serving checks |

See [data preparation](../docs/basic_usage/data_preparation.md),
[offline teachers](../docs/ENVIRONMENT.md#offline-teachers) and
[native Ling replay](../docs/NATIVE_LING_REPLAY.md) for commands and limits.
Do not run GPU gates just to inspect the package or treat a gate script as a
production certification. One-off campaign wrappers belong outside public source.
