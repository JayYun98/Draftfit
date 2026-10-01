# Source-checkout verification tools

User workflows ship in the installed package:

| Command | Purpose |
| --- | --- |
| `draftfit data prepare` | Normalize local conversations |
| `draftfit data presets` | Prepare public dataset mixtures |
| `draftfit data regenerate` | Regenerate target responses |
| `draftfit data expand-reasoning` | Expand reasoning events |
| `draftfit data validate` | Validate conversations |
| `draftfit capture` | Capture offline teacher features |

Repository verification lives in `packaging/check_public_install.py`,
`integration/validate_ling_replay.py`, and [GPU gates](integration/README.md).
The patch installer is [patches/sglang/apply.sh](../patches/sglang/apply.sh).
Performance tools live in [benchmarks](../benchmarks/README.md).
See [data preparation](../docs/basic_usage/data_preparation.md) and
[environment setup](../docs/ENVIRONMENT.md) for commands and requirements.
GPU gates are opt-in checks, not a general production certification.
