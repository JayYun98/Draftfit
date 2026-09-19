# CPU preparation and publication checklist

## Isolated CPU environment

The distribution and primary command are `draftfit`.
Legacy `speculative_train_platform`, `dspark` and `specforge` imports and their
command aliases remain available. Do not install
the upstream SpecForge distribution alongside it in the same environment: both
would own the legacy Python namespace. The implementation itself is now in
`draftfit`; use a dedicated environment when migrating from
upstream SpecForge or the previous `speculative-train-platform` or
`dspark-train-platform` distributions.

Training resume and export use the restricted tensor/primitive checkpoint loader,
including per-rank optimizer and RNG payloads. Arbitrary pickle objects are
rejected without an unsafe fallback. Prefer safetensors draft exports for sharing
weights; restricted loading is not a resource-isolation sandbox. Legacy checkpoints
containing custom Python objects must be converted in a trusted environment.

Run these contributor commands from the source checkout: the CPU lock and test
suite are not runtime wheel assets. Exported wheel recipes include the public
guides and original LICENSE, but do not provision a CPU or GPU environment.

This profile is for onboarding/testing, not inference or GPU training. It leaves
the existing Ling environment unchanged. Python 3.12 and uv 0.9.18:

```sh
uv venv --seed .venv-cpu
uv pip sync requirements-cpu.lock --python .venv-cpu/bin/python \
  --torch-backend cpu --require-hashes
uv run --no-project --python .venv-cpu/bin/python python -m tests.public_cpu
```

The lock includes transitive versions and hashes; Torch uses CPU wheels on
Linux, not CUDA dependencies. The public CPU CI uses the same lock and test
entry point. Regeneration is explicit, not automatic during tests:

```sh
uv pip compile requirements-cpu.in --python-version 3.12 --universal \
  --torch-backend cpu --default-index https://pypi.org/simple \
  --generate-hashes -o requirements-cpu.lock
```

**2026-09-08 installation status: Linux CPU PASS, macOS blocked.** A fresh
Linux Python environment installed this lock with hash enforcement and passed
209 tests (202 passed, seven expected accelerator skips) on the rented host.
This was CPU execution, not a CUDA installation. The local fresh
install rejected the Torch 2.11.0 macOS wheel because its downloaded SHA-256
did not match the package index hashes. Refreshing against the explicit public
index and a subsequent no-cache download produced the same mismatch. Hash checking was not bypassed and the
downloaded hash was not added as trusted. The existing local environment passes
the CPU gate, but uses Torch 2.13.0 / Transformers 5.14.1. The declared profile
pins 2.11.0 / 5.8.1 and is verified on Linux only.
Linux CI is configured but has not been run remotely in this task.

After a successful dependency gate, build/install the local wheel without
replacing the environment's backend:

```sh
uv build --python .venv-cpu/bin/python --no-build-isolation
uv pip install --python .venv-cpu/bin/python --no-deps dist/draftfit-0.1.0-py3-none-any.whl
uv run --no-project --python .venv-cpu/bin/python python -m draftfit.assets list
uv run --no-project --python .venv-cpu/bin/python python -I scripts/check_public_install.py
```

Use the actual wheel version if changed. The general package dependencies and
GPU image pins are separate profiles; never install this CPU lock over a GPU
training environment. `--no-deps` is safe only after provisioning dependencies
for the intended workflow; it is not a complete training installation.

The isolated installed-wheel check exercises inspect, DSpark/DFlash customization,
launch planning, overwrite refusal, invalid options and warm-start/resume conflict
rejection. It creates only temporary metadata; it neither trains nor certifies
real weights. The CPU suite separately checks tiny tensor/checkpoint contracts.
Packaging tests also rebuild a wheel from the sdist and compare package bytes
with a direct clean wheel, so a working checkout alone cannot satisfy the gate.

## Publication controls

The native teacher release evidence is bounded: real HF BF16 and native vLLM
FP32 feature/training gates passed, with deterministic settings required for
exact FP32 next-step replay. vLLM BF16 cross-engine parity failed and remains
unverified despite a successful short managed BF16 training run. Publish these
limits with the candidate; do not describe all teacher backends as production
certified. The [model validation matrix](MODEL_VALIDATION.md#native-teacher-two-gpu-functional-evidence-2026-09-14)
records the exact runtime and scope. Remote CPU CI, package publication and
serving validation remain separate gates.

- CPU pull requests use hosted CPU runners and read-only repository permissions.
  The inherited privileged GPU workflow is restricted to the upstream repository,
  so this fork's PRs cannot trigger its self-hosted cleanup/GPU jobs.
- Publishing remains manual, main-branch-only and depends on the CPU workflow.
  It requires `SPECFORGE_ENABLE_PUBLISH=true`, an approved `SPECFORGE_PYPI_PROJECT`
  matching project metadata, and a matching manual distribution-name confirmation.
  Forks retaining upstream distribution names are rejected. Do not enable it until identity and release
  checks below are approved. No publication was performed.
- Package discovery includes `draftfit*` and the small `speculative_train_platform`,
  `dspark` and `specforge` compatibility
  package; recipes are copied from the
  existing allowlisted source directories. Wheel tests check license inclusion,
  overwrite refusal and obvious local-path/private-key leaks. This bounded
  check is not a full secret/history audit.

## Decisions still required from the maintainer

1. Review the `draftfit` distribution and command identity
   before publication. The internal `specforge` namespace and legacy command remain
   for checkpoint/import compatibility, and upstream authorship remains credited.
   Publication is still opt-in; changing metadata does not publish a package.
2. Complete per-file attribution review before claiming license clearance.
   See [source inventory](SOURCE_ATTRIBUTION.md). Preserve existing notices.
3. Decide the exact supported model/backend set from [support evidence](PUBLIC_SUPPORT.md),
   not from the number of available example configs.
4. Audit the eventual public repository/history separately from the wheel;
   never publish the parent platform directory containing runs, checkpoints,
   local model data or experiment manifests.

The user subsequently authorized GPU validation; consult PUBLIC_SUPPORT.md for
the bounded results and explicit failures. Neither that authorization nor a
passing smoke test authorizes publication under the upstream package identity.
