# Experimental paper intake

## Problem and scope

Researchers need a repeatable way to turn a growing literature directory into
reviewable Draftfit experiments. Existing research recipes cover selected methods,
but they do not track new notes, revisions, or implementation decisions.

Add a local intake ledger and a TableTune offline table recipe. Keep experimental
methods behind explicit research commands. Existing training recipes and runtime
defaults retain their contracts.

## Contract

- Track paper IDs, note revisions, and content fingerprints.
- Treat a scan as discovery. Require an explicit decision to close a candidate.
- Reopen a candidate when its selected source changes. Preserve earlier decisions.
- Select the highest paper version and retain older notes as historical input.
- Reject stale decisions whose fingerprint differs from the current candidate.
- Keep the caller's literature directory read-only. Keep local state out of Git.
- Require source, license, overlap, and correctness review before implementation.
- Bound each scheduled implementation batch to one candidate and one draft PR.

The first experiment builds draft-length tables from cost and acceptance files.
It adapts TableTune's dependency-light core at
`38170878d901cb06f696a122a0b4140a3d365f02` under Apache-2.0.
The integration does not install its vLLM plugin or start a serving process.

## Acceptance and evidence

| Requirement | Acceptance | Status |
| --- | --- | --- |
| Repeatable intake | Unchanged rescans preserve pending work and completed decisions | pass |
| Revised sources | New versions and changed content reopen work without losing history | pass |
| Missing sources | Removing every paper marks the tracked sources unavailable | pass |
| Stale decision guard | A decision cannot close a different source revision | pass |
| Private input boundary | Public source and packaged assets contain no personal paths or note bodies | pass |
| Experimental recipe | Plan writes nothing; CPU execution persists the selected table | pass |
| Numerical contract | Low acceptance can select zero draft tokens; bad profiles fail explicitly | pass |
| Packaging | Installed artifacts include the command, recipe, fixture, and retained license | pass, scoped CPU environment |
| Full locked CPU gate | Install the current hash lock and complete `tests.public_cpu` | blocked |
| GPU and serving | Real engine parity and throughput | not-run |

## References

- [TableTune paper](https://arxiv.org/abs/2610.38968)
- [Pinned implementation](https://github.com/asm582/tabletune/tree/38170878d901cb06f696a122a0b4140a3d365f02)
- [Research methods](../research/README.md)

## Usage and execution evidence

Follow [paper intake](../research/paper-intake.md) for scan, record, and recipe commands.
The [execution receipt](../research/results/paper-intake-20261111.json) records the checked source identities and commands.

The final research suite passed 54 tests. Existing research regressions passed
30 tests, and packaging passed three tests. Installed-wheel onboarding also passed.
These checks used Python 3.12.8, Torch 2.11.0, and Transformers 5.8.1 on macOS CPU.
They establish the scoped intake and offline-table behavior.

The repository requires Transformers 5.12.1. The reused environment does not
establish its full CPU gate. That run was stopped after an environment mismatch.
A fresh hash-locked installation failed twice while downloading Torch 2.11.0,
including a retry with an empty cache. The computed SHA256 was
`43b35116802c85fb88d99f4a396b8bd4472bfca1dd82e69499e5a4f9b8b4e252`.
The lock did not contain that hash. No hash check was bypassed.

Resume the full gate after the lock or artifact discrepancy is resolved:

```sh
uv pip sync --python .venv/bin/python requirements-cpu.lock \
  --torch-backend cpu --require-hashes
uv run --no-project --python .venv/bin/python python -m tests.public_cpu
```

Live controller, CUDA graphs, model parity, training quality, and serving speed
remain unverified. This experimental PR is not release certification.

## Small-model GPU follow-up — 2026-11-13

The [component experiment](../research/results/paper-components-small100-20261113.md)
tested the unchanged TableTune module from `e990dae50f3206809965cfdaf85353bbb30d16e0`.
It built offline tables from 36 measured CUDA forward profiles across three
trained small students. Every raw schedule matched the independent argmax.
Most choices disabled drafting. This is a diagnostic offline check; live engine
parity and serving throughput remain not-run. The linked receipt also records
separate D3D, REAL and DForkTree source snapshots and their bounded results.
