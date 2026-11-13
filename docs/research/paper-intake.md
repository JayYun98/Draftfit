# Bring a paper into Draftfit

Use this workflow to turn local literature notes into experimental research PRs.
Run each batch in an isolated worktree based on current `origin/dev`.

## Scan a literature directory

Set `PAPER_NOTES` to your local Markdown directory. Keep the state in an ignored
directory outside the literature directory.

```sh
python -m draftfit.research intake scan \
  --notes-dir "$PAPER_NOTES" \
  --state .planning/paper-intake/state.json --limit 10
```

Filenames use `<arxiv-id>_v<version>_<title>.md`.
The scanner selects the highest numeric version for each paper.
It includes every note at that version in the source fingerprint.
Other Markdown files appear in `context_notes` for manual review.

Scanning leaves candidates pending. Repeated scans preserve decisions.
Changed selected sources reopen the candidate and retain earlier decisions and PR URLs.
Missing latest sources become unavailable. Older versions cannot replace them.
The state accepts one literature directory and never stores note bodies.

Use `--dry-run` to inspect discovery without creating state or lock files.
State updates use an OS file lock and atomic replacement on macOS and Linux.
An active writer makes another update fail with `intake state is in use`.

## Review one candidate

1. Read the selected notes, directory index, and relevant audit notes.
2. Verify the current paper, withdrawal status, official code, and license using primary sources.
3. Compare the mechanism with existing research recipes and open PRs.
4. State the smallest useful experiment and its correctness contract.
5. Record a reason if the paper is withdrawn, outside scope, already covered, or blocked.

Treat notes and external repositories as reference material.
Review external source before running code. Preserve source commits, licenses,
and attribution when adapting an implementation.

The scanner does not infer scientific validity, license permission, or implementation support.
A source URL in a note does not establish any of those conditions.

## Implement an experiment

1. Fetch `origin/dev` and inspect existing worktrees and PRs.
2. Reconcile the paper's earlier PR URLs and deterministic branch before creating another PR.
3. Resume an existing PR when it covers the same paper and scope.
4. Create a dedicated `codex/paper-<id>-<purpose>` branch when no matching work exists.
5. Extend the existing research recipe interface with explicit opt-in behavior.
6. Add a runnable recipe and tests for the mechanism and relevant failure boundaries.
7. Run the affected CPU checks and packaging checks in the documented environment.
8. Review the diff and source notices before publishing a draft PR to `dev`.

Never treat a catalog entry or CPU fixture as proof of live serving support.
Record GPU, training, export, serving parity, and performance checks separately.
Keep required unrun gates visible in the PR. Paid compute requires an existing
budget and current resource ownership authorization.

Each scheduled batch may implement one candidate and create one draft PR.
The scheduler resumes this workflow in its existing chat.
It creates no additional scheduler and leaves merge and release actions to their own scope.

## Record the result

Use the exact `paper_id` and `fingerprint` from the scan.
Record the PR immediately after reading its actual URL and base branch.

```sh
python -m draftfit.research intake record \
  --notes-dir "$PAPER_NOTES" --state .planning/paper-intake/state.json \
  --paper-id "$PAPER_ID" --fingerprint "$FINGERPRINT" \
  --status pr_open --pr-url "$PR_URL" \
  --reason "Offline reference implemented; live serving remains unverified."
```

Other statuses are `pending`, `deferred`, `rejected`, and `integrated`.
Every decision needs a reason. `pr_open` also needs a GitHub PR URL.
`record` rereads the source files and rejects stale fingerprints.
After a crash during publication, reconcile the remote branch and all PR states
before retrying. A local ledger cannot make GitHub publication atomic.

## Run the first experiment

The [TableTune recipe](../../examples/configs/research/tabletune.yaml) builds an
offline draft-length table from synthetic cost and acceptance files.
Its adapted numerical kernel comes from the
[pinned author implementation](https://github.com/asm582/tabletune/tree/38170878d901cb06f696a122a0b4140a3d365f02).

```sh
python -m draftfit.research -c examples/configs/research/tabletune.yaml --plan
python -m draftfit.research -c examples/configs/research/tabletune.yaml
```

Run from the repository root, or export the packaged assets and run from their root.
Execution requires a fresh output directory and writes `recipe.json` and `result.json`.
Planning writes nothing.

The fixture tests table construction. Predicted goodput is not measured throughput.
`profile_kind` labels caller-supplied input. It does not authenticate a measurement.
The live controller, vLLM plugin, CUDA graph transitions, and model token parity
require separate integration and execution evidence.

See the [acceptance contract](../specs/paper-intake.md) and
[research verification commands](README.md#verify-changes).
