# Repository lineage and release history

Draftfit v0.1.0 consolidates the previous main branch and the reviewed integration
branch into one root commit: `Initial release: Draftfit v0.1.0`.
The integration tip `84e2853` already contained main `de43b88`; no divergent main
changes were discarded. Version and release-documentation edits follow that tree.

## Recoverable pre-release history

Before rewriting main, the following branches were saved locally and remotely:

- `backup/pre-0.1.0-main`: `de43b883f129ba8b9de3be08bdabd43e3311e74e`.
- `backup/pre-0.1.0-integration`: `84e2853e5241a632f33caff0039589d8741c0152`.

Earlier commits remain accessible through those branches. Squashing main is not
a purge of repository history or sensitive data. Historical validation records
continue to identify the revision actually tested, not the new root commit.

For an existing clone, prefer a fresh checkout. Do not run an ordinary pull that
merges the old history back into main. Preserve local work before aligning any
existing branch with the rewritten remote.

## Source lineage

The engine is derived from SpecForge and retains its applicable copyright and
license notices. The original source extraction was based on `ab2f720` plus
platform changes. Earlier source milestones include `f9f8c3b`, `b90afde`,
`8c4db37`, `e04145f` and `5cd36da`; these may belong to the original source
workspace rather than this repository's reachable history.

See [third-party notices](../THIRDPARTY_NOTICES.md) and
[source attribution](SOURCE_ATTRIBUTION.md). A new root commit does not imply
that upstream code was written from scratch by the current maintainer.

## Publication boundary

The package and canonical import are `draftfit`. Earlier import/command names
remain compatibility aliases. Source tagging does not publish to PyPI, make a
private repository public, or certify all runtime combinations. Existing release
approval controls and documented unsupported/failed gates remain in force.
