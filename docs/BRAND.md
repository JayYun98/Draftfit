# Draftfit

**Your workload. Your draft.**

Draftfit helps people train and fine-tune speculative decoding drafts for their
target model, data and workload, then evaluate the result against target-only inference.

## Naming

- Product: **Draftfit**. Package and primary command: `draftfit`.
- Category: workload-specific speculative draft training.
- DSpark, DFlash and DFlash2 are draft methods, not alternative product names.
- The existing repository URL stays valid; a repository rename is a separate action.
- Legacy import/command names remain compatibility aliases, not parallel engines.

## Writing

Lead with the user's workload and the prepare → train → export → evaluate flow.
Show concrete supported workflows before discussing internals. Prefer “train and
measure” to unqualified claims that every workload will become faster.

Describe automatic agent tracing and continual deployment as future work. Do not
claim universal model support, production certification or measured speedups from
loss reduction alone. Keep validation details in the support/evidence guides.

Credit the SpecForge-derived engine and adapted sources in provenance and notices.
Product identity does not imply all code was written from scratch. Blog/social
files are drafts until explicitly approved for publication. Domain, registry-name
availability and trademark clearance have not been established by this rename.
