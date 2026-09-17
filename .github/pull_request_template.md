## Problem and scope

<!-- Link the spec or issue. Explain the user-visible change and exclusions. -->

## Implementation and ownership

<!-- Affected boundaries: config, capture/data, algorithm, training, checkpoint,
export, serving. Preserve source notices for adapted code. -->

## Usage

<!-- Reproducible commands, runtime profile and expected artifacts. -->

## Verification

| Gate | Result | Evidence / limitations |
| --- | --- | --- |
| Relevant regression tests | | |
| CPU / installed wheel | | |
| Training / resume (if affected) | | |
| Export / serving (if affected) | | |

<!-- Use pass/fail/not-run/not-applicable; explain the latter two.
Keep failed criteria unchanged. CPU checks are not real GPU certification. -->

## Compatibility and release limits

<!-- Config/checkpoint/import compatibility, migrations, unsupported combinations.
Separate loss decrease, acceptance, correctness and measured speedup claims. -->

## Checklist

- [ ] Relevant tests and pinned pre-commit checks passed.
- [ ] Usage, support scope and third-party notices are accurate.
- [ ] No secrets, private conversations, weights or operational artifacts included.
- [ ] Unrun or failed gates are explicit; no unmeasured production/speedup claims.
