# Redline: select a configuration from paired measurements

[Selection](method.py) compares each configuration with a fixed reference on the
same calibration prompts. Risk is the fraction of all prompts where the
reference was correct and the candidate was wrong. The selector applies exact
binomial tests and a Holm correction across the supplied configurations.

Use the [Redline recipe example](../../../../docs/specs/research-methods.md#redline-recipe)
with `python -m draftfit.research -c recipe.json`. The input JSON needs a binary
`correct[configuration][prompt]` matrix and one positive `cost` per configuration.
Optional `names` identify the rows. Higher values are preferred by default
(e.g. throughput); set `calibration.lower_is_better=true` for latency.

The recipe writes `result.json` to a fresh output directory. Read `deployed`
together with `valid`: if no candidate qualifies, the selector falls back to
the reference without certifying it. The input should come from a fixed family
of configurations and independent, representative prompts. The selector uses
measurements you supply and does not run a model.

The [source catalog](../../catalog.py) pins js-lee-AI/Redline. See the
[source notice](../../NOTICE.md#alr-ira-and-redline) for licensing.

The [regression tests](../../tests/test_research_core.py) cover calibration decisions and recipe output. The [MIT license](LICENSE) is included with this package.
