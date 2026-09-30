# Draftfit documentation

**Your workload. Your draft.**

The maintained entry points are [environment setup](ENVIRONMENT.md),
[public workflow](PUBLIC_WORKFLOW.md) and [support boundaries](PUBLIC_SUPPORT.md).
Preserve upstream credits when editing inherited material; see
[source attribution](SOURCE_ATTRIBUTION.md).

## Read the guides

| Task | Guide |
| --- | --- |
| Install and migrate | [uv environments](ENVIRONMENT.md) |
| Prepare, train, fine-tune, resume, export | [Workflow](PUBLIC_WORKFLOW.md) |
| Choose a method/model combination | [Support matrix](PUBLIC_SUPPORT.md) · [Validation evidence](MODEL_VALIDATION.md) |
| Add your target | [Configuration and adapters](TARGET_EXTENSION.md) |
| Select a backend environment | [Runtime profiles](RUNTIME_PROFILES.md) · [Model environments](MODEL_ENVIRONMENTS.md) |
| Measure inference performance | [Workload evaluation](PERFORMANCE_GATE.md) |
| Contribute and validate | [Release checks](PUBLIC_RELEASE.md) · [Source tools](../scripts/README.md) |

## Documentation languages

English is the canonical language for public documentation, examples and agent
guides. Private notes are exempt. The [Korean README](../README.ko.md) translates
the project overview; detailed guides currently remain in English.

To contribute another language, add `README.<language-code>.md` beside the English
README and add reciprocal language links only after the translation exists.
Keep commands, configuration keys, model IDs, support levels and validation
limitations consistent with the English source. Update affected translations
with the source, or clearly mark them as outdated; do not translate API names.

## Build and preview

Use a separate documentation environment so building the site does not replace
the CPU training lock or a custom GPU runtime. The current documentation
requirements are Sphinx, MyST, the book theme, copybutton and autobuild;
`requirements.txt` is not a hashed reproducibility lock.

```sh
uv venv --python 3.12 .venv-docs
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
uv run --no-project --python .venv-docs/bin/python python -m sphinx \
  -b html docs docs/_build/html
uv run --no-project --python .venv-docs/bin/python sphinx-autobuild \
  docs docs/_build/html --host 127.0.0.1 --port 8003
```

Documentation sources are Markdown and reStructuredText. Add maintained pages
to `index.rst` and use relative links between documentation pages. Links to
checkout-only scripts or configurations are not installed-package resources.

Inspect build warnings before publishing. Inherited reference pages may have
separate link or formatting debt; a successful HTML build alone does not
validate their runtime claims. Follow the repository's contributor environment
for formatting checks.
