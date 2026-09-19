# Draftfit documentation

**Your workload. Your draft.**

The maintained entry points are [environment setup](ENVIRONMENT.md),
[public workflow](PUBLIC_WORKFLOW.md) and [support boundaries](PUBLIC_SUPPORT.md).
Preserve upstream credits when editing inherited material; see
[source attribution](SOURCE_ATTRIBUTION.md).

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
