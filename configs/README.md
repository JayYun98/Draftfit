# Model recipes

This directory contains model/draft architecture settings and target metadata.
Full training-run configurations live in [examples/configs](../examples/configs/README.md).
The Python schema and validation implementation lives in `draftfit/config/`;
it is code, not a second collection of recipes.

Both recipe directories are authoritative source files bundled into wheels by
the build. Do not manually copy them into `draftfit/assets/`. Relative recipe
paths are preserved to avoid breaking existing training configurations.

Use [target preparation](../docs/TARGET_EXTENSION.md) to create a project for your
model. A recipe is not proof of real-weight capture, training or serving support;
see [validation scope](../docs/MODEL_VALIDATION.md).
