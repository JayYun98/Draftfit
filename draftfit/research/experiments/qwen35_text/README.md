# Qwen3.5 hybrid text capture

Choose the Transformers teacher (`--teacher-backend transformers` during project preparation
or `--target-backend transformers` during offline capture) and a compatible text draft.
The [capture adapter](../../../../draftfit/offline_capture/transformers.py) selects the language model for
`qwen3_5` and accepts `qwen3_5_text`; taps are post-block and the final target state is normalized separately.

The [recent target record](../../../../docs/RECENT_MODEL_EXPERIMENTS.md) describes tiny Qwen3.5 capture through
DSpark and DFlash2 updates. [Capture tests](../../../../tests/test_offline_capture/test_transformers.py)
exercise the text decoder. Vision inputs, hybrid state rollback and equivalent features in vLLM are outside this path.
