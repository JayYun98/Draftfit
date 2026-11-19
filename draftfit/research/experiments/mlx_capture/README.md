# MLX offline feature capture

Run `python -m draftfit.offline_capture.prepare_mlx` with a local target, immutable revision,
draft config and tokenized JSONL with reviewed loss masks. Start with `--dry-run`:
the [producer](../../../../draftfit/offline_capture/prepare_mlx.py) hashes local inputs before loading weights.
The [capture adapter](../../../../draftfit/offline_capture/mlx.py) accepts Qwen3.5 text or the reviewed Prism Hadamard pack.

Follow the [input schema and command](../../../../docs/specs/quantized-capture.md#usage).
[Portable checks](../../../../tests/test_quantized_capture_boundaries.py) cover identity and persistence;
[Metal tests](../../../../tests/test_mlx_capture.py) require MLX and MLX-LM, and the Bonsai path needs MLX-VLM.
MLX supplies offline features, not a managed teacher or a draft trainer.
