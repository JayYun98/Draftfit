# Native vLLM teacher

Use `--teacher-backend vllm` with `draftfit target prepare` for the managed service, or
`--target-backend vllm` with `python -m draftfit.offline_capture.prepare` for saved features.
The [adapter](../../../../draftfit/offline_capture/vllm.py) uses native hidden-state extraction and stages
connector files locally. The trainer consumes the same feature contract in PyTorch.

The [environment guide](../../../../docs/ENVIRONMENT.md#offline-teachers) requires a separate vLLM runtime, one process and TP1/PP1.
The [Qwen3-0.6B GPU record](../../../../docs/MODEL_VALIDATION.md#native-teacher-two-gpu-functional-evidence-2026-09-14)
reports FP32 parity and a failed BF16 parity check. Neither it nor the [mock connector tests](../../../../tests/test_offline_capture/test_vllm.py)
validate prequantized checkpoints accepted by `auto`; see [quantized targets](../../../../docs/QUANTIZED_TARGETS.md).
