# Prism GGUF joint capture

Select `model.target_backend="prism_gguf"` in a joint online family base config.
Set `target_model_path` to the local GGUF, `target_gguf_sha256` to its digest and
`prism_capture_bridge` to the compiled library. The [joint producer](../../../../draftfit/training/joint_online.py)
uses the [capture adapter](../../../../draftfit/offline_capture/prism_gguf.py); verification requests native logits.

The [build and tensor contract](../../../../docs/specs/quantized-capture.md) requires Prism's extended llama API and the reviewed Bonsai Qwen35 layout.
Stock llama.cpp and arbitrary GGUF files do not satisfy it.
The [ABI tests](../../../../tests/test_prism_capture_api.py) use a fake native API; the
[verification tests](../../../../tests/test_prism_joint_verification.py) cover acceptance, rejection and prefix replay.
Neither exercises the real pinned target.
