# Ling on the Ampere runtime profile

Compose the [SM80 overrides](../../../../configs/runtime_profiles/linux-sm80-cu129.yaml) with the
[Ling online recipe](../../../../examples/configs/ling-3.0-tiny-dspark-online.yaml) as shown in the
[runtime profile guide](../../../../docs/RUNTIME_PROFILES.md). This config selects Triton kernels and does not install or validate a server.

The [profile composition test](../../../../tests/test_runtime/test_runtime_profiles.py) checks resolved settings.
[Native replay evidence](../../../../docs/NATIVE_LING_REPLAY.md) is tied to other pinned hardware and builds.
The Ampere profile does not fix the recorded fused-top-k failure; no SM80 Ling training or replay pass is established.
