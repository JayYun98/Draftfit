# SGLang milestone verification

Enable the optional `serving` block with `selection="acceptance"` in the
[unattended controller configuration](../../../../docs/UNATTENDED_NATIVE.md#verification-and-publication).
The [producer](../../../../draftfit/training/sglang_producer.py) runs the pinned live engine for a bounded milestone.
This handoff does not support existing-draft mappings or the native Prism/Spark adapters.

[Controller tests](../../../../tests/test_unattended_serving.py) and [gate tests](../../../../tests/test_unattended_serving_gate.py)
check lifecycle, failure and report handling. See the [support matrix](../../../../docs/PUBLIC_SUPPORT.md#joint-and-unattended-execution)
for the single-device dense-target boundary and known long-output parity failures.
