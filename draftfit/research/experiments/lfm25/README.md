# LFM2.5 dense and MoE warm starts

The [2.6B dense](../../../../configs/lfm2.5-2.6b-dspark.json) and [8B-A1B MoE](../../../../configs/lfm2.5-8b-a1b-dspark.json)
configs have different tap lists. Keep conv-block taps in both. For a warm start use the
official draft directory with `--strategy dspark --teacher-backend transformers --draft-checkpoint`
as described in the [pinned target guide](../../../../docs/RECENT_MODEL_NEXT_TARGETS.md).
Do not substitute a candidate JSON for the official tensors.

The [capture tests](../../../../tests/test_offline_capture/test_transformers.py) exercise tiny dense and MoE models.
The guide records 62/62 official draft tensors loaded for each model; this does not establish
source-runtime numerical parity, original-target capture, or served token equality.
