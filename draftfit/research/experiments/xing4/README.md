# Xing4.0-29B-A4B hyperconnection capture

Use `draftfit target prepare` with the pinned Xing target, `--teacher-backend transformers`
and the matching [DSpark](../../../../configs/xing4-29b-a4b-dspark.json) or [DFlash2](../../../../configs/xing4-29b-a4b-dflash2.json)
draft config. The [capture adapter](../../../../draftfit/offline_capture/transformers.py) requires a four-dimensional
block output and averages the hyperconnection streams before exposing the tap.

See [Xing prerequisites](../../../../docs/RECENT_MODEL_EXPERIMENTS.md#xing-prerequisites).
[Capture tests](../../../../tests/test_offline_capture/test_transformers.py) check the reduction;
[rotary tests](../../../../tests/test_dflash2_integration.py) cover YaRN. Real-weight capture and serving
must reproduce the same tap definition. There is no Xing vLLM teacher lane.
