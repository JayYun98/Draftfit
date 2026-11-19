# Agnes-3.0-Flash text capture

Select the pinned Agnes target with `draftfit target prepare --teacher-backend transformers`
and the [DSpark](../../../../configs/agnes3-flash-dspark.json) or [DFlash2](../../../../configs/agnes3-flash-dflash2.json) draft config.
The [Transformers adapter](../../../../draftfit/offline_capture/transformers.py) unwraps the language model;
[decoder-selection tests](../../../../tests/test_offline_capture/test_transformers.py) cover this boundary with a fixture.

Use the [target revision and preparation limits](../../../../docs/RECENT_MODEL_EXPERIMENTS.md).
These are new text drafts, not conversions of multimodal or MTP heads.
A fixture does not validate original Agnes weights, hybrid state replay or a vLLM/SGLang route.
