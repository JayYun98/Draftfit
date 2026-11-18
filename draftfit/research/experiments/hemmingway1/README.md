# Hemmingway-1 text drafts

Select the pinned Altworld/Hemmingway-1 target from the [recent target recipes](../../../../docs/RECENT_MODEL_EXPERIMENTS.md).
Pass its [DSpark](../../../../configs/hemmingway-1-dspark.json) or [DFlash2](../../../../configs/hemmingway-1-dflash2.json) config to
`draftfit target prepare` with the matching strategy and `--teacher-backend transformers`.

The [package architecture checks](../../../../tests/test_runtime/test_package_architecture.py) cover the DSpark recipe's metadata.
These are new draft configurations. The [target roadmap](../../../../plans/README.md#recent-targets-dspark-and-dflash2)
identifies hybrid capture and cache/state correctness as open checks.
A prepared config establishes neither real-weight training nor exact serving.
