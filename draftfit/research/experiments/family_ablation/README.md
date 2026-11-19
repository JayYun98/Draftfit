# DFlash-family transfer and ablations

Use the [family planner](../../../../draftfit/family.py) via `draftfit family --manifest family.json --plan`.
The [MiniCPM5 matrix](../../../../docs/RECENT_MODEL_EXPERIMENTS.md#one-command-minicpm5-matrix) includes
DSpark head-only, EAL and WTV settings, DFlash CE/DPACE, and a DFlash2 retarget.
Select `training.retrain_anchor_alignment=true` to initialize an anchor-retarget arm from compatible DSpark weights.

[Family tests](../../../../tests/test_family.py) cover held-out features, plan overrides and sequential handoff.
The [family guide](../../../../docs/DFLASH_FAMILY.md) distinguishes weights-only initialization from same-run optimizer resume.
A successful copy or lower proxy loss does not imply zero-step equivalence or faster serving.
