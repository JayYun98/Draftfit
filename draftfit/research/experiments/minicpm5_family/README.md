# MiniCPM5-2B draft family

Prepare the pinned target with the [DSpark recipe](../../../../configs/minicpm5-2b-dspark.json) and
official draft snapshot, or the [DFlash2 scratch recipe](../../../../configs/minicpm5-2b-dflash2.json).
For the five-layer retarget experiment, follow the [family manifest and capture instructions](../../../../docs/RECENT_MODEL_EXPERIMENTS.md).
The three-layer scratch recipe uses different taps and needs its own feature cache.

The [warm-start tests](../../../../tests/test_runtime/test_official_dspark_warmstart.py) check tensor transfer;
[family tests](../../../../tests/test_family.py) check manifest planning and handoff.
The [v4 report](../../../../docs/reports/minicpm5-family-2026-10-03.html) records the actual holdout and
hardware. It includes token mismatches and no compatible DFlash2 serving path on the measured A100.
