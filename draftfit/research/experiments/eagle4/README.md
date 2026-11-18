# EAGLE4 training and MiniCPM5-1B comparison

Select `--strategy eagle4` with `draftfit target prepare` and a compatible official checkpoint,
or use the [continual-training example](../../../../examples/configs/eagle4-official-continual.yaml).
The [algorithm](../../../../draftfit/algorithms/eagle4/) trains the dense recurrent draft at
`training.eagle4_stage=draft` or fits the candidate factors at `candidates`.
The [adaptation guide](../../../../docs/EAGLE4.md) covers externally captured native logits, alignment, precision, WSD and export.

The [MiniCPM5-1B comparison](../../../../docs/research/eagle4-minicpm.md) used a separate scratch draft.
Its extension passed the FP32 reference but failed strict BF16 parity; native EAGLE4 matched 7/8 cases.
[Native-logit tests](../../../../tests/test_eagle4_native_logits.py), [export tests](../../../../tests/test_eagle4_export.py)
and [comparison tests](../../tests/test_eagle4_comparison.py) cover different contracts.
Online streaming and optimized EAGLE-HARNESS serving require their own runtime gates.

The [25,000-update follow-up receipt](../../../../docs/research/results/eagle4-minicpm-gpu-followup-20261030.json)
compares the candidate and dense proposal heads on the same 64 diagnostic
prompts. FP32 matched 64/64 before and after; BF16 matched 53/64. FP32
candidate acceptance went from 2.052% to 1.927%. This run did not establish
an improvement or test native serving of the final export. See the
[controls and evaluation command](../../../../docs/research/eagle4-minicpm.md#further-10000-updates-with-the-gpu-optimizer).
