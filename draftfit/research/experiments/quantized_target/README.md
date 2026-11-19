# Quantized-target draft adaptation

Use `draftfit target prepare --teacher-backend transformers` with `--target-quantization bitsandbytes-4bit`
or `bitsandbytes-8bit` on a floating target. For an already packed checkpoint select `auto`.
[Preparation examples](../../../../docs/QUANTIZED_TARGETS.md) cover fresh and weights-only initialization.
The [target loader contract](../../../../draftfit/modeling/target/quantization.py) binds capture and training to the quantization identity.

The ordinary route requires floating embeddings and output head from the same target.
[CPU tests](../../../../tests/test_quantized_target.py) exercise cache isolation and export metadata.
The [GPU gate](../../../../tests/gpu/quantized_target.py) did not reach model construction in the
[recorded attempt](../../../../docs/reports/quantized-target-adaptation.md); there is no NF4/INT8 training or serving result from it.
