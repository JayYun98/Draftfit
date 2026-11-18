# Paired reference timing

The [Apple Silicon evaluation](https://arxiv.org/abs/2610.31479v1) motivates a timing
comparison. The local implementation is the [shared PyTorch
harness](../../diagnostics.py), described in the [research
notice](../../NOTICE.md#independently-implemented-published-algorithms). There is no
Apple-specific runner or MLX port.

**Use.** Set `diagnostics.compare_ar: true`, `warmup`, and `repetitions` in the [NgramRL
recipe](../../../../examples/configs/research/ngramrl.yaml) or another recipe using the
harness. Paired comparison requires greedy generation. For each pair, the harness starts
from the same RNG state and checks the emitted tokens against autoregressive decoding.
Read raw times and medians under `answers[].stats.measurement`; warmups are excluded.

Times cover full-prefix generation on the configured PyTorch device. They do not measure
model loading or establish performance on an Apple production engine.

**Tests.** [RIFT integration
coverage](../../../../tests/research/test_latest_kv_tree.py) checks token equality and
repeated measurement output.
