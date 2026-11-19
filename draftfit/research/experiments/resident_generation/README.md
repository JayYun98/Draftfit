# Resident target generation and CUDA batch tuning

Use `draftfit data regenerate --backend transformers` with a pinned local target and reviewed conversation rows.
Add `--auto-batch` in a CUDA runtime to probe batch sizes before generation.
The [resident generator](../../../../draftfit/data/local_regenerate.py) and [batch selector](../../../../draftfit/data/generation_batching.py)
keep the model loaded and retry pending rows on CUDA OOM.

See the [command and journal contract](../../../../docs/basic_usage/data_preparation.md#resident-local-generation),
[CPU generation tests](../../../../tests/test_local_regeneration.py) and [batch policy tests](../../../../tests/test_generation_batching.py).
Completed rows can be resumed; this does not restore the exact sampling RNG sequence or validate CUDA tuning performance.
