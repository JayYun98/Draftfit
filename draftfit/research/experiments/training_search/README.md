# Optuna training-option screening

Use `python -m draftfit.training.search validate --contract CONTRACT.json` to check an immutable study.
The [controller](../../../../draftfit/training/search.py) proposes bounded overrides and records checkpoint observations;
the existing trainer runs the trial. The [study guide](../../../../docs/TRAINING_SEARCH.md) specifies the
receipt format, budgets and optional Optuna 4.5.0 dependency for SQLite operations.

[Controller tests](../../../../tests/test_training_search.py) separate dependency-free validation from durable SQLite tests.
A score ranks configurations only within one family and evaluation protocol. Serving correctness, acceptance and throughput need independent measurements.
