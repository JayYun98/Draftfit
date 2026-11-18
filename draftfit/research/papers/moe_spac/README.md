# MoE-SpAc trace policy

[method.py](method.py) updates expert utility for each layer and evaluates the
candidate thresholds against a trace row's free memory, routing frequencies and
cost ratios. The result lists feasible thresholds and the proposed prefetch IDs,
or records a CPU fallback if none fits.

## Run

The [recipe](../../../../examples/configs/research/moe-spac.yaml) includes a
[routing fixture](../../../../examples/configs/research/data/moe-spac.json).
From the repository root:

```sh
python -m draftfit.research -c examples/configs/research/moe-spac.yaml
```

The [policy tests](../../../../tests/research/test_latest_control.py) check
utility transitions, threshold selection, memory-constrained fallback and the CLI.

## Source and limits

Independent implementation of the [paper's v2 policy](https://arxiv.org/abs/2603.09983v2).
The earlier MIT preview uses a different estimator and a fixed threshold;
see the [source notice](../../NOTICE.md). The trace runner neither transfers nor
executes expert weights. Its cost estimates are not latency measurements.
