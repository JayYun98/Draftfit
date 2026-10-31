# Batch calibration and runtime forecasts

Calibrate a supported joint-family manifest before a long run. A probe creates
fresh scratch models and optimizer state in its own subprocess, measures capture,
all family backwards and the optimizer update, then exits without a checkpoint.
It never changes a running optimizer or an exact-resume checkpoint.

```sh
draftfit calibrate batch --manifest family.json --output ./batch-probe \
  --device cuda --maximum 8 --effective-batch 8 --probe-steps 3 --max-seconds 120
```

The candidate microbatches are powers of two that divide the requested effective
batch. Accumulation compensates for the microbatch difference. The first update
is warmup; the remaining updates determine throughput. The receipt records peak
reserved memory, available memory and the full resolved configuration contract.
The selector excludes OOM, failed and over-budget trials; candidates that cannot
fit their accumulation window in the configured feature queue are recorded as
incompatible and skipped. Within 3% of the fastest
measurement it chooses the smaller microbatch. A shared GPU's measured throughput
can be affected by other jobs. Keep the same data, target, capture layers, draft
geometry, dtype and training options for a meaningful calibration.

Inspect `batch-probe/calibration.json` and each `batch-*/probe.log`. Use the selected
`batch_size` and `accumulation_steps` in a new run or a predeclared unattended
phase. This command does not silently edit a training configuration. A failed
probe leaves its diagnostic files; use a fresh output directory for a retry.

## Forecast from actual observations

```sh
draftfit calibrate forecast --profile run/learning-curves.jsonl \
  --current-step 100 --target-step 10000 --available-seconds 3600 \
  --overhead-seconds 300 --hourly-usd 0.5 --remaining-usd 0.6 \
  --transfer-reserve-usd 0.1 --checkpoint-interval 250 \
  --evaluation-interval 250 --verification-interval 1000 \
  --checkpoint-seconds 10 --evaluation-seconds 5 --verification-seconds 30 \
  --output forecast.json
```

Only one attempt's monitor log is accepted. Optimizer-only time is not used as
end-to-end update time. The forecast uses capture-inclusive samples, p90 update
time, measured checkpoint/evaluation/verification overhead and a safety factor.
The cost ceiling can only reduce available time. One-time setup, export and
recovery allowance belongs in `overhead_seconds`; network reserve belongs in
`transfer_reserve_usd`. The optional per-operation values above are explicit
conservative bounds in seconds, not measurements. When observations are also
available, the forecast uses the larger of the bound and measured p90. Missing
costs for scheduled operations make the result inadmissible with
`unmeasured_recurring_cost`; estimated time, cost and feasible-step ceiling remain
unknown rather than assuming those operations are free. The receipt distinguishes
measured, explicitly bounded and unmeasured costs.
This is an empirical estimate, not a rental quote or a guarantee of completion.

## Optional unattended admission

Add this top-level object to an existing unattended configuration:

```json
{
  "forecast": {
    "hourly_usd": 0.5,
    "overhead_seconds": 300,
    "safety_factor": 1.25,
    "minimum_samples": 5,
    "checkpoint_seconds": 10,
    "evaluation_seconds": 5,
    "verification_seconds": 30
  }
}
```

At a committed checkpoint boundary, the controller records its forecast in
`state.json`. Once enough samples exist, it stops with
`forecast_runtime_budget` if the next policy interval will not fit. Best/latest
exports and the resume checkpoint are preserved before this decision. With too
few samples the estimate is advisory and the controller's existing hard budget
and deadline remain authoritative. Once enough update samples exist, scheduled
operations without measurements or explicit bounds stop the run with
`forecast_unmeasured_overhead`. The feature is opt-in; existing configurations
retain their current admission policy. It does not rent, stop or delete hardware.

See [unattended execution](UNATTENDED_NATIVE.md),
[monitor metrics](JOINT_MONITORING.md) and
[the integration contract](specs/training-feature-integration.md).
