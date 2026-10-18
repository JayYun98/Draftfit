# Joint training metrics and decisions

Joint runs write machine-readable records so training progress, policy decisions
and inference measurements can be inspected separately. Use these records for
plots instead of scraping progress bars.

| Artifact | Meaning |
| --- | --- |
| `monitor.json` | Run identity, initialization/resume provenance and arm metadata |
| `learning-curves.jsonl` | Train/evaluation/checkpoint events, losses and timing counters |
| `verification-step*.json` | Native exact-token checks, acceptance counts and per-domain diagnostic timing |
| `joint-step*.pt` | Authoritative weights, optimizer, RNG and consumed cursor |
| `control-step*.json` | Direct joint-control decisions, when that controller is enabled |
| Unattended `state.json` | Phase/attempt history, decisions, best/latest export bindings and completion reason |
| Publication `receipt.json` | Verified Hub revision/inventory or publication failure |

Names and placement can be phase-specific. Resolve artifacts through the run's
state and checkpoint references instead of assuming the largest filename is
validated or best. Optional SGLang reports are separate from native verification.

## Interpret the metrics

- Training loss is measured before an update; held-out loss is measured after
  updates. DSpark, DFlash and DFlash2 have different objectives, so their raw
  losses are not a cross-family quality ranking.
- An optimizer step may contain several microbatches. A data row, generated
  response, proposed token and optimizer step are different units.
- Joint-step timing covers draft training work; feature capture, validation,
  checkpoint writes and process startup must also be included in campaign wall
  time. Do not add per-arm rates to claim overall throughput.
- CUDA allocator peaks describe the measured process/interval, not total device
  memory. Missing CPU/GPU telemetry is missing data, not zero usage.
- Acceptance is computed from accepted/proposed speculative token counts.
  Training token accuracy is not acceptance. Failed exact-token parity must
  remain visible and prevents treating that result as a correctness-qualified
  speedup.
- Native verification uses full-prefix generation without KV caching and labels
  its timing scope. Live-engine smoke timings and separate MLX benchmarks are
  distinct measurements; compare the same prompts, runtime and generation setup.
- Policy patience counts comparable validation observations. Export identity,
  objective and workload fingerprints prevent mixing unrelated candidates.

## CSV and optional MLX import

```sh
python -m draftfit.training.joint_monitor \
  --run-dir /path/to/joint-output --csv /path/to/curves.csv
python -m draftfit.training.joint_monitor \
  --run-dir /path/to/joint-output --import-mlx /path/to/results.jsonl
```

The importer requires the expected benchmark fields, checkpoint/step identity
and artifact provenance. It rejects conflicting or invalid results. Run one
importer per output directory and choose a new CSV destination. Missing metrics
stay missing; they must not appear as fabricated zero values in plots.

The source-tree MLX benchmark adapter can record joint-run identity directly:

```sh
python benchmarks/benchmark_mlx_family.py \
  --runtime /path/to/mlx-dspark --manifest /path/to/step-arms.json \
  --data /path/to/heldout.jsonl --output /path/to/results.jsonl \
  --per-domain 8 --trials 3 --max-tokens 64 \
  --joint-run-dir /path/to/joint-output --joint-step 100
```

MLX requires its own compatible Apple Silicon inference environment and export.
It is optional evaluation support, not an MLX training backend or a substitute
for validating the CUDA serving artifact. A recorded conversion provenance is
not by itself numerical parity proof.
