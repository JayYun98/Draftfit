# Multi-stage family campaigns

`python -m draftfit.campaign` schedules explicit training and evaluation jobs
across milestones. It is separate from the in-process
[unattended controller](UNATTENDED_NATIVE.md): use unattended for resident-target
queue training and native milestone decisions; use a campaign when external
workers produce comparable serving reports and speed-based branch decisions.

Start from [the campaign recipe](../configs/campaigns/joint-autopilot.example.json).
Copy it to `experiment.json` and replace target paths, evaluation command,
resource names, milestones and cost estimates. Compile this recipe into the
scheduler configuration before running it. Neither file is an unattended
controller configuration.

```sh
python -m draftfit.campaign_recipe --spec experiment.json --output campaign.json
python -m draftfit.campaign plan --config campaign.json
python -m draftfit.campaign run --config campaign.json --output campaign-output
python -m draftfit.campaign status --output campaign-output
python -m draftfit.campaign run --config campaign.json --output campaign-output \
  --retry-interrupted
```

## Training and resource scheduling

Milestones divide one declared optimizer horizon into resumable joint-training
segments. A checkpoint resumes the same schedule; it does not restart warmup at
every milestone. Per-arm HF exports feed evaluation workers. Branches are
explicitly registered worker commands; naming a branch does not implement a
new architecture or objective automatically.

Resource names enforce concurrency limits. If training and validation use the
same GPU, give them the same resource name. A separate local MLX evaluator can
use another resource, but the scheduler does not provision machines, transfer
models over SSH or infer whether two labels refer to one physical GPU.

Workers receive `DRAFTFIT_JOB_OUTPUT`, `DRAFTFIT_JOB_RESULT`,
`DRAFTFIT_JOB_ATTEMPT` and `DRAFTFIT_INPUT_RESULTS`. A successful exit code
alone is insufficient: the worker must write the expected completed-result
receipt. Interrupted attempts are retained for diagnosis and require explicit
retry. Optional `fail_fast` stops other workers on failure; otherwise independent
jobs may continue.

## Evaluation and policy

The `draftfit-campaign-evaluation-v1` report binds candidates to the training
result SHA and comparison identity. It includes exact-token parity and per-domain
request counts, output tokens and elapsed time. Speed thresholds, domain floors,
patience and relative-gain decisions require valid comparable reports.
A lower loss or a worker's self-reported speedup alone is insufficient.

The [MLX adapter](../draftfit/campaign_mlx.py) is one external evaluation option.
Its profile and runtime must match the selected checkpoints. Keep failed parity
and missing evaluation visible; do not rank them as validated speed results.

## Budget and evidence boundaries

Campaign cost uses declared fixed charges and hourly elapsed-time estimates,
with a reserve and time ceiling. It is not a live cloud-account bill and does
not replace an independent stop/backup/delete guard. Validate download, setup,
verification and export costs before a paid run.

The scheduler and policy have CPU coverage. Real GPU training, live evaluation,
network recovery and long-run speed improvement still require a bounded test
of the exact target and runtime. See [support boundaries](PUBLIC_SUPPORT.md)
and [workload evaluation](PERFORMANCE_GATE.md).
