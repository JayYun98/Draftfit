# Unattended joint training

`draftfit unattended` runs prepared token data through shared target capture,
a bounded RAM queue, joint training, milestone verification, policy decisions,
best/latest exports and durable recovery state inside one instance. It supports
scratch and compatible existing-draft initialization through the same
[family manifest](JOINT_ONLINE.md).

It does not rent hardware, discover arbitrary architectures, generate an
unbounded stream of responses or guarantee a faster draft. A supported target,
compatible draft configuration, tokenized train/holdout data, explicit limits
and an external budget observer are required.

![Unattended capture, joint updates, verification, policy and durable exports.](_static/diagrams/joint-lifecycle.svg)

## Configure and launch

Use the manifest and training fragment in [Joint online](JOINT_ONLINE.md).
Keep `training.joint_control` unset: the unattended controller owns policy.
Use a positive queue byte limit and keep per-arm checkpoint emission enabled.
Save this **controller template** beside `family.json`; replace the deadline
with your actual future Unix deadline and provide `budget.json` before planning:

```json
{
  "schema": "draftfit-unattended-v1",
  "manifest": "family.json",
  "output_dir": "unattended-output",
  "max_steps": 1000,
  "deadline_unix": 0,
  "budget_path": "budget.json",
  "selection": "acceptance",
  "policy": {
    "schema": "draftfit-joint-policy-v1",
    "revision": "my-workload-v1",
    "domains": ["general"],
    "patience": 3,
    "min_requests": 4,
    "next_steps": 100,
    "next_cost_usd": 0.10,
    "limits": {
      "max_usd": 2.0,
      "reserve_usd": 0.3,
      "max_steps": 1000,
      "max_pilot_usd": 0.2,
      "max_pilot_steps": 100,
      "max_exploration_usd": 0.2,
      "max_pilots": 1,
      "max_unvalidated_steps": 100
    }
  }
}
```

The four selected holdout rows in this example must have `domain="general"`.
For multiple domains, list them and provide sufficient selected rows for every
domain. Counts include verification trials. `policy.next_steps` must be at
least the verification interval and an integer multiple of it. Controller and
policy `max_steps` must agree; each phase must fit that total and available data.

`deadline_unix=0` is intentionally expired, not a runnable deadline. Set the
actual deadline before the first invocation; do not extend it on resume.
`next_cost_usd` is an admission estimate, not automatic cloud billing.

```sh
draftfit unattended --config controller.json --plan
draftfit unattended --config controller.json
draftfit unattended --config controller.json --resume
```

Controller paths resolve relative to its file. Manifest and training paths
follow their own config resolution. Planning validates data, contracts and
capacity without training; real model files and a valid environment are still
needed. Use an empty output directory for a new run, and `--resume` for an
existing run.

## Budget observation and stopping

An external observer must atomically refresh `budget.json` throughout the run:

```json
{"timestamp_unix": 0, "spent_usd": 0.0, "reserved_usd": 0.0}
```

Replace the timestamp with the observation time. At runtime, missing, future
or more-than-120-second-old observations fail closed. Do not maintain an
artificial zero-spend heartbeat on paid hardware. The controller combines
observed spend, reservations, declared pilot/verification costs and deadline
admission; an independent cloud guard must enforce the actual instance budget.
It does not stop, delete or back up a cloud instance for you.

Policy patience counts comparable observations, not raw steps. A run can finish
early for plateau, available-data/phase boundaries, budget or deadline. The
configured step maximum is a ceiling, not a promise to use all of it. A finished
phase does not automatically create another phase to fill the total allowance.

## Registered adjustments

`selection` accepts `loss` or `acceptance`. Native reference timing and the
optional SGLang smoke path do **not** support benchmark-quality speed decisions.
Positive `min_speedup`/`min_domain_speedup` thresholds and speed-only preset
triggers are rejected during planning. Use the separate
[campaign workflow](FAMILY_CAMPAIGN.md) for externally measured speed policies.

Declare alternatives in top-level `presets`, for example this optional fragment:

```json
{
  "presets": [{
    "id": "lower-lr",
    "when": ["loss_plateau"],
    "reason": "Compare a lower learning rate after loss plateaus",
    "overrides": ["training.learning_rate=0.00001"],
    "pilot_steps": 100,
    "pilot_cost_usd": 0.10
  }]
}
```

Allowed overrides are batch size, accumulation steps, learning rate, LR
scheduler and warmup ratio. A selected preset starts a recorded **weights-only
phase** with a new optimizer/schedule. Exact recovery within a phase restores
optimizer/RNG/cursor state. Presets cannot silently change family membership,
head depth, objective, target taps, token block size or datasets.

Optional `resources` contains a resource `policy` and `pilot_seconds` for
`initial` plus every registered preset ID. It uses actual GPU/VRAM/queue and
throughput observations with cooldown and bounded admission. See
[`resource_policy.py`](../draftfit/training/resource_policy.py) for policy fields.
It selects registered settings; it does not invent arbitrary batch/LR values or
provide a universal model VRAM/time estimator.

## Verification and publication

Native verification keeps the target resident and supports compatible
existing-draft comparisons. [Queue and verification](SHARED_TARGET_FEATURE_QUEUE.md)
explains offload, exactness and metric scope.

For an optional live-engine handoff, add this controller fragment after
provisioning the pinned SGLang runtime:

```json
{
  "serving": {
    "max_seconds": 300,
    "cost_usd": 0.10,
    "config": {
      "max_new_tokens": 32,
      "trials": 1,
      "context_length": 512,
      "startup_timeout_seconds": 180,
      "request_timeout_seconds": 60,
      "warmup_requests": 1
    }
  }
}
```

This requires `selection="acceptance"`; existing-draft mappings are not
supported in this controller handoff. Startup, requests and cleanup must fit
the remaining deadline and admitted cost. See
[`sglang_producer.py`](../draftfit/training/sglang_producer.py) for supported runtime
identity and configuration checks. A successful short smoke does not certify
long-output parity or serving speed. Known longer-output parity failures remain
an open gate.

[Background HF publication](HF_PUBLICATION.md) is opt-in. It snapshots completed
exports before upload, verifies remote content and preserves durable retry
receipts. Model publication does not back up optimizer state or experiment logs.

## Recovery artifacts

The controller records phases, attempts, decisions, best arms and latest
verified exports in `state.json`. Export inventories and checkpoint hashes bind
results to exact bytes. Best and latest can differ. A newer raw checkpoint after
an interrupted verification is not automatically a verified export.

Keep the output tree, input files and source checkpoints together. On resume,
source/config mutation is rejected; failed observation can be replayed before
new updates. A completed run verifies retained artifacts and returns without
retraining. Publication retries are separate from training retries.

The source-tree CPU smoke is available without renting a GPU:

```sh
python benchmarks/verify_unattended.py --device cpu \
  --output-dir /tmp/draftfit-unattended-smoke-new
```

Use a new output directory. This runs tiny real Torch updates with synthetic
tokens; it does not certify CUDA, cloud billing, live Hub transport or production
serving. See [monitoring](JOINT_MONITORING.md) and
[support boundaries](PUBLIC_SUPPORT.md) before interpreting results.
