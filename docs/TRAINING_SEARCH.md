# Bounded training option screening

`python -m draftfit.training.search` is an optional, standalone Optuna TPE
controller. It proposes whitelisted config overrides and accepts checkpoint
observations from an existing trainer. It does not start processes, load models,
query CUDA, modify checkpoints, search checkpoint directories, merge weights, or
change the training CLI, campaign scheduler, or unattended controller.
It does not autonomously execute trials or replace campaign policy. The existing
campaign/controller still owns execution, resource enforcement and promotion.

## Meaning of a result

Every result has `metric_role: screening_only`. A study minimizes one scalar
heldout objective loss within **one family and one immutable evaluation
protocol**. The final score is the last prescribed checkpoint's loss, not the
best intermediate loss. `MedianPruner` compares intermediate observations at
matching update counts. Completed trials inform TPE after the configured startup
period. TPE ranks only identical validation objective cohorts; a shared family
name alone is insufficient. TPE uses good/bad parameter densities; this is not Gaussian process
Bayesian optimization; see the [Optuna TPE reference](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html). Cost is bounded by admission and reservations; the sampler
does not fit a cost model or optimize loss per second.

Do not compare raw DSpark, DFlash, DFlash2, or differently weighted objective
losses. Objective weights, anchors, seeds, data, batch geometry, training horizon,
and draft architecture are frozen by the base config and identity. Changing a
family, objective, data split, checkpoint, search space, pruning protocol, or
budget requires a different study contract. A new study requires a new explicitly
approved campaign budget; creating one does not refund the original study.

No acceptance, parity, or throughput metric is produced or inferred. A selected
checkpoint still needs independent native generation, correctness/parity,
acceptance and production workload throughput checks in its actual serving
runtime. Keep those original reports separately; loss screening is not a serving
pass. See [public support](PUBLIC_SUPPORT.md) and the existing campaign records.

## Optional controller environment

The module, `--help`, and `validate` use the standard library only. Other commands
load Optuna lazily and require the tested version **4.5.0**. Installing the
optional `search` extra adds Optuna; it is absent from mandatory dependencies and
the CPU lock. Do not synchronize or replace a custom training/serving runtime to
use this controller. Follow [environment management](ENVIRONMENT.md).

For a separate controller environment, from the checkout:

```sh
uv venv --python 3.12 /tmp/draftfit-search
uv pip install --python /tmp/draftfit-search/bin/python 'optuna==4.5.0'
/tmp/draftfit-search/bin/python -m draftfit.training.search --help
```

The controller may live on a POSIX CPU host (Linux/macOS). Use a local SQLite path on one host;
network filesystems and distributed writers are unsupported. Adapter commands
serialize with an advisory `.lock` file beside the database. Do not write to the
Optuna study through another client. Snapshot the complete SQLite database while
the controller is idle; it contains both the Optuna tables and the adapter's
contract/reservation ledger. Retain its lock path rather than deleting a live lock.

## Immutable contract

All keys below are required; unknown keys, duplicate JSON keys, non-finite
numbers, invalid types and unsupported parameters are rejected. JSON object key
order does not affect the SHA256 contract digest. All contract content, including
budgets and conditions, is bound by that digest. Reusing a study name in the same
database with different content fails. An unmanaged Optuna study is not adopted.

This **illustrative metadata fixture** is valid JSON, not an experiment record.
Replace the example names and digests with reviewed immutable artifact identities
before requesting a real trial. Digests are caller attestations: the adapter
checks their shape and consistency, and does not read/hash model weights or
prove that the declared artifacts were used. For a directory checkpoint or
dataset, hash a deterministic manifest of all relevant file hashes. Preserve the
manifest itself with the campaign. The split manifest should bind source/group
lineage, disjoint train/validation rows and token content, preprocessing and
tokenizer revisions. A different file hash alone does not prove split isolation.

```json
{
  "schema_version": 1,
  "study_name": "dflash2-screening-example",
  "identity": {
    "target": {
      "model_id": "example-only/target",
      "revision": "immutable-target-revision",
      "weights_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    },
    "draft": {
      "config_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "checkpoint_sha256": "2222222222222222222222222222222222222222222222222222222222222222"
    },
    "data": {
      "train_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
      "validation_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "split_sha256": "5555555555555555555555555555555555555555555555555555555555555555"
    },
    "objective": {
      "family": "dflash2",
      "name": "full_heldout_objective_loss",
      "config_sha256": "6666666666666666666666666666666666666666666666666666666666666666",
      "validation_protocol_sha256": "7777777777777777777777777777777777777777777777777777777777777777"
    },
    "base_config_sha256": "8888888888888888888888888888888888888888888888888888888888888888",
    "code_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "space": {
    "training.learning_rate": {
      "kind": "float", "low": 0.00001, "high": 0.0002, "log": true
    },
    "training.lr_scheduler": {
      "kind": "categorical", "choices": ["constant", "cosine"]
    },
    "training.objective_chunk_blocks": {
      "kind": "int", "low": 8, "high": 32,
      "strategies": ["dflash", "dflash2"]
    }
  },
  "budget": {
    "max_trials": 4,
    "max_total_updates": 160,
    "max_total_seconds": 3600,
    "trial_max_updates": 40,
    "trial_max_seconds": 900
  },
  "memory": {"max_trial_mib": 6000, "reserve_mib": 1024},
  "sampler": {"seed": 42, "startup_trials": 3},
  "pruner": {
    "startup_trials": 3, "warmup_updates": 9,
    "interval_updates": 10, "min_trials": 2
  },
  "report_updates": [10, 20, 40]
}
```

The objective digest must cover the exact loss formula, its weights, anchor and
mask rules, reduction, precision and any family-specific objective settings.
The validation protocol digest must cover heldout evaluation inputs, fixed
evaluation seeds/anchor policy, aggregation and checkpoint schedule. Training
hyperparameters that are being searched are explicit exceptions to the frozen
base config; apply only the proposed overrides to a copy. All trials start from
the same declared draft checkpoint; do not warm-start from another trial's best.

Allowed parameters map directly to `TrainingConfig` fields:

| Parameter | Distribution and bounds |
| --- | --- |
| `training.learning_rate` | `float`, `0 < low <= high <= 1`, explicit Boolean `log` |
| `training.warmup_ratio` | `float`, `0 <= low <= high <= 1`, explicit Boolean `log` |
| `training.max_grad_norm` | `float`, `0 < low <= high <= 100`, explicit Boolean `log` |
| `training.lr_scheduler` | `categorical`, nonempty subset of `constant`, `cosine` |
| `training.optimizer_cpu_offload` | `categorical`, nonempty subset of JSON `false`, `true` |
| `training.objective_chunk_blocks` | `int`, `1 <= low <= high <= 65536`, unit steps |

Every distribution may have an optional nonempty `strategies` list containing
only `dspark`, `dflash`, `dflash2`. A parameter inactive for the contract's family
is omitted, not sampled or written. This supports reusable conditional spaces;
it does not mix families in one study. At least one parameter must be active.
Log ranges require positive bounds. Objective changes, arbitrary dotted config
paths, strings containing commands, paths to resume from, and executable search
expressions are not supported. Batch/accumulation selection remains with the
existing hardware tuning path.

## Ask, run externally, report, tell

Write the contract to `study.json`. These commands return JSON on stdout and
operational errors as JSON on stderr with exit code 2. Argument syntax errors use
the standard CLI usage message. A memory rejection or pruning decision
is a successful recorded operation (exit 0); **inspect `state` before continuing**.

```sh
PY=/tmp/draftfit-search/bin/python
$PY -m draftfit.training.search validate --contract study.json
$PY -m draftfit.training.search ask --contract study.json --storage study.sqlite \
  --free-memory-mib 7577.6 --estimated-memory-mib 5000
```

Only a `running` proposal may be run. Enforce the returned `limits.max_updates`,
absolute `limits.deadline` (UTC epoch seconds) and `limits.max_memory_mib` in the
external controller. Save its `trial`, `contract_digest`,
`config_digest`, and `params` with the copied trainer config. `trial` is the
adapter attempt number, not necessarily `optuna_trial_number`. The config digest
binds the contract, adapter attempt number and proposed parameters. The adapter
does not apply overrides or execute a shell command. Use the existing trainer
and checkpoint manager with the same full training horizon and the externally
enforced trial stop/deadline. Do not shorten the LR horizon to each checkpoint.

At each prescribed boundary, save the real checkpoint and run the existing full
heldout evaluator. Write a receipt with exactly these fields:

```json
{
  "contract_digest": "<64 lowercase hex from ask>",
  "config_digest": "<64 lowercase hex from ask>",
  "checkpoint_sha256": "<64 lowercase hex of checkpoint manifest>",
  "parent_checkpoint_sha256": "<base or preceding reported checkpoint SHA256>",
  "updates": 10,
  "elapsed_seconds": 120.5,
  "validation_loss": 1.25,
  "peak_memory_mib": 4800,
  "free_memory_mib": 2048
}
```

These receipt numbers are schema examples, not measured generation or training
results. Use actual observations only. `updates` counts optimizer updates since
the declared starting checkpoint, not an absolute step inherited from it. Report
every rung in order; completion requires the full schedule. Optuna receives
`step = updates - 1`, so its warmup step is zero based. Elapsed seconds and peak
memory are cumulative since the proposal/run began and cannot decrease. Include
capture, evaluation and checkpoint overhead in elapsed time. The checkpoint
chain begins at the contract's base and links successive reported checkpoints;
keep intervening trainer lineage records separately. A checkpoint cannot be its
own parent. An identical receipt retry is idempotent; a conflicting duplicate,
wrong digest, wrong parent, missing rung or non-finite value is rejected without
adding an observation. An actual finite budget or memory overrun is recorded as
a durable `rejected` trial, with its rejected receipt retained outside Optuna's
objective observations.

```sh
$PY -m draftfit.training.search report --contract study.json --storage study.sqlite \
  --trial 0 --receipt checkpoint-10.json
# Repeat at the next prescribed rung only while state remains running.
$PY -m draftfit.training.search tell --contract study.json --storage study.sqlite \
  --trial 0 --state complete
$PY -m draftfit.training.search status --contract study.json --storage study.sqlite
```

`report` can finish a trial as `pruned`; stop further training when it does.
`tell complete` takes its score from the final validated receipt, so there is no
separate score override or serving score input. Other explicit terminal states
are `pruned` (requires an observation), `rejected`, `interrupted` and `failed`;
each requires `--reason`. They cannot be reopened or changed. To stop a running
trial after a worker or CLI interruption:

```sh
$PY -m draftfit.training.search tell --contract study.json --storage study.sqlite \
  --trial 0 --state interrupted --reason 'Worker stopped; reservation retained'
```

Obtain and retain an `ask` proposal before running an experiment, then attach its
actual observations through `report`/`tell`, including when the controller process
has exited. For a seed configuration, `ask --params seed-params.json` accepts an
exact object of **all active** whitelisted parameter names and values within the
declared distributions. It uses Optuna's public enqueue API, charges the same
full reservation and memory admission gate, and then follows the same receipt
protocol. This can register real finished observations if the operator can attest
that the experiment used this exact base/target/draft/data/objective contract and
has every prescribed checkpoint receipt, cumulative cost and memory measurement.
There is no independent score import or parameter relabeling; incomplete or
incompatible historical experiments are not seed observations. The study's wall
deadline starts when the controller first opens the study; historical run cost
must still fit its charged reservation and reported per-trial caps.

## Persistence, cost and memory gates

Each syntactically valid admitted attempt reserves its **full** per-trial update
and seconds caps before asking Optuna. Rejected memory admission, pruning,
failures and interruptions consume the attempt and retain that entire
reservation. No unused reservation is automatically refunded. Status calls the
totals `reserved`, not actual consumed training time. Status also returns
`charged`: for each attempt, the greater of its reservation and any reported
overrun. Subsequent asks use those charged totals, so a reported overrun cannot
disappear behind a smaller reservation. Receipts retain measured cost separately.
Thus a small budget may deliberately leave unused work; this
conservative first version cannot oversubscribe a budget by repeatedly failing
or restarting before a report.

The contract requires a positive trial count and at least one non-null aggregate
update or seconds cap. Both per-trial caps are mandatory. A new ask must fit all
aggregate reservation caps. If a total seconds cap is specified it is also a
wall deadline from the first opening of the study; the per-trial wall deadline
starts at ask. Idle time and process restarts do not pause those deadlines.
Only one trial can run at once. An abandoned running trial blocks further asks
until explicitly terminated. No timeout or restart silently reruns it. A crash
while a proposal is being allocated becomes a charged `interrupted` attempt on
the next opening. Pending intermediate/final storage writes are reconciled from
the ledger without rerunning training. Sampler seeds are derived from the fixed
study seed and durable attempt number because SQLite does not persist an RNG
stream; there is no pickle or resetting one random stream on each CLI invocation.

`ask` requires caller-supplied fresh free memory and an estimated incremental
trial allocation bounding every possible proposed config (or the explicit seed
config). Admission requires `estimate <= max_trial_mib` and
`estimate + reserve_mib <= free_memory_mib`. Reports require the cumulative peak
to stay within both that estimate and `max_trial_mib`, and observed free memory
to remain at least `reserve_mib`. Values are MiB, not decimal MB. Measure the
same GPU while the reserved vLLM server remains resident; total A100 capacity is
not the available trial budget. With only 7.4 GiB free, budget against that free
allocation and preserve the reserved server. This module cannot enforce a GPU
process's allocator or kill it; the caller must enforce stop/deadline/memory
limits and check decisions before launching/continuing at a boundary. It never
stops or reconfigures the serving process.

## Verification

CPU only, from this checkout and the separate controller environment:

```sh
/tmp/draftfit-search/bin/python -m unittest tests.test_training_search -v
```

The tests use real Optuna SQLite files, separate CLI processes and concurrent
ask attempts; cover intermediate values, final fixed-endpoint scoring, pruning,
conditional spaces, invalid input/lineage, irrevocable caps and rejected contract
reuse; and inject allocation/report/tell storage interruptions followed by a
real process reopening. The dependency-free checks use Python `-S`. If Optuna
is absent, SQLite tests explicitly skip; that is not a persistence pass. No GPU
training, model quality, native acceptance/parity or production throughput is
established by this suite.

## Primary references

- [Optuna 4.5 ask/tell interface](https://optuna.readthedocs.io/en/v4.5.0/tutorial/20_recipes/009_ask_and_tell.html)
- [TPE sampler and startup behavior](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [MedianPruner rules](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.pruners.MedianPruner.html)
- [Trial report and pruning APIs](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.trial.Trial.html)
- [RDB persistence and sampler state](https://optuna.readthedocs.io/en/v4.5.0/tutorial/20_recipes/001_rdb.html)

Source boundaries: `draftfit/training/search.py` owns the optional adapter;
`draftfit/config/schema.py:TrainingConfig` owns parameter meanings;
`draftfit/training/joint_family.py` and `draftfit/training/unattended.py` retain
training/checkpoint/observation lifecycle ownership; and
`draftfit/training/hardware_tuning.py` retains empirical batch probing. This
adapter adds no hooks to those modules.
