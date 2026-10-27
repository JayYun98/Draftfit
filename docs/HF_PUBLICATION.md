# Background Hugging Face publication

The unattended controller can publish selected family exports while later
training work continues. Uploading is optional and uses immutable snapshots:
training never writes into a directory being uploaded.

## Enable it in an unattended run

Add this fragment to the [controller configuration](UNATTENDED_NATIVE.md).
Keys must match every manifest arm, and each arm needs a distinct repository:

```json
{
  "publication": {
    "repos": {
      "dspark": "your-account/your-target-dspark",
      "ce": "your-account/your-target-dflash",
      "selector": "your-account/your-target-dflash2"
    },
    "private": true
  }
}
```

Authenticate the instance through the normal Hugging Face credential mechanism
with write access to those repositories. Set `private` explicitly. Include a
model card in the completed export when publishing manually; only report
measurements from the corresponding artifacts and runtime.

The controller prepares a durable publication job, starts a background worker,
and checks its receipt. Pending work is bounded. A remote commit is verified
against the expected file inventory before the publication is accepted. Failed
jobs remain retryable without retraining. Snapshot creation and final draining
still take time and local disk; background upload is not free or instantaneous.

## Publish an already completed export

The preparation API copies and hashes a stable export into a new job directory:

```python
from draftfit.export.hf_publish import prepare_publication

prepare_publication(
    folder="./exports/dspark",
    repo_id="your-account/your-target-dspark",
    job_dir="./publication-jobs/dspark-step100",
    run_id="my-run-step100",
    private=True,
    resume_from="./joint-output/joint-step100.pt",
)
```

Keep the source unchanged until preparation returns, then run the durable job:

```sh
python -m draftfit.export.hf_publish ./publication-jobs/dspark-step100 \
  --max-attempts 3
```

Use the same command to retry the existing job. Create a new job for a new
checkpoint. Inspect `receipt.json` for verified status and remote identity;
process launch or upload initiation is not publication success.

## What publication preserves

A snapshot contains files in the completed HF export and an integrity manifest.
The optional `resume_from` checkpoint is referenced and hashed, **not uploaded
or deleted**. Optimizer state, token cursors, controller state, datasets and
training logs need separate retention/backup. Do not delete the instance merely
because model upload succeeded when those recovery artifacts are still needed.

Local mocked-transport tests cover publication state handling; they do not prove
live Hub permissions, transfer throughput or eventual network success. Validate
those separately before relying on remote publication as the sole model copy.
