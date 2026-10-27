# Optional local web workspace

The workspace adapts the current Draftfit engine to a local browser. It reuses
the earlier experiment UI's Vite/React tooling, upload flow, process records,
evidence helpers and checkpoint timeline. Training, bounded feature queues,
checkpoint recovery, verification and unattended policy stay in the engine.

## Start from a source checkout

Follow [environment setup](ENVIRONMENT.md) for the trainer. The web service has
its own small, hash-locked Python profile and does not install or replace Torch,
Transformers, SGLang, Mooncake or a provisioned CUDA runtime.

```sh
# Set this to the interpreter of your existing, compatible training environment.
export DRAFTFIT_TRAINER_PYTHON=/path/to/trainer/bin/python
bash scripts/run_web.sh
```

Open `http://127.0.0.1:8080`. The script prepares `.venv-web`, installs the
frontend with `npm ci`, builds it and starts `draftfit.web.server` on loopback.
Node 20.19+ or 22.12+ and uv 0.9.18+ are required. To use another existing HTTP
environment set `DRAFTFIT_WEB_PYTHON` to its Python interpreter; it must already
contain the locked web dependencies.

By default, config/data preparation, command previews, plans, token comparisons
and saved-state reads are enabled. Starting or resuming training requires:

```sh
bash scripts/run_web.sh --allow-local-execution
```

This enables local engine execution in the explicitly selected trainer runtime.
It does not provision hardware. Engine prerequisites and budget admission still
apply. Jobs have a bounded timeout (default one hour). Stop terminates the UI's
child process; recovery uses engine checkpoints and its existing resume flow.
Server shutdown also stops its children. Restart marks unfinished UI jobs as
interrupted and never automatically restarts training.

The process environment takes precedence over dotenv. The project `.env` is
loaded without overriding explicit variables. Only when that file is absent,
the launcher supplies the user's global dotenv fallback to this local process.
Credentials are not accepted in saved config fields. The UI does not display
environment files or credential values.

## Use the workspace

1. **Configure:** import JSON/YAML or choose a valid current-engine recipe.
   Validation uses `draftfit.application.composition.resolve_run`, including
   registered algorithm, modality, data mode and topology restrictions. Save a
   new experiment after editing. Existing records/configs are not overwritten.
2. **Datasets:** upload reviewed conversation JSON/JSONL (64 MiB maximum). The
   existing `prepare_dataset` produces canonical rows and a deterministic
   grouped holdout. Selecting it for an online config attaches the training file
   only. The current single-run capture path rejects online evaluation; keep the
   holdout for separate capture or verification. Offline configs need captured
   feature files; attaching raw conversations to them is rejected.
3. **Experiments:** preview the command, run the engine launch plan, or train
   with `draftfit train -c ...`. A checkpoint resume belongs in the saved engine
   config (`training.resume_from`). Compare existing token-ID JSON arrays using
   `draftfit validate tokens`. Read process output and exit status separately
   from validation evidence.
4. **Unattended:** select an existing `draftfit-unattended-v1` controller config
   within this project. Plan/start/resume delegate to `draftfit unattended`.
   Saved-state reads show `state.json` and only the latest attempt's
   `learning-curves.jsonl`, including its last saved queue observation. The UI
   never drains a queue, edits controller state, bypasses a lock or fabricates a
   fresh external budget observation. Use **Read saved state** to refresh it.

Relative training paths retain the engine's normal meaning: they are resolved
from the selected project root, not from the web record directory. Uploaded
dataset paths are absolute. Controller configs and token comparison paths must
be inside the selected project, including after resolving symlinks. Inputs and
outputs used by a trusted training config retain the engine's existing access
rules. Work with reviewed local configs.

UI records live under `outputs/web-workspace/` by default; use `--state-dir` to
choose another location. These records are UI history, not a training scheduler.
Only one UI-launched process runs at a time. The existing unattended controller
still owns its queue and durable recovery. An externally launched engine job is
not managed or cancelled by this workspace.

## Supported boundary

The adapter supports only plan/train, supplied-token comparison and the existing
unattended plan/start/resume path. Remote instance actions, rental, publishing,
custom Python adapters, arbitrary command execution, old research/pretrain
pipelines and unregistered algorithms are rejected. A controller configured for
publication is rejected by this adapter. Other current CLI workflows can be used
directly; absence of a UI control does not add or remove engine capabilities.

The algorithm catalog reports registered implementations, not universal target
support. A successful plan/process/config check is not GPU correctness,
compatible export serving, heldout quality or a speedup measurement. Timeline
rows say **recorded**, not **passed**; each row requires its own saved event.
Missing stages remain unobserved even after the controller finishes. Native
verification and serving smoke retain the engine's existing evidence limits.

This is a loopback-only personal service, protected by a per-process request
token and origin checks for mutations. It is not a hosted or multi-user service.
The built frontend is served from the source checkout's `web/dist`; shipping
frontend assets inside an installed wheel is outside this integration.

## Verification commands

```sh
npm ci --prefix web --no-audit --no-fund
npm test --prefix web
npm run build --prefix web

# Use an environment containing requirements-web.lock, pytest and httpx.
uv run --no-project --python .venv-web/bin/python python -m pytest tests/test_web -q
```

`test_engine_cpu.py` additionally uses `DRAFTFIT_TRAINER_PYTHON` and skips if it
is absent. With a provisioned CPU interpreter it exercises actual tiny training
and the current unattended plan/run/resume via HTTP. These checks establish local
adapter behavior only. GPU execution and production serving gates are separate.
