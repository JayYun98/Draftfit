# Add a target through configuration

Inference backends execute targets. Draftfit connects target metadata, capture
layers, existing draft architectures and data. Configuration does not implement
a missing backend architecture.

## Prepare with an existing implementation

Use `draftfit algorithms` for registered methods, architectures and overrides.
This example needs target depth of at least six, compatible offline features,
and target config/tokenizer metadata in `/models/my-target`.

```sh
draftfit target prepare /models/my-target --local-only \
  --output-dir ./projects/my-target --strategy dspark \
  --hidden-states ./features/my-target \
  --set 'model.target_layer_ids=[1,3,5]' \
  --set model.draft_num_hidden_layers=2
```

Add `--draft-config ./my-draft.json` to adapt an existing configuration.
Its `architectures` must name a compatible entry in `draftfit algorithms`.
Existing values change only when explicitly overridden. You can also edit the
generated `draft.json` and `train.json`.

For online data, replace `--hidden-states` with `--train-data ./train.jsonl`.
The `hf` renderer requires native `{% generation %}` spans; validate actual
assistant masks separately. Select registered renderers with
`--set data.chat_template=NAME`. Remote targets use repository ID and
`--revision`; preparation pins the commit SHA.

Use individual field paths for `--set`. Source-bearing parent replacements
such as `model={...}` or `data={...}` and source overrides are rejected.
Choose sources with `--draft-config`, `--draft-checkpoint`, `--train-data`
or `--hidden-states`.

`inspect` reads JSON/template metadata only. `prepare` also avoids remote code
execution and rejects `--set model.trust_remote_code=true`. If reading a custom
draft configuration requires remote code, first create compatible local JSON.
If training needs remote code, review it before explicitly enabling
`model.trust_remote_code` in generated `train.json`; training's opt-in remains.

## Scaffold from reviewed inspection

```sh
draftfit target inspect /models/my-target --local-only > inspection.json
# Review recommendations and adjust candidate values.
draftfit target scaffold --inspection inspection.json --output ./custom-run.json
```

Input is `target_inspect_v1`, not a directly supplied `target_spec_v1` sidecar.
Choose `recommendations.target_tap_candidates`, `draft_depth_candidates`
and `train_block_candidates`. A scaffold does not replace `prepare`'s
draft/target consistency checks. Do not invent facts to imply compatibility.

## Validation scope

Preparation checks registered implementations, fields/sources, draft dimensions
and capture indices, not feature contents or GPU execution. Manifest statuses
for capture/training/export/serving remain `not_run`.
Validate tokenizer masks, feature boundaries/normalization, greedy token parity,
training, export and serving on the real target. Hybrid/recurrent targets also
need real-backend state snapshot, rollback and replay checks.
