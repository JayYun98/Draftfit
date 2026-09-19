# Target inspection fixtures

These files are metadata-only `target_inspect_v1` snapshots. They are used by
the local scaffold and schema tests; no model weights are vendored. Re-run
`specforge target inspect <repo> --revision <commit>` before a GPU run and keep
the returned Hub commit in the sidecar.

| Fixture | Lane | State | First required gate |
|---|---|---|---|
| `lfm2.5-1.2b-instruct.json` | hybrid | conv + KV | state snapshot rollback/replay |
| `granite-4.2-3b.json` | dense | KV | target-only greedy parity |
| `granite-swash-3b-a600m.json` | MoE hybrid | MoE + KV | explicit renderer + state replay |
| `rwkv7-1.5b.json` | recurrent | recurrent | backend/state probe |
| `qwen3.8-27b.json` | linear hybrid | linear + KV | large-target backend probe |
