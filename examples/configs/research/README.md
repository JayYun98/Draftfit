# Research recipes

These recipes run experimental references and diagnostics.
They do not select a production training or serving backend.
Use the [research guide](../../../docs/research/README.md) for results and limits.

| Recipe | Required input |
| --- | --- |
| [NgramRL](ngramrl.yaml) | A local target model and repeated token prompts |
| [SpecMem](specmem.yaml) | A local target model, user IDs, and query embeddings |
| [ValuSpec + CAST](valuspec-cast.yaml) | A compatible target, DFlash draft, and calibration data |
| [RIFT](rift.yaml) | Independent compatible target and AR draft models |
| [ARR](arr.yaml) | A local target model and stationary revision prompts |
| [TreeWY](treewy.yaml) | The included numerical fixture |
| [MoE-SpAc](moe-spac.yaml) | The included policy trace |
| [TableTune](tabletune.yaml) | Included synthetic cost and acceptance profiles |
| [D3D corrected KL](d3d-corrected-kl.yaml) | Included teacher, student and label fixture |
| [DForkTree scoring](dforktree-scoring.yaml) | Included candidate probabilities and branch evidence |
| [MFD prefix tree](mfd-prefix-tree.yaml) | Included mode probabilities and token budget |
| [REAL loss](real-loss.yaml) | Included proposal logits and target-sampled token IDs |
| [Query layouts](query-layout.yaml) | The included attention fixture |

Run commands from the repository root.
Model recipes contain paths that you must replace.
Research recipes use their own schema.
Do not pass them to `draftfit train`.

```sh
python -m draftfit.research -c examples/configs/research/treewy.yaml --plan
python -m draftfit.research -c examples/configs/research/treewy.yaml
```

Use a new output directory for each run.
