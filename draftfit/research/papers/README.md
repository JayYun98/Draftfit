# Paper experiments

Use the [research CLI](../../../docs/research/README.md#run-a-recipe) to select a method.
Each directory below links its implementation, inputs, and evidence.

| Task | Methods |
| --- | --- |
| Prepare or convert a draft for training | [ALR/IRA](alr_ira/README.md), [Osprey](osprey/README.md) |
| Propose tokens from verified history | [NgramRL](ngramrl/README.md), [SpecMem](specmem/README.md), [session lookup](session_lookup/README.md) |
| Generate and filter draft trees | [CAST](cast/README.md), [ValuSpec](valuspec/README.md) |
| Refine a response or select draft cache positions | [ARR](arr/README.md), [RIFT](rift/README.md) |
| Evaluate a numerical kernel, layout, or trace policy | [TreeWY](treewy/README.md), [query layouts](query_layout/README.md), [MoE-SpAc](moe_spac/README.md) |
| Compare target and proposal distributions | [D3D corrected KL](d3d/README.md), [REAL loss](real/README.md) |
| Rank speculative candidates or prefixes | [DICE token ranking](dice/README.md), [DForkTree scoring](dforktree/README.md), [MFD prefix selection](mfd/README.md) |
| Choose a draft length from cost and acceptance profiles | [TableTune](tabletune/README.md) |
| Select configurations from observed costs and correctness | [Redline](redline/README.md) |
| Measure attempted proposals and compare reference timings | [Acceptance position](acceptance_position/README.md), [paired timing](apple_silicon_measurement/README.md) |

The [source catalog](../catalog.py) also records papers with no executable implementation.
For experimental backends and target integrations, use the [platform index](../../../docs/research/experimental-platform.md).
