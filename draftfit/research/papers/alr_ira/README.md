# ALR/IRA: prepare target rollouts for distillation

[Preparation](method.py) selects anchors from supervised positions in a JSONL
corpus and greedily extends each prefix with the target. It captures the target
features and labels used by the DFlash trainer. Enabling
`training.in_rollout_anchors` allows a secondary anchor to reuse a primary
rollout's features without another target forward.

Run `python -m draftfit.research -c recipe.json` with the
[ALR/IRA recipe example](../../../../docs/specs/research-methods.md#alrira-preparation-and-training).
Each input row needs integer `input_ids` and a same-length binary `loss_mask`.
Use a dense Llama, Qwen2, or Qwen3 target and a fresh output directory. The
producer writes feature blocks, `capture_manifest.json`, and a per-anchor
receipt in `preparation.json`.

To train, set `training.strategy=alr_ira`, `training.loss_type=dflash`, and
`data.hidden_states_path` to the prepared directory. Target identity, taps and
block size must match. The [registered algorithm](../../../algorithms/alr_ira/providers.py)
owns the cache reader and resume contract; its [model](../../../algorithms/alr_ira/model.py)
and [sparse distillation loss](../../../algorithms/alr_ira/loss.py) own the
training objective. Generic offline teacher captures do not contain ALR anchors.

The [source catalog](../../catalog.py) pins js-lee-AI/ALR-IRA. See the
[source notice](../../NOTICE.md#alr-ira-and-redline) for licensing.

The [regression tests](../../../../tests/test_research_methods.py) cover ALR/IRA preparation and cache reuse. The [MIT license](LICENSE) is included with this package.
