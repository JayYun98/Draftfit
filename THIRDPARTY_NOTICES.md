# Third-party implementation notes

DSpark Train Platform is derived from SpecForge and retains applicable upstream
licenses and copyright headers. Its locally maintained application, training and
runtime modules do not replace or erase the authorship of adapted source.

| Source | Pinned workspace revision | Used for |
|---|---|---|
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) | `bd64d93` | resource/checkpoint design reference and FSDP2 boundary |
| [z-lab/dflash](https://github.com/z-lab/dflash) | `07ebd93` | DFlash2 grouped convolution and candidate selector |
| [AngelSpec](https://github.com/Tencent/AngelSpec) | `d3412be` | packing/objective design reference |
| [SGLang](https://github.com/sgl-project/sglang) | `8ba213f` | pinned target serving and capture backend |
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | `4143092` | MLX inference reference, not a trainer dependency |

## Borrowed boundaries

- TorchSpec `torchspec/config/train_config.py`, `torchspec/ray/placement_group.py`,
  and `torchspec/training/checkpoint.py` were reviewed at `bd64d93`. Only the
  declarative launch/checkpoint contract was adapted in
  `dspark/torchspec_bridge.py`; Ray actors and DCP state wrappers are not
  copied. The reference project's full MIT notice is bundled in
  `licenses/torchspec-LICENSE`.
- z-lab/dflash DFlash2 grouped-convolution, candidate-selector, configuration,
  and serving-state boundaries were reviewed at `07ebd93`. The local model and
  training wrapper adapt those boundaries to the platform's inherited Qwen3 DFlash
  seams; no z-lab runtime dependency is added. The source notice is bundled in
  `licenses/z-lab-dflash-LICENSE`.
- AngelSpec `angelspec/training/data_fetcher.py` and
  `angelspec/tests/test_packing_safety.py` were reviewed at `d3412be`. The
  strict validation ideas and anchor-policy behavior were adapted into the
  existing DSpark model/config; Mooncake packing/controller code is not copied.
  The reference project's full license and third-party exceptions are bundled
  in `licenses/angelspec-LICENSE`; this does not imply all its components are copied.
- mlx-dspark `src/mlx_dspark/config.py` and `src/mlx_dspark/server.py` were
  reviewed at `4143092` for serving-state separation only. No MLX code is
  imported by this platform.

## Sync policy

Upstream code is copied only when a failing gate identifies a concrete
requirement. Preserve the upstream license header, record the source commit in
this table, add a focused contract test, and keep the local adapter boundary
independent from the upstream trainer loop. Do not merge duplicate controller,
Ray, or checkpoint implementations without a passing single-node gate first.
