# Third-party implementation notes

This fork keeps the upstream licenses in each copied repository. The local
changes are integration code and do not replace upstream copyright headers.

| Source | Pinned workspace revision | Used for |
|---|---|---|
| [TorchSpec](https://github.com/lightseekorg/TorchSpec) | `bd64d93` | resource/checkpoint design reference and FSDP2 boundary |
| [AngelSpec](https://github.com/Tencent/AngelSpec) | `d3412be` | packing/objective design reference |
| [SGLang](https://github.com/sgl-project/sglang) | `8ba213f` | pinned target serving and capture backend |
| [mlx-dspark](https://github.com/ARahim3/mlx-dspark) | `4143092` | MLX inference reference, not a trainer dependency |

## Borrowed boundaries

- TorchSpec `torchspec/config/train_config.py`, `torchspec/ray/placement_group.py`,
  and `torchspec/training/checkpoint.py` were reviewed at `bd64d93`. Only the
  declarative launch/checkpoint contract was adapted in
  `specforge/torchspec_bridge.py`; Ray actors and DCP state wrappers are not
  copied. The reference project's full MIT notice is bundled in
  `licenses/torchspec-LICENSE`.
- AngelSpec `angelspec/training/data_fetcher.py` and
  `angelspec/tests/test_packing_safety.py` were reviewed at `d3412be`. The
  strict validation ideas and anchor-policy behavior were adapted into the
  existing DSpark model/config; Mooncake packing/controller code is not copied.
  The reference project's full license and third-party exceptions are bundled
  in `licenses/angelspec-LICENSE`; this does not imply all its components are copied.
- mlx-dspark `src/mlx_dspark/config.py` and `src/mlx_dspark/server.py` were
  reviewed at `4143092` for serving-state separation only. No MLX code is
  imported by SpecForge.

## Sync policy

Upstream code is copied only when a failing gate identifies a concrete
requirement. Preserve the upstream license header, record the source commit in
this table, add a focused contract test, and keep the local adapter boundary
independent from the upstream trainer loop. Do not merge duplicate controller,
Ray, or checkpoint implementations without a passing single-node gate first.
