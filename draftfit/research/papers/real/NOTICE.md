# REAL loss reference

The opt-in REAL component adapts the accepted-prefix estimator and loss
mathematics from King-ty/speculators at
`3dff4579c5f93f692108df8b5178c9e9200b8536`. The relevant source is
`src/speculators/models/dflash/metrics.py`. The paper is
https://arxiv.org/abs/2611.14399v1.

Source: https://github.com/King-ty/speculators/tree/3dff4579c5f93f692108df8b5178c9e9200b8536.
License: Apache-2.0, retained in `REAL_LICENSE`.
The pinned source tree has no root NOTICE file.

Local changes provide a standalone, proposal-only tensor boundary, explicit
validation, FP64 preservation, and a synthetic CPU research recipe. No author
training package, fused kernel, model weight, or serving pipeline is included.
This experimental component does not reproduce the paper's full training.
