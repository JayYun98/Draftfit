# Research source notices

Native adaptations use the following upstream sources. Transformers and EAGLE
fragments retain their existing Apache-2.0 notices and repository licenses.

## TableTune

`papers/tabletune/upstream.py` is an unmodified copy of `dsd_tuning/core.py` from
[`asm582/tabletune`](https://github.com/asm582/tabletune), commit
`38170878d901cb06f696a122a0b4140a3d365f02`, under Apache-2.0.
The complete upstream license is bundled as `papers/tabletune/TABLETUNE_LICENSE`.
The upstream repository contains no separate NOTICE file.

Source SHA256:
`8d4c7e0d8b5d7a04f85f00fe7c28d80f1f76afd4c497657430b412df744ee80c`.
License SHA256:
`c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c`.

The owned adapter validates prefix survival, rounded token estimates, K0 and
acceptance support for every cost profile. It restricts schedules to the shared
profile coverage and reuses the author's interpolation and table rules.
Stored token estimates may differ from prefix sums by at most 0.0005, matching
the author's three-decimal rounding. The author's full-proposal threshold of
0.98 is retained. Below that threshold, proposal rate scales token gains and
the full-K step cost is retained.
The installer, profiling scripts, plugin, controller and serving engine are not
bundled or invoked. The example profiles are synthetic arithmetic fixtures;
their predicted goodput is not a measured speed or a speedup claim.

## Multi-Future Drafting

See the [MFD source notice and licenses](papers/mfd/NOTICE.md).

## TreeWY

`papers/treewy/upstream.py` contains the first 142 lines (single-tree reference only)
of `vllm/third_party/flash_linear_attention/ops/tree_wy_ref.py` from
`sneha5gsm/vllm`, commit `b073ed6cacfa1cf5ae23111854729e662a72a6d1`.
Those lines are unmodified. The upstream SPDX Apache-2.0 header is preserved;
the full Apache-2.0 license is bundled as `papers/treewy/TREEWY_LICENSE`.
Upstream file SHA256:
`aa5f0f07824065044ee3b1c98a5a05ff542feace826239e0d38e9cc6292f01eb`.
The owned adapter adds strict topology, shape, gate and accepted-node validation.
Neither the upstream Triton kernels nor its vLLM serving lifecycle are bundled.

## D3D corrected KL

`papers/d3d/method.py` adapts the probability correction and forward-KL component from
`training/pruning/losses.py` in `fudan-generative-vision/D3D`, commit
`cd7762436217fb51d7e7a310b310b72766bb816f` (MIT).
Copyright (c) 2026 Fusion Lab: Generative Vision Lab of Fudan University and SII.
Copyright (c) 2026 D3D contributors.
The full, unmodified license is bundled as `papers/d3d/D3D_LICENSE`.
Upstream file SHA256:
`7b1bc6370200212ba06c38e551051c802075db8ba09203d23ec270f4c98d2510`.
The local adapter adds strict input checks, masked-label handling, float64
preservation, actual positive-weight normalization, and an offline recipe.
It does not copy the distributed reduction, activation checkpointing, full
recovery objective, model, layer selection, or asynchronous engine.

## Independently implemented published algorithms

NgramRL (`2610.32881v1`), SpecMem (`2610.31840v1`), ValuSpec
(`2610.31338v1`), RIFT (`2610.31098v1`), ARR (`2610.28294v1`) and the
MoE-SpAc paper-v2 policy (`2603.09983v2`) were implemented independently
from their described algorithms. No unpublished source, ARR source/prompt,
or third-party model weights are included. ARR author code has no verified
reuse license. MoE-SpAc's public MIT preview differs from its v2 policy;
the preview engine is not imported by the native trace runner.
Query-layout, acceptance-position and paired-timing diagnostics implement
bounded numerical/measurement experiments, not the papers' original engines.

## ALR-IRA and Redline

MIT License

Copyright (c) 2026 Jungseob Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Osprey

MIT License

Copyright (c) 2025 sgl-project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## CAST

MIT License

Copyright (c) 2026 Jungseob Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DForkTree scoring reference

Copyright 2026 Zhicheng Zheng, Cheng Zhang, and Chenkai Tang.

Reference repository: https://github.com/Perseus037/DForkTree
Revision: `eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3`.
License: Apache-2.0. The full original license and notice are retained in
`papers/dforktree/DFORKTREE_LICENSE` and `papers/dforktree/DFORKTREE_NOTICE`.

Reviewed source identities:

| File | SHA256 |
| --- | --- |
| `dforktree/ngram_gpu.py` | `97137b4692fbe28cb7bd77e3c711a8ed9a4c0919e7045ad9255c9cfa56a428fd` |
| `dforktree/tree_ops.py` | `5ba3417fcfe251b45859ebc691d7e74f618bb67ebed41d218c1dfbe44c9647c5` |
| `dforktree/fc1_builder.py` | `7a7e261010f332bf7190d7b636c8b8cf582a7d87332f48ec564210e52d99b7a2` |

The local CPU reference implements paper equations 2 and 3 over supplied
conditional and unigram probabilities. It uses CPU floating-point arithmetic
and a true zero tail. The author's builder uses float32 and a tail floor of
`1e-6`. No exact GPU numerical parity is claimed. Raw count extraction,
smoothing, GPU tree construction, GDN verification, and SGLang integration
are not ported. The upstream `ngram_path_fn` returns an undefined `fn` at this
pin; the local component does not import or execute that path.

No code from the upstream SGLang worker or Flash Linear Attention recurrence
is copied by this component. Their attribution remains in the retained
author notice to preserve its original contents.

## REAL loss reference

See the [REAL source notice](papers/real/NOTICE.md) and [Apache-2.0 license](papers/real/REAL_LICENSE).
