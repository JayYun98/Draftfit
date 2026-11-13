# Research source notices

Native adaptations use the following upstream sources. Transformers and EAGLE
fragments retain their existing Apache-2.0 notices and repository licenses.

## TableTune

`_vendor/tabletune.py` is an unmodified copy of `dsd_tuning/core.py` from
[`asm582/tabletune`](https://github.com/asm582/tabletune), commit
`38170878d901cb06f696a122a0b4140a3d365f02`, under Apache-2.0.
The complete upstream license is bundled as `_vendor/TABLETUNE_LICENSE`.
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

## TreeWY

`_vendor/treewy.py` contains the first 142 lines (single-tree reference only)
of `vllm/third_party/flash_linear_attention/ops/tree_wy_ref.py` from
`sneha5gsm/vllm`, commit `b073ed6cacfa1cf5ae23111854729e662a72a6d1`.
Those lines are unmodified. The upstream SPDX Apache-2.0 header is preserved;
the full Apache-2.0 license is bundled as `_vendor/TREEWY_LICENSE`.
Upstream file SHA256:
`aa5f0f07824065044ee3b1c98a5a05ff542feace826239e0d38e9cc6292f01eb`.
The owned adapter adds strict topology, shape, gate and accepted-node validation.
Neither the upstream Triton kernels nor its vLLM serving lifecycle are bundled.

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
