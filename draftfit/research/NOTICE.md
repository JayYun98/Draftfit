# Research source notices

Native adaptations use the following upstream sources. Transformers and EAGLE
fragments retain their existing Apache-2.0 notices and repository licenses.

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
