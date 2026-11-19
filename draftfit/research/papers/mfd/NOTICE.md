# Multi-Future Drafting

[`method.py`](method.py) implements the best-mode prefix ranking in Section 4.3 of
[MFD v1](https://arxiv.org/abs/2611.13017v1). Its mathematical contract was
cross-checked against `inference/hf/ddtree.py::build_mixture_tree` from
[`hhhhhhy5/multi-future-drafting`](https://github.com/hhhhhhy5/multi-future-drafting),
commit `95040343de00f15aef8c6350a578c35e0cd53475`.
The local selector is independently written. It does not copy the executable
HF implementation, attention helpers, compiled kernels, or generation loop.

The repository root license is Apache-2.0, retained in
[MFD_ROOT_LICENSE](MFD_ROOT_LICENSE). The HF subtree has a separate MIT license,
Copyright (c) 2026 Liran Ringel, retained in [MFD_HF_LICENSE](MFD_HF_LICENSE).
The root license does not replace that subtree notice.

Reviewed source SHA256:
`51934e4af7b381a86619b1cca76486da394776fefdeaf89f475946e5689dcad1`.
Root license SHA256:
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
HF license SHA256:
`6355c7a7042a1f75a011d616446386d19fac37ea3f03560425f7f225ba9ae1d3`.

The CPU reference ranks supplied mode probabilities. It does not implement
mode training, target verification, GPU tree construction, or serving.
