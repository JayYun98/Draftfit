# SPDX-License-Identifier: Apache-2.0
"""
Tree-WY reference for the gated delta rule (Gated DeltaNet) speculative-tree
verification.  PyTorch, dtype/layout-matched to vLLM's GDN forward
(``chunk_gated_delta_rule``) so it can serve as ground truth for the fused
tree-mask kernel and the reconstruct-on-commit rollback .

This is the torch port of the validated NumPy prototype, generalized to:
  * log-space gating ``g`` (alpha_t = exp(g_t)), as the FLA kernels use;
  * grouped heads (vectorized over H);
  * state layout ``[H, V, K]`` (value-dim x key-dim), matching ``initial_state``;
  * optional L2-norm of q,k and a query ``scale`` (the kernel's knobs).

A **chain** (parent = path) reduces to the ordinary gated delta rule, which is
what ``chunk_gated_delta_rule`` computes; a **tree** (``parent[t] < t``) swaps
the dense causal mask for the strict-ancestor mask.  See README.md /
the WY/UT transform of the gated delta rule for the math.

NOTE: reference clarity over speed.  O(N^2) dense masks; fine for spec trees
(N ~ 16-64) and for testing; the fused Triton kernel is the production path.
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def build_ancestor_masks(parent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict-ancestor and inclusive-ancestor masks from a parent array.

    parent: long tensor [N], parent[t] < t, root = -1.
    Returns (anc_strict, anc_incl), each [N, N] float, with [t, i] = 1 iff
    i is a (strict / inclusive) ancestor of t.  Strictly-lower-triangular by
    construction (parent[t] < t), so the WY system is unit-lower-triangular.
    """
    n = parent.shape[0]
    anc_incl = torch.zeros(n, n, dtype=torch.float32, device=parent.device)
    for t in range(n):
        j = t
        while j != -1:
            anc_incl[t, j] = 1.0
            j = int(parent[j].item())
    anc_strict = anc_incl.clone()
    anc_strict[range(n), range(n)] = 0.0
    return anc_strict, anc_incl


def path_cumulative_decay(g: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    """Inclusive path-cumulative log-decay G_cum[t] = sum_{i <= t} g_i along
    the root->t path.  g: [N, H] (log space).  Returns [N, H]."""
    n, h = g.shape
    gcum = torch.empty_like(g)
    for t in range(n):
        p = int(parent[t].item())
        gcum[t] = g[t] + (gcum[p] if p != -1 else 0.0)
    return gcum


class TreeWYResult:
    """Holds the single-solve outputs so the accepted state can be committed."""

    def __init__(self, output, vtilde, gcum, anc_incl, k, initial_state):
        self.output = output  # [N, H, V]  -- all node outputs
        self.vtilde = vtilde  # [H, N, V]  -- pseudo-values (the cache)
        self._gcum = gcum  # [N, H]
        self._anc_incl = anc_incl  # [N, N]
        self._k = k  # [N, H, K]  (post-l2norm)
        self._s0 = initial_state  # [H, V, K]

    def reconstruct_state(self, accepted_node: int) -> torch.Tensor:
        """Committed state S_a for the accepted node, rebuilt from vtilde:
            S_a = g_a S0 + sum_{i <= a} (g_a/g_i) vtilde_i k_i^T .
        Returns [H, V, K]."""
        a = accepted_node
        gcum, k = self._gcum, self._k
        # decay weights g_a / g_i along a's inclusive ancestor set, else 0
        w = self._anc_incl[a].unsqueeze(0) * torch.exp(
            (gcum[a].unsqueeze(1) - gcum.transpose(0, 1)).clamp(max=0.0)
        )  # [H, N]; clamp: masked (non-ancestor) entries can be >0 -> exp overflow
        kperm = k.permute(1, 0, 2)  # [H, N, K]
        delta = torch.einsum("hi,hiv,hik->hvk", w, self.vtilde, kperm)
        return torch.exp(gcum[a]).view(-1, 1, 1) * self._s0 + delta


def tree_wy_gated_delta_rule(
    q: torch.Tensor,  # [N, H, K]
    k: torch.Tensor,  # [N, H, K]
    v: torch.Tensor,  # [N, H, V]
    g: torch.Tensor,  # [N, H]  (log-space decay; alpha = exp(g))
    beta: torch.Tensor,  # [N, H]
    parent: torch.Tensor,  # [N]     long, parent[t] < t, root = -1
    initial_state: torch.Tensor,  # [H, V, K]
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> TreeWYResult:
    """Single-pass tree-WY verification of one draft tree (H heads).

    Returns a TreeWYResult with all node outputs and the pseudo-values; call
    .reconstruct_state(a) to commit the accepted node's recurrent state.
    """
    n, h, kdim = k.shape
    vdim = v.shape[-1]
    dev, dt = k.device, torch.float32
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    initial_state = initial_state.to(dt)
    if scale is None:
        scale = kdim**-0.5
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    q = q * scale

    anc_strict, anc_incl = build_ancestor_masks(parent)
    gcum = path_cumulative_decay(g, parent)  # [N, H]
    gcumT = gcum.transpose(0, 1)  # [H, N]

    # pairwise decay ratio g_t/g_i  ->  [H, N, N]
    dec = torch.exp((gcumT.unsqueeze(2) - gcumT.unsqueeze(1)).clamp(max=0.0))
    KK = torch.einsum("nhk,mhk->hnm", k, k)  # [H, N, N], KK[h,t,i]=k_t.k_i
    G = anc_strict.unsqueeze(0) * dec * KK  # [H, N, N]
    betaT = beta.transpose(0, 1)  # [H, N]
    M = torch.eye(n, device=dev, dtype=dt).unsqueeze(0) + betaT.unsqueeze(2) * G

    Sk = torch.einsum("hvk,nhk->hnv", initial_state, k)  # [H, N, V]  (S0 k_t)
    R = (
        betaT.unsqueeze(2) * v.permute(1, 0, 2)
        - (betaT * torch.exp(gcumT)).unsqueeze(2) * Sk
    )  # [H, N, V]
    vtilde = torch.linalg.solve(M, R)  # [H, N, V]

    base = torch.exp(gcumT).unsqueeze(2) * torch.einsum(
        "hvk,nhk->hnv", initial_state, q
    )  # [H,N,V]
    KQ = torch.einsum("nhk,mhk->hnm", k, q)  # [H, N, N], KQ[h,i,t]=k_i.q_t
    Co = anc_incl.unsqueeze(0) * dec * KQ.transpose(1, 2)  # [H,N,N] -> [h,t,i]
    out = (base + torch.einsum("hti,hiv->htv", Co, vtilde)).permute(1, 0, 2)  # [N,H,V]

    return TreeWYResult(out, vtilde, gcum, anc_incl, k, initial_state)


