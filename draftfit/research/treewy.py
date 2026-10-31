"""Validated input/commit boundary for the pinned author's TreeWY reference.

This is a numerical Gated DeltaNet state experiment, not a serving backend.
Convolution history and attention KV are not updated by this component.
"""

import json

import torch

from ._vendor.treewy import tree_wy_gated_delta_rule


def solve_tree(
    q,
    k,
    v,
    g,
    beta,
    parent,
    initial_state,
    *,
    accepted_node=None,
    max_nodes=64,
    use_qk_l2norm=True,
    scale=None,
):
    if (
        parent.ndim != 1
        or parent.dtype != torch.long
        or not 1 <= len(parent) <= max_nodes
    ):
        raise ValueError("TreeWY requires a bounded integer parent vector")
    parents = parent.tolist()
    if parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(parents[1:], 1)):
        raise ValueError(
            "TreeWY requires one connected root and parent-before-child ordering"
        )
    if q.ndim != 3 or q.shape != k.shape or v.ndim != 3:
        raise ValueError(
            "Q/K must be [nodes, heads, key dim]; V [nodes, heads, value dim]"
        )
    n, h, kd = k.shape
    if (
        n != len(parent)
        or min(n, h, kd) < 1
        or v.shape[:2] != (n, h)
        or v.shape[-1] < 1
        or g.shape != (n, h)
        or beta.shape != (n, h)
        or initial_state.shape != (h, v.shape[-1], kd)
    ):
        raise ValueError("inconsistent TreeWY tensor/state shapes")
    tensors = (q, k, v, g, beta, initial_state)
    if (
        any(t.device != q.device or not torch.isfinite(t).all() for t in tensors)
        or parent.device != q.device
    ):
        raise ValueError("TreeWY tensors must be finite and on the same device")
    if (g > 0).any() or (beta < 0).any() or (beta > 1).any():
        raise ValueError("TreeWY log decays must be nonpositive and gates in [0, 1]")
    if accepted_node is not None and (
        type(accepted_node) is not int or not 0 <= accepted_node < n
    ):
        raise ValueError(
            "accepted_node must identify a forwarded node, or be null for no commit"
        )
    result = tree_wy_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        parent,
        initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
    )
    state = (
        initial_state.float().clone()
        if accepted_node is None
        else result.reconstruct_state(accepted_node)
    )
    if not torch.isfinite(result.output).all() or not torch.isfinite(state).all():
        raise ValueError("TreeWY numerical overflow")
    return result.output, state


def run_treewy(config):
    answers = []
    with (
        torch.inference_mode(),
        open(config.data.input_path, encoding="utf-8") as handle,
    ):
        for index, line in enumerate(handle):
            if index >= config.data.max_samples:
                break
            row = json.loads(line)
            parents = row["parent"]
            if any(type(p) is not int for p in parents):
                raise ValueError("TreeWY parent IDs must be integers")
            tensors = {
                name: torch.tensor(
                    row[name], dtype=torch.float32, device=config.runtime.device
                )
                for name in ("q", "k", "v", "g", "beta", "initial_state")
            }
            output, state = solve_tree(
                **tensors,
                parent=torch.tensor(
                    parents, dtype=torch.long, device=config.runtime.device
                ),
                accepted_node=row.get("accepted_node"),
                **config.tree.model_dump(),
            )
            answers.append(
                {
                    "id": row.get("id", index),
                    "output": output.tolist(),
                    "committed_gdn_state": state.tolist(),
                }
            )
    if not answers:
        raise ValueError("TreeWY input is empty")
    return {
        "answers": answers,
        "implementation": "pinned_author_numerical_reference",
        "source_revision": "b073ed6cacfa1cf5ae23111854729e662a72a6d1",
        "serving_backend": False,
        "commits_convolution_or_attention_kv": False,
    }
