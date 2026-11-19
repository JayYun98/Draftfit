"""Phase-scoped PQ/QB tensor diagnostic; no unavailable HeteroSpec runtime import.

arXiv:2610.03228v4. This verifies masks, GQA mapping and output ordering in
floating-point attention only. It cannot validate GGML quantized reductions,
graph caching, native reconciliation or endpoint token parity.
"""

import json

import torch


def compare_layouts(
    q,
    k,
    v,
    mask,
    *,
    role="target",
    phase="continuation",
    phase_mask=3,
    atol=1e-5,
    rtol=1e-4,
):
    phases = {
        ("target", "prefill"): 1,
        ("target", "continuation"): 2,
        ("draft", "prefill"): 4,
        ("draft", "continuation"): 8,
    }
    if (role, phase) not in phases:
        raise ValueError("unknown query-layout role/phase")
    if (
        q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or mask.ndim != 2
        or k.shape[:2] != v.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or mask.shape != (q.shape[0], k.shape[0])
        or min(*q.shape, *k.shape, *v.shape) < 1
    ):
        raise ValueError("invalid query-layout tensor or query-specific mask shapes")
    if (
        any(not torch.isfinite(t).all() for t in (q, k, v))
        or torch.isnan(mask).any()
        or torch.isposinf(mask).any()
    ):
        raise ValueError(
            "query-layout inputs must be finite, with finite or -inf mask entries"
        )
    if q.shape[1] % k.shape[1]:
        raise ValueError("query layouts require integral GQA groups")
    if q.shape[0] > 4:
        return {"admitted": False, "reason": "more_than_four_occupied_queries"}
    selected = bool(phases[(role, phase)] & phase_mask)
    if not selected:
        return {"admitted": False, "reason": "phase_not_selected"}
    groups = q.shape[1] // k.shape[1]
    keys, values = k.float().repeat_interleave(groups, 1), v.float().repeat_interleave(
        groups, 1
    )
    scale = q.shape[-1] ** -0.5
    pq_logits = torch.einsum("qhd,shd->hqs", q.float(), keys) * scale + mask[None]
    pq = torch.einsum("hqs,shv->qhv", pq_logits.softmax(-1), values)
    # Occupied queries become an explicit batch axis; each keeps its own mask.
    qb_logits = (
        torch.matmul(
            q.float().permute(1, 0, 2).unsqueeze(2), keys.permute(1, 2, 0)[:, None]
        )
        * scale
        + mask[None, :, None]
    )
    qb = (
        torch.matmul(qb_logits.softmax(-1), values.permute(1, 0, 2)[:, None])
        .squeeze(2)
        .permute(1, 0, 2)
    )
    if not torch.isfinite(pq).all() or not torch.isfinite(qb).all():
        raise ValueError("every query must attend at least one unmasked key")
    error = float((pq - qb).abs().max())
    if not torch.allclose(pq, qb, atol=atol, rtol=rtol):
        raise ValueError(f"PQ/QB tensor reference mismatch: max_abs_error={error}")
    return {
        "admitted": True,
        "pq_output": pq.tolist(),
        "qb_output": qb.tolist(),
        "max_abs_error": error,
        "input_q_stride": list(q.stride()),
        "mask_stride": list(mask.stride()),
    }


def run_query_layout(config):
    answers = []
    with (
        torch.inference_mode(),
        open(config.data.input_path, encoding="utf-8") as handle,
    ):
        for index, line in enumerate(handle):
            if index >= config.data.max_samples:
                break
            row = json.loads(line)
            tensors = {
                key: torch.tensor(
                    row[key],
                    device=config.runtime.device,
                    dtype=torch.float16 if key in {"k", "v"} else torch.float32,
                )
                for key in ("q", "k", "v", "mask")
            }
            comparison = compare_layouts(
                **tensors,
                role=row.get("role", "target"),
                phase=row.get("phase", "continuation"),
                **config.layout.model_dump(),
            )
            answers.append({"id": row.get("id", index), **comparison})
    if not answers:
        raise ValueError("query-layout input is empty")
    return {
        "answers": answers,
        "implementation": "floating_point_tensor_layout_diagnostic",
        "native_runtime_available": False,
        "endpoint_parity_validated": False,
        "production_engine_speed": False,
    }
