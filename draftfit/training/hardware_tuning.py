"""Bounded empirical tuning. Estimates bound candidates; measurements select them."""

from __future__ import annotations

import math
import time


def candidates(maximum: int, effective: int | None = None) -> list[int]:
    if (
        type(maximum) is not int
        or maximum < 1
        or (effective is not None and (type(effective) is not int or effective < 1))
    ):
        raise ValueError("batch limits must be positive integers")
    values = [2**i for i in range(maximum.bit_length()) if 2**i <= maximum]
    if effective is not None:
        values = [x for x in values if effective % x == 0]
    return values


def choose(measurements: list[dict], memory_fraction: float = 0.9) -> dict:
    if isinstance(memory_fraction, bool) or not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must be between zero and one")
    valid = []
    for row in measurements:
        if row.get("status") != "ok":
            continue
        values = [
            row.get(k)
            for k in ("samples_per_second", "peak_bytes", "memory_budget_bytes")
        ]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        ):
            continue
        if (
            type(row.get("batch_size")) is not int
            or row["batch_size"] < 1
            or values[0] <= 0
        ):
            continue
        budget = row["memory_budget_bytes"]
        if row.get("hardware", {}).get("total_bytes", 0):
            budget = min(budget, row["hardware"]["total_bytes"] * memory_fraction)
        if row["peak_bytes"] <= budget:
            valid.append(row)
    if not valid:
        raise RuntimeError("no measured batch fits the memory/time budget")
    # Prefer the smaller allocation when the difference is under 3%.
    fastest = max(r["samples_per_second"] for r in valid)
    return min(
        (r for r in valid if r["samples_per_second"] >= fastest * 0.97),
        key=lambda r: r["batch_size"],
    )


def cuda_spec(torch, device=None) -> dict:
    device = device or torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    p = torch.cuda.get_device_properties(device)
    return dict(
        name=p.name,
        total_bytes=total,
        free_bytes=free,
        capability=list(torch.cuda.get_device_capability(device)),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def generation_limit(config, spec, max_length, maximum=128):
    """KV estimate only: nonstandard/hybrid caches still require real probes."""
    cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    layers = getattr(cfg, "num_hidden_layers", 1)
    heads = max(
        getattr(cfg, "num_key_value_heads", 1),
        getattr(cfg, "num_global_key_value_heads", 1) or 1,
    )
    dim = max(
        getattr(cfg, "head_dim", 128), getattr(cfg, "global_head_dim", 128) or 128
    )
    kv = 2 * layers * heads * dim * max_length * 2
    available = max(0, int(spec["free_bytes"] * 0.9))
    return max(1, min(maximum, int(available // max(kv * 2, 1))))


def probe_training(
    torch,
    training_loader,
    stages,
    strategies,
    bundles,
    optimizer,
    backward,
    *,
    steps=2,
    deadline=None,
    consume=None,
):
    """Scratch process only. Includes capture, backward and Adam allocations; no saves."""
    if type(steps) is not int or steps < 2:
        raise ValueError("probe needs at least two updates including warmup")
    t = stages[0].config.training
    device = next(bundles[0].draft_model.parameters()).device
    gpu = device.type == "cuda"
    spec = cuda_spec(torch, device) if gpu else {"name": "cpu", "total_bytes": 0}
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    windows, timings = [], []
    initial_reserved = torch.cuda.memory_reserved(device) if gpu else 0
    for epoch in range(t.num_epochs):
        loader, _ = training_loader(epoch)
        start = time.perf_counter()
        for batch in loader:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("training tuning deadline")
            windows.append(batch)
            if len(windows) < t.accumulation_steps:
                continue
            backward(stages, strategies, bundles, windows)
            optimizer.step()
            if consume is not None:
                consume(windows)
            windows.clear()
            if gpu:
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            if len(timings) == steps:
                # First update includes kernel and optimizer warmup.
                seconds = sum(timings[1:]) / len(timings[1:])
                return {
                    "status": "ok",
                    "batch_size": t.batch_size,
                    "accumulation_steps": t.accumulation_steps,
                    "samples_per_second": t.batch_size * t.accumulation_steps / seconds,
                    "step_seconds": seconds,
                    "hardware": spec,
                    "peak_bytes": torch.cuda.max_memory_reserved(device) if gpu else 0,
                    "memory_budget_bytes": (
                        max(0, initial_reserved + int(spec["free_bytes"] * 0.9))
                        if gpu
                        else 0
                    ),
                    "model_spec": {
                        "target": stages[0].config.model.target_model_path,
                        "max_length": stages[0].config.data.max_length,
                        "families": [s.config.training.strategy for s in stages],
                        "dtype": str(next(bundles[0].draft_model.parameters()).dtype),
                        "optimizer_cpu_offload": t.optimizer_cpu_offload,
                    },
                    "probe_steps": steps,
                    "checkpoint_written": False,
                }
            start = time.perf_counter()
    raise RuntimeError("not enough samples for training probe")
