"""Paired reference timing and attempted-proposal position diagnostics.

Inspired by arXiv:2610.30644v1 and 2610.31479v1. Timings compare this exact
full-prefix PyTorch runtime on identical greedy workloads; they are not MLX
measurements or production-engine acceleration. Rejected suffixes are censored.
"""

from collections import defaultdict
from statistics import median

import torch

from .memory import Proposal
from .reference_decode import generate


def position_bins(records, width):
    bins = defaultdict(list)
    for record in records:
        bins[record["output_position"] // width].append(record)
    return [
        {
            "start_position": i * width,
            "attempted_proposals": len(rows),
            "observed_acceptance": sum(r["accepted"] for r in rows) / len(rows),
            "mean_distribution_overlap": sum(r["distribution_overlap"] for r in rows)
            / len(rows),
            "argmax_agreement": sum(r["argmax_agreement"] for r in rows) / len(rows),
        }
        for i, rows in sorted(bins.items())
    ]


def profile_generate(target, ids, factory, generation, diagnostics, generator):
    if diagnostics.compare_ar and generation.temperature != 0:
        raise ValueError(
            "paired token-workload AR comparison requires greedy generation"
        )
    state = generator.get_state()
    outputs, timings, baselines = [], [], []
    for index in range(diagnostics.warmup + diagnostics.repetitions):
        proposer = factory()
        rng = torch.Generator(device="cpu").set_state(state)
        output, stats = generate(
            target,
            ids,
            proposer,
            **generation.model_dump(),
            generator=rng,
            record_acceptance=diagnostics.record_acceptance,
        )
        if diagnostics.compare_ar:
            baseline, ar_stats = generate(
                target,
                ids,
                lambda *_: Proposal(),
                **generation.model_dump(),
                generator=torch.Generator(device="cpu").set_state(state),
            )
            if not torch.equal(output, baseline):
                raise ValueError(
                    "paired greedy reference output differs from autoregressive target"
                )
        if index >= diagnostics.warmup:
            outputs.append(output)
            timings.append(stats["elapsed_seconds"])
            if diagnostics.compare_ar:
                baselines.append(ar_stats["elapsed_seconds"])
    generator.set_state(rng.get_state())
    if any(not torch.equal(outputs[0], other) for other in outputs[1:]):
        raise ValueError(
            "repeat runs with identical RNG state produced different token workloads"
        )
    stats["measurement"] = {
        "runtime": "pytorch_full_prefix_reference",
        "device": str(ids.device),
        "torch_version": torch.__version__,
        "dtype": str(next(target.parameters()).dtype),
        "warmup": diagnostics.warmup,
        "repetitions": diagnostics.repetitions,
        "generation_seconds": timings,
        "median_generation_seconds": median(timings),
        "production_engine_speed": False,
    }
    if baselines:
        stats["measurement"]["paired_ar"] = {
            "token_ids_equal": True,
            "generation_seconds": baselines,
            "median_generation_seconds": median(baselines),
            "reference_speed_ratio": median(baselines) / max(median(timings), 1e-12),
        }
    if diagnostics.record_acceptance:
        stats["position_bins"] = position_bins(
            stats["token_acceptance"], diagnostics.position_bin_size
        )
        stats["position_population"] = (
            "attempted_proposals_only_rejected_suffixes_censored"
        )
    return outputs[-1], stats, proposer
