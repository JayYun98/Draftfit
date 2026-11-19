"""Paper-v2 MoE-SpAc utility/threshold policy on profiled activation traces.

Independent implementation of arXiv:2603.09983v2 Algorithms 1 and Eqs. 6-12.
No expert weights are transferred or executed by this trace reference.
The author's older MIT preview uses a different estimator and fixed threshold.
"""

import json
import math


class UtilityEstimator:
    def __init__(self, experts, gamma=8, max_utility=4, smoothing=0.1):
        if experts < 1 or gamma < 1 or max_utility < 1 or not 0 <= smoothing <= 1:
            raise ValueError("invalid MoE utility parameters")
        self.previous = [0] * experts  # Explicit local initialization convention.
        self.utility = [0] * experts
        self.up = [gamma // 2] * experts
        self.down = [gamma // 2] * experts
        self.maximum, self.smoothing = max_utility, smoothing

    def update(self, frequencies):
        if len(frequencies) != len(self.utility) or any(
            type(f) is not int or f < 0 for f in frequencies
        ):
            raise ValueError(
                "expert frequencies must be nonnegative integers with fixed geometry"
            )
        for i, (current, previous) in enumerate(zip(frequencies, self.previous)):
            delta = current - previous
            # Preserve upward-first precedence, including zero thresholds.
            if delta >= self.up[i]:
                self.utility[i] = min(self.maximum, self.utility[i] + 1)
            elif -delta >= self.down[i]:
                self.utility[i] = max(0, self.utility[i] - 1)
            if delta > 0:
                self.up[i] = math.floor(
                    (1 - self.smoothing) * self.up[i] + self.smoothing * delta
                )
            elif delta < 0:
                self.down[i] = math.floor(
                    (1 - self.smoothing) * self.down[i] - self.smoothing * delta
                )
        self.previous = list(frequencies)
        return {
            "utility": list(self.utility),
            "up": list(self.up),
            "down": list(self.down),
        }


def choose_threshold(utility, frequencies, resident, ratios, free_bytes, options):
    if (
        any(type(i) is not int or not 0 <= i < len(utility) for i in resident)
        or len(resident) != len(set(resident))
        or type(free_bytes) is not int
        or free_bytes < 0
    ):
        raise ValueError("invalid resident IDs or free-memory snapshot")
    if len(ratios) != options.max_utility:
        raise ValueError(
            "one measured/modelled ratio pair is required for every threshold"
        )
    activations = sum(frequencies)
    if activations == 0 or activations % options.gamma:
        raise ValueError(
            "frequencies must describe a full gamma-token, fixed-top-k window"
        )
    k, b = activations // options.gamma, sum(f > 0 for f in frequencies)
    candidates = []
    for tau, ratio in enumerate(ratios, 1):
        rc, rg = ratio["cpu_ratio"], ratio["gpu_ratio"]
        if not all(
            isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1
            for x in (rc, rg)
        ):
            raise ValueError("allocation ratios must be finite fractions")
        hot = [i for i, u in enumerate(utility) if u >= tau]
        transfers = sorted(set(hot) - set(resident))
        cpu = rc * options.gamma * k * options.cpu_unit_ms
        gpu = rg * b * options.gpu_unit_ms
        total = max(cpu, gpu)
        feasible = (
            len(transfers) * options.io_unit_ms
            <= total + options.gamma * options.draft_unit_ms / options.target_layers
            and len(transfers) * options.expert_bytes <= free_bytes
        )
        candidates.append(
            {
                "threshold": tau,
                "cpu_ms": cpu,
                "gpu_ms": gpu,
                "modeled_layer_ms": total,
                "imbalance_ms": abs(cpu - gpu),
                "prefetch_expert_ids": transfers,
                "gpu_expert_ids": hot,
                "feasible": feasible,
            }
        )
    feasible = [c for c in candidates if c["feasible"]]
    selected = (
        min(feasible, key=lambda c: (c["imbalance_ms"], c["threshold"]))
        if feasible
        else None
    )
    return {
        "selected": selected,
        "candidates": candidates,
        "cpu_fallback": selected is None,
    }


def run_moe_policy(config):
    layers, answers = {}, []
    opts = config.policy
    with open(config.data.input_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= config.data.max_samples:
                break
            row = json.loads(line)
            layer, frequencies = row["layer"], row["frequencies"]
            if type(layer) is not int or not 0 <= layer < opts.target_layers:
                raise ValueError("layer ID is outside the target")
            if any(type(f) is not int or not 0 <= f <= opts.gamma for f in frequencies):
                raise ValueError(
                    "expert frequencies must count at most gamma token activations"
                )
            if layer not in layers:
                layers[layer] = UtilityEstimator(
                    len(frequencies), opts.gamma, opts.max_utility, opts.smoothing
                )
            state = layers[layer].update(frequencies)
            decision = choose_threshold(
                state["utility"],
                frequencies,
                row["resident_expert_ids"],
                row["ratios"],
                row["free_bytes"],
                opts,
            )
            answers.append(
                {"id": row.get("id", index), "layer": layer, **state, **decision}
            )
    if not answers:
        raise ValueError("MoE policy input is empty")
    return {
        "answers": answers,
        "implementation": "paper_v2_profiled_trace_policy_reference",
        "initial_frequency_convention": "zero",
        "threshold_tie_break": "lowest_threshold",
        "expert_execution_performed": False,
        "production_engine_speed": False,
    }
