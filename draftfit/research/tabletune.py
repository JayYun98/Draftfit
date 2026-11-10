"""Bounded offline tables from the pinned TableTune numerical core.

Profiles use prefix-survival probabilities, not conditional acceptance rates.
No profiling, controller, serving engine, or GPU work is performed here.
"""

import glob
import hashlib
import json
import math
from pathlib import Path

from ._vendor import tabletune as core

SOURCE_REVISION = "38170878d901cb06f696a122a0b4140a3d365f02"


def experiment_metadata(config):
    return {
        "implementation": "pinned_author_offline_table",
        "source_revision": SOURCE_REVISION,
        "source_license": "Apache-2.0",
        "profile_kind": config.profile_kind,
        "limits": {
            "max_batch_size": config.table.max_batch_size,
            "cost_extrapolation": False,
            "live_controller": "not-run",
            "engine_parity": "not-run",
            "throughput": "not-run",
        },
        "serving_backend": False,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_object(path):
    try:
        content = Path(path).read_bytes()
        data = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data, hashlib.sha256(content).hexdigest()
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"bad profile {path}: {error}") from error


def _load_costs(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"no cost files match {pattern!r}")
    costs, hashes = {}, {}
    for path in files:
        data, digest = _read_object(path)
        try:
            rows = data["rows"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("rows must be a nonempty list")
            if any(
                not isinstance(row, dict) or type(row.get("k")) is not int
                for row in rows
            ):
                raise ValueError("every row must have an integer K")
            ks = {row["k"] for row in rows}
            if len(ks) != 1:
                raise ValueError("rows must describe exactly one K")
            k = rows[0]["k"]
            rows = sorted(rows, key=lambda row: row["batch_size"])
            entry = (
                [row["batch_size"] for row in rows],
                [row["step_ms"] for row in rows],
            )
            core.validate_costs({k: entry})
            if k in costs:
                raise ValueError(f"duplicate cost profiles for K={k}")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"bad cost file {path}: {error}") from error
        costs[k] = entry
        hashes[k] = digest
    return costs, hashes


def _load_acceptance(path, cost_ks):
    data, digest = _read_object(path)
    try:
        accept = data["accept_by_pos"]
        input_rate = data.get("proposal_rate", 1.0)
        core.validate_acceptance(accept, input_rate)
        if any(right > left for left, right in zip(accept, accept[1:])):
            raise ValueError("accept_by_pos must be non-increasing prefix survival")
        unsupported = sorted(k for k in cost_ks if k > len(accept))
        if unsupported:
            raise ValueError(f"cost K values lack acceptance support: {unsupported}")

        rate = 1.0 if input_rate >= core.PROPOSAL_RATE_FULL else float(input_rate)
        tps = core.tokens_per_step(accept, proposal_rate=rate)
        stored = data.get("tokens_per_step_by_k")
        source = "prefix_survival"
        if "tokens_per_step_by_k" in data:
            if not isinstance(stored, dict) or not stored:
                raise ValueError("tokens_per_step_by_k must be a nonempty object")
            rounded = {}
            full = core.tokens_per_step(accept)
            for key, value in stored.items():
                k = int(key)
                if str(k) != key or k not in full:
                    raise ValueError(f"unsupported tokens_per_step_by_k key {key!r}")
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 1 <= value <= k + 1
                    or abs(value - full[k]) > 0.000500000001
                ):
                    raise ValueError(
                        "tokens_per_step_by_k must match rounded prefix sums"
                    )
                rounded[k] = float(value)
            if not cost_ks <= rounded.keys():
                raise ValueError("tokens_per_step_by_k lacks cost K coverage")
            ordered = [rounded[k] for k in sorted(rounded)]
            if any(right < left for left, right in zip(ordered, ordered[1:])):
                raise ValueError("tokens_per_step_by_k must be non-decreasing")
            # Upstream uses stored rounding only for full-rate drafters.
            if rate == 1.0:
                tps, source = rounded, "stored_rounded"
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"bad acceptance file {path}: {error}") from error
    return {k: tps[k] for k in sorted(cost_ks)}, rate, input_rate, source, digest


def run_tabletune(config):
    options = config.table
    costs, cost_hashes = _load_costs(options.costs_pattern)
    if 0 not in costs:
        raise ValueError("cost profiles require a K=0 baseline")
    tps, rate, input_rate, source, acceptance_hash = _load_acceptance(
        config.data.input_path, costs.keys()
    )
    lower = max(xs[0] for xs, _ in costs.values())
    upper = min(xs[-1] for xs, _ in costs.values())
    if lower > upper:
        raise ValueError("cost profiles have no overlapping batch coverage")
    if lower > 1 or upper < options.max_batch_size:
        raise ValueError(
            f"every cost profile must cover batches 1..{options.max_batch_size}; "
            f"common coverage is {lower}..{upper}; extrapolation is disabled"
        )
    try:
        goodput = core.goodput_table(costs, tps, options.max_batch_size)
    except ArithmeticError as error:
        raise ValueError("cost profiles produce invalid predicted goodput") from error
    if any(
        not math.isfinite(g) or g <= 0 for row in goodput.values() for g in row.values()
    ):
        raise ValueError(
            "cost profiles produce non-finite or nonpositive predicted goodput"
        )
    raw, best = core.build_schedule(
        goodput, options.eps, options.min_width_frac, options.merge_tol
    )
    return {
        **experiment_metadata(config),
        "schedule": core.compress(best),
        "raw_schedule": core.compress(raw),
        "predicted_goodput_tokens_per_second": goodput,
        "tokens_per_step_by_k": tps,
        "tokens_per_step_source": source,
        "proposal_rate": rate,
        "proposal_rate_input": input_rate,
        "cost_coverage_by_k": {k: [xs[0], xs[-1]] for k, (xs, _) in costs.items()},
        "profiles": {
            "acceptance_sha256": acceptance_hash,
            "costs_sha256_by_k": cost_hashes,
        },
    }
