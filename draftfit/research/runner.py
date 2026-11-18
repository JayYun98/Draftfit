"""Run native research preparation, reference decoding, and calibration."""

import json
import shutil
import tempfile
from pathlib import Path

IMPLEMENTATIONS = {
    "cast": "native_reference",
    "ngramrl": "native_synchronous_reference",
    "specmem": "native_memory_reference",
    "rift": "native_kv_recompute_reference",
    "arr": "independent_stationary_reference",
    "treewy": "author_numerical_reference",
    "moe_spac": "policy_trace_reference",
    "query_layout": "tensor_diagnostic",
}


def plan(config, recipe_path=None):
    """Describe the selected work without loading models or writing output."""
    return {
        "project": config.method,
        "implementation": IMPLEMENTATIONS.get(config.method, "native"),
        "config": config.model_dump(),
        "execution": "not-run",
        "serving_compatibility": "not_validated",
    }


def run(config, recipe_path=None):
    if config.method == "alr_ira":
        from .papers.alr_ira.method import prepare

        return prepare(config)
    if config.method == "osprey":
        from .papers.osprey.recipe import run_osprey

        return run_osprey(config)
    destination = Path(config.output_dir)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if config.method not in {
        "cast",
        "redline",
        "ngramrl",
        "specmem",
        "rift",
        "treewy",
        "arr",
        "moe_spac",
        "query_layout",
    }:
        raise ValueError(f"unsupported research method: {config.method}")
    if config.method in {"ngramrl", "specmem"}:
        from .memory_runner import run_memory

        result = run_memory(config)
    elif config.method == "rift":
        from .papers.rift.method import run_rift

        result = run_rift(config)
    elif config.method == "treewy":
        from .papers.treewy.method import run_treewy

        result = run_treewy(config)
    elif config.method == "arr":
        from .papers.arr.method import run_arr

        result = run_arr(config)
    elif config.method == "moe_spac":
        from .papers.moe_spac.method import run_moe_policy

        result = run_moe_policy(config)
    elif config.method == "query_layout":
        from .papers.query_layout.method import run_query_layout

        result = run_query_layout(config)
    elif config.method == "cast":
        from .papers.cast.recipe import run_cast

        result = run_cast(config)
    else:
        from .papers.redline.recipe import run_redline

        result = run_redline(config)
    memory_state = result.pop("memory_state", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix=".draftfit-research-", dir=destination.parent)
    )
    try:
        (scratch / "recipe.json").write_text(config.model_dump_json(indent=2) + "\n")
        (scratch / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        if memory_state is not None:
            (scratch / "memory.json").write_text(
                json.dumps(memory_state, indent=2, allow_nan=False) + "\n"
            )
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return result
