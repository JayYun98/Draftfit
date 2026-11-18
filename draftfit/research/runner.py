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
    "tabletune": "pinned_author_offline_table",
    "d3d_corrected_kl": "adapted_author_corrected_kl_component",
    "dforktree_scoring": "cpu_probability_scoring_reference",
    "mfd_prefix_tree": "cpu_reference",
    "real_loss": "adapted_author_real_branch_component",
}


def plan(config, recipe_path=None):
    """Describe the selected work without loading models or writing output."""
    result = {
        "project": config.method,
        "implementation": IMPLEMENTATIONS.get(config.method, "native"),
        "config": config.model_dump(),
        "execution": "not-run",
        "serving_compatibility": "not_validated",
    }
    if config.method == "tabletune":
        from .papers.tabletune.method import experiment_metadata

        result.update(experiment_metadata(config))
    elif config.method == "d3d_corrected_kl":
        result["scope"] = "corrected_forward_kl_component_only"
        result["model_training"] = "not_implemented"
        result["pruning"] = "not_implemented"
        result["asynchronous_serving"] = "not_implemented"
    elif config.method == "mfd_prefix_tree":
        from .catalog import MFD_PAPER_REVISION, MFD_SOURCE_REVISION

        result.update(
            source_revision=MFD_SOURCE_REVISION,
            paper_revision=MFD_PAPER_REVISION,
            model_execution_performed=False,
            serving_validation="not-run",
        )
    elif config.method == "real_loss":
        result.update(
            scope="target_sampled_real_branch_component_only",
            ce_target="target_logits_argmax",
            ce_position_decay=False,
            author_wrapper_parity=False,
            sampling_provenance_verified=False,
            model_training="not_implemented",
            draft_sampled_d_real="not_implemented",
            gpu_parity="not-run",
            real_model_quality="not-run",
            throughput="not-run",
        )
    return result


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
    elif config.method == "tabletune":
        from .papers.tabletune.method import run_tabletune

        result = run_tabletune(config)
    elif config.method == "d3d_corrected_kl":
        from .papers.d3d.method import run_d3d_corrected_kl

        result = run_d3d_corrected_kl(config)
    elif config.method == "dforktree_scoring":
        from .papers.dforktree.method import run_dforktree_scoring

        result = run_dforktree_scoring(config)
    elif config.method == "mfd_prefix_tree":
        from .papers.mfd.method import run_mfd

        result = run_mfd(config)
    elif config.method == "real_loss":
        from .papers.real.method import run_real_loss

        result = run_real_loss(config)
    elif config.method == "cast":
        from .papers.cast.recipe import run_cast

        result = run_cast(config)
    elif config.method == "redline":
        from .papers.redline.recipe import run_redline

        result = run_redline(config)
    else:
        raise ValueError(f"unsupported research method: {config.method}")
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
