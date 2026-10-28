"""Dependency-light provenance and explicit implementation boundaries."""

SOURCES = {
    "alr_ira": ("js-lee-AI/ALR-IRA", "fa473642d48ea2667675d706ca8f16a74111183e"),
    "cast": ("js-lee-AI/CAST", "2eed73f38f9210f1d5ddf01de8059cf3ffb43c19"),
    "osprey": ("LeanModels/Osprey", "f62cd48844b03f3cb929543da25c15756e1f1095"),
    "redline": ("js-lee-AI/Redline", "d78ede9b035420f3c8c8c63c185ed55be5ea038f"),
}


# Unavailable author code is not replaced by a guessed repository/revision.
LATEST = [
    ("ngramrl", "2610.32881v1", "native_synchronous_reference", None, None),
    ("specmem", "2610.31840v1", "native_memory_reference", None, None),
    ("valuspec", "2610.31338v1", "cast_conditional_branch_filter", None, None),
    ("rift", "2610.31098v1", "native_kv_recompute_reference", None, None),
    (
        "treewy",
        "2608.20961v2",
        "author_numerical_reference",
        "sneha5gsm/vllm",
        "b073ed6cacfa1cf5ae23111854729e662a72a6d1",
    ),
    (
        "arr",
        "2610.28294v1",
        "independent_stationary_reference",
        "AhsanBilal7/adaptive_refine",
        "a704327e0759a303dd9563e9ad9cba347618eb48",
    ),
    (
        "moe_spac",
        "2603.09983v2",
        "paper_v2_trace_policy",
        "lshAlgorithm/MoE-SpAc",
        "77bb9cf1e3593b1e1ea0ec543b3e338e7edd1129",
    ),
    (
        "query_layout",
        "2610.03228v4",
        "tensor_diagnostic_source_unavailable",
        "leecdiang/HeteroSpec",
        None,
    ),
    (
        "acceptance_position",
        "2610.30644v1",
        "attempted_proposal_diagnostics",
        None,
        None,
    ),
    (
        "apple_silicon_measurement",
        "2610.31479v1",
        "paired_reference_timing_only",
        None,
        None,
    ),
    (
        "graph_mtp",
        "2610.31439v1",
        "not_integrated_separate_scratch_learning_task",
        None,
        None,
    ),
    (
        "diffusion_two_tradeoffs",
        "2610.31819v1",
        "not_integrated_requires_multistep_engine",
        None,
        None,
    ),
]


def source_catalog():
    return [
        {
            "project": name,
            "url": "https://github.com/" + repo,
            "revision": revision,
            "integration": "native",
        }
        for name, (repo, revision) in SOURCES.items()
    ] + [
        {
            "project": name,
            "paper": "https://arxiv.org/abs/" + paper,
            "url": "https://github.com/" + repo if repo else None,
            "revision": revision,
            "integration": integration,
            "paper_revision": paper,
            "production_engine_validated": False,
        }
        for name, paper, integration, repo, revision in LATEST
    ]
