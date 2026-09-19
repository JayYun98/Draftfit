"""Bounded, network-offline CPU gate: python -m tests.public_cpu."""

import os
import unittest

# Set before importing any model/tokenizer/test modules. No GPU discovery suite.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"


def load_tests(loader, tests, pattern):
    modules = [
        "tests.test_namespace_compat",
        "tests.test_target_project",
        "tests.test_target_spec",
        "tests.test_target_inspector",
        "tests.test_runtime.test_model_loading",
        "tests.test_runtime.test_target_revision",
        "tests.test_data.test_hf_parser",
        "tests.test_packaging.test_assets",
        "tests.test_runtime.test_package_architecture",
        "tests.test_runtime.test_cli_config_build",
        "tests.test_runtime.test_launch_plan",
        "tests.test_runtime.test_checkpoint_resume",
        "tests.test_runtime.test_export",
        "tests.test_runtime.test_disaggregated_model_loading",
        "tests.test_runtime.test_model_assembly_utils",
        "tests.test_runtime.test_fsdp2_backend",
        "tests.test_scripts.test_ling_replay_validator",
        "tests.test_packaging.test_release_workflow",
        "tests.test_data.test_prepare",
        "tests.test_data.test_personal_workflow",
        "tests.test_benchmarks.test_sglang_benchmark",
        "tests.test_benchmarks.test_compare",
        "tests.test_utils.test_dflash_mask",
        "tests.test_utils.test_dflash_losses",
        "tests.test_dflash2_integration",
        "tests.test_runtime.test_export_checkpoint_security",
        "tests.test_offline_capture.test_transformers",
        "tests.test_offline_capture.test_vllm",
        "tests.test_teacher_training",
        "tests.test_scripts.test_prepare_hidden_states",
        "tests.test_capture_manifest",
        "tests.test_runtime.test_teacher_server",
        "tests.test_runtime.test_owned_teacher_stream",
        "tests.test_runtime.test_feature_dataloader",
    ]
    suite = loader.loadTestsFromNames(modules)
    suite.addTests(loader.discover("tests/test_config", top_level_dir="."))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
