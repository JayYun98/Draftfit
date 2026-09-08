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
        "tests.test_target_project", "tests.test_target_spec", "tests.test_target_inspector",
        "tests.test_runtime.test_model_loading", "tests.test_runtime.test_target_revision",
        "tests.test_data.test_hf_parser", "tests.test_packaging.test_assets",
        "tests.test_runtime.test_package_architecture", "tests.test_runtime.test_cli_config_build",
        "tests.test_runtime.test_launch_plan", "tests.test_runtime.test_checkpoint_resume",
        "tests.test_runtime.test_export", "tests.test_runtime.test_disaggregated_model_loading",
        "tests.test_runtime.test_model_assembly_utils",
        "tests.test_runtime.test_fsdp2_backend",
        "tests.test_scripts.test_ling_replay_validator",
        "tests.test_packaging.test_release_workflow",
    ]
    suite = loader.loadTestsFromNames(modules)
    suite.addTests(loader.discover("tests/test_config", top_level_dir="."))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
