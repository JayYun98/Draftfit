"""Run the source checkout's offline research CPU tests."""

import os
import unittest
from pathlib import Path


def load_tests(loader, tests, pattern):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        SPECFORGE_DEVICE="cpu",
        DRAFTFIT_RESEARCH_TEST_DEVICE="cpu",
        DRAFTFIT_RESEARCH_TEST_DTYPE="float32",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    modules = sorted(Path(__file__).parent.glob("test_*.py"))
    if not modules:
        raise RuntimeError("No research CPU test modules found")
    names = [f"{__package__}.{path.stem}" for path in modules]
    # The joint capture module also owns an opt-in CUDA case for the same CLI.
    joint = f"{__package__}.test_async_capture_joint"
    names[names.index(joint)] = f"{joint}.JointAsyncCaptureTests"
    return loader.loadTestsFromNames(names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
