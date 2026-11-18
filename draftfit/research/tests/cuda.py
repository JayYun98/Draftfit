"""Run the four tiny research device and async capture cases on an available CUDA GPU."""

import os
import unittest


def load_tests(loader, tests, pattern):
    os.environ.update(
        DRAFTFIT_RESEARCH_TEST_DEVICE="cuda",
        DRAFTFIT_RUN_QUEUE_CUDA="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The research CUDA gate requires a visible CUDA GPU")
    return loader.loadTestsFromNames(
        [
            f"{__package__}.test_research_device",
            f"{__package__}.test_async_capture_joint.JointAsyncCaptureCUDA",
        ]
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
