# Research experiment drivers

These scripts are development tools for research references.
They are not production entry points.
Run them from a source checkout in a compatible environment.

| Driver | Purpose | Runtime |
| --- | --- | --- |
| [run_latest_smoke.py](run_latest_smoke.py) | Execute all eight CLI fixtures with tiny local models | CPU |
| [run_latest_pretrained_gpu.py](run_latest_pretrained_gpu.py) | Check bounded pretrained correctness and state behavior | A provisioned CUDA runtime |
| [run_latest_effects.py](run_latest_effects.py) | Compare conditional memory and RIFT effects; check ARR answers | A provisioned CUDA runtime |

Run commands from the repository root.
Use a new output directory.
Check resource limits before a GPU run.
See the [research protocol](../../docs/research/latest-speculative-methods.md) for commands and retained evidence.
