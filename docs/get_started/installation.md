# Install Draftfit

Use the maintained [environment guide](../ENVIRONMENT.md) for the exact uv
commands, Python version and CPU dependency lock:

```sh
git clone https://github.com/JayYun98/Draftfit.git
cd Draftfit
```

Create a fresh environment when migrating from an earlier distribution. Do not
co-install Draftfit with the old platform or upstream SpecForge distributions:
their compatibility packages can own overlapping files. Follow the migration
instructions in the environment guide instead of installing upstream
`specforge` as a substitute.

## Choose an execution profile

CPU preparation and contributor checks do not certify GPU training or serving.
For GPU work, use the [runtime profiles](../RUNTIME_PROFILES.md) and preserve the
backend's matched PyTorch, CUDA and capture dependencies. Do not apply the CPU
lock to a custom GPU runtime.

CUDA, ROCm and Ascend are not interchangeable installation targets. Consult
the [current support boundaries](../PUBLIC_SUPPORT.md) before choosing a
backend or interpreting an inherited tutorial as a tested recipe.

After preparing the environment, follow the
[public workflow](../PUBLIC_WORKFLOW.md) for data preparation, training and export.
