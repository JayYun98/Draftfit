# Draftfit

[English](README.md) | [한국어](README.ko.md)

Draftfit trains draft models for speculative decoding.
Use JSON configuration files to select the target model, draft method, and training settings.

The target model remains fixed during training.
The draft model proposes tokens.
The target model checks the proposals.

Source version **0.2.0** · [Changelog](CHANGELOG.md) · [Plans](plans/README.md)

## Quick start

Install the [required environment](docs/ENVIRONMENT.md).
This online example uses GPU 0 for target capture and GPU 1 for draft training.
`/path/to/target` is the local target model directory.

1. Prepare the training data and a separate holdout set.

   ```sh
   draftfit data prepare --input ./my-conversations.jsonl \
     --output ./data/train.jsonl --split-eval --eval-output ./data/holdout.jsonl
   ```

2. Create a DSpark training configuration.

   ```sh
   draftfit target prepare /path/to/target --local-only \
     --strategy dspark --train-data ./data/train.jsonl --output-dir ./my-draft
   ```

3. Check the training plan.

   ```sh
   draftfit train -c ./my-draft/train.json --plan
   ```

4. Start training.

   ```sh
   draftfit train -c ./my-draft/train.json
   ```

Use the [complete workflow](docs/PUBLIC_WORKFLOW.md) for data formats, saved features, fine-tuning, recovery, and export.

## Draft methods

All six methods support online target capture.
Five methods also accept saved target features.

| Method | Saved features | Requirement |
| --- | --- | --- |
| DSpark | Yes | Auxiliary features and final target hidden states |
| DFlash | Yes | Compatible hidden features and a mask token |
| DFlash2 | Yes | Full vocabulary, convolution training, and selector training |
| EAGLE3 | Yes | One draft layer and vocabulary mapping for online training |
| PEagle | No | Flex attention and the required feature layout |
| Domino | Yes | A compatible draft configuration |

Select compatible target and draft configurations from the [model recipes](examples/configs/README.md).

## Joint training and unattended runs

Joint training shares one backbone and target features across compatible DSpark, DFlash, and DFlash2 models.
The trainer exports each method separately.
Use the [family guide](docs/DFLASH_FAMILY.md) to configure the methods and their initial weights.
Use the [joint training guide](docs/JOINT_ONLINE.md) to run the manifest.

For automatic runs, configure the [unattended controller](docs/UNATTENDED_NATIVE.md).
Set run limits and training phases before the run.
The controller selects only from the phases that you define.

## Backends

PyTorch trains all draft models.
Select a target capture backend from the [support matrix](docs/PUBLIC_SUPPORT.md).
Validation applies to specific model and runtime configurations.
Qwen3-0.6B vLLM FP32 checks passed.
Its BF16 parity check failed.
Capture tests do not establish exported draft correctness in a serving engine.

## Example models

See the [model recipes](examples/configs/README.md) and [target specifications](configs/target_specs/) for examples.
A configuration alone does not establish target or backend support.
Check the [model validation records](docs/MODEL_VALIDATION.md) before you select a model.

## Research

Research implementations use separate recipes and validation limits.
Use the [research guide](docs/research/README.md) for those methods and experiment results.
These implementations are not production backends.

## Documentation

| Task | Guide |
| --- | --- |
| Install a runtime | [Environment](docs/ENVIRONMENT.md) |
| Prepare, train, resume, and export | [Complete workflow](docs/PUBLIC_WORKFLOW.md) |
| Add a target or runtime | [Target extension](docs/TARGET_EXTENSION.md) · [Runtime profiles](docs/RUNTIME_PROFILES.md) |
| Compare inference speed | [Workload evaluation](docs/PERFORMANCE_GATE.md) |
| Plan training experiments | [Calibration](docs/CALIBRATION.md) · [Training search](docs/TRAINING_SEARCH.md) |
| Inspect jobs and results | [Web workspace](docs/WEB_WORKSPACE.md) · [Monitoring](docs/JOINT_MONITORING.md) |
| Reproduce validation | [Source tools](tests/VALIDATION.md) |

See the [documentation index](docs/README.md) for all guides.
Watch the [workflow overview](https://github.com/JayYun98/Draftfit/releases/download/v0.2.0/draftfit-system-demo-en.mp4) for an illustration of the process.
The video does not show measured performance.

## Acknowledgments

Draftfit maintains a training runtime derived from [SpecForge](https://github.com/sgl-project/SpecForge).
It includes attributed adaptations from [TorchSpec](https://github.com/lightseekorg/TorchSpec), [AngelSpec](https://github.com/Tencent/AngelSpec), and DSpark-related projects.

See [code ownership](docs/CODE_OWNERSHIP.md), [third-party notices](THIRDPARTY_NOTICES.md), and the [license](LICENSE).
