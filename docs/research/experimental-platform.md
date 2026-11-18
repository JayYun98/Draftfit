# Experimental platform integrations

Use these entry points from a source checkout. Each folder links the configuration or command that selects the feature, its implementation, tests, and evidence limits. Platform integrations use their own training, capture or campaign commands; `draftfit research -c` runs the separate [paper recipes](README.md).

This index also includes maintained tools used for experiments. Their support level remains the one documented by the owning subsystem.

PyTorch trains the draft models. A working capture adapter does not establish token parity in a serving engine. See the [public support matrix](../PUBLIC_SUPPORT.md) for maintained configurations and the [environment guide](../ENVIRONMENT.md) for runtime requirements.

## Capture, generation and verification

| Experiment | Validation boundary |
| --- | --- |
| [Prism GGUF joint capture](../../draftfit/research/experiments/prism_gguf/README.md) | Bonsai PQ2_0 only; native parity requires the pinned bridge and target. |
| [MLX offline feature capture](../../draftfit/research/experiments/mlx_capture/README.md) | Local Qwen3.5 text and reviewed Bonsai packs; training stays in PyTorch. |
| [MLX campaign evaluation](../../draftfit/research/experiments/mlx_evaluation/README.md) | External runtime; only complete matched requests with exact tokens qualify. |
| [Native vLLM teacher](../../draftfit/research/experiments/vllm_teacher/README.md) | Dense Llama/Qwen2/Qwen3, TP1/PP1; Qwen3-0.6B BF16 parity failed. |
| [Quantized-target draft adaptation](../../draftfit/research/experiments/quantized_target/README.md) | NF4/INT8 GPU updates and serving unverified; draft weights remain floating point. |
| [Spark sequential-prefix verification](../../draftfit/research/experiments/spark_verification/README.md) | Spark 2.5 text reference; native runtime parity and serving speed remain open. |
| [Resident target generation and CUDA batch tuning](../../draftfit/research/experiments/resident_generation/README.md) | Text-only generation; simulated CUDA OOM checks do not measure hardware throughput. |
| [SGLang milestone verification](../../draftfit/research/experiments/sglang_milestone/README.md) | Dense HF target on one CUDA device; longer-output parity failures remain open. |
| [Ling on the Ampere runtime profile](../../draftfit/research/experiments/ling_ampere/README.md) | SM80 has a basic Torch probe only; known fused-top-k failure. |

## Training methods and comparisons

| Experiment | Validation boundary |
| --- | --- |
| [EAGLE4 training and MiniCPM5-1B comparison](../../draftfit/research/experiments/eagle4/README.md) | Saved-feature training; live streaming unverified and strict serving parity incomplete. |
| [DFlash-family transfer and ablations](../../draftfit/research/experiments/family_ablation/README.md) | Cross-method initialization needs retraining; training proxies do not measure acceptance. |
| [Raw-text DFlash-family initialization](../../draftfit/research/experiments/raw_text_pretraining/README.md) | Single-device Qwen3-shaped drafts; no established acceptance improvement. |
| [Optuna training-option screening](../../draftfit/research/experiments/training_search/README.md) | Single-family held-out objective; the controller does not execute trials or measure speed. |

## Target-specific experiments

| Experiment | Validation boundary |
| --- | --- |
| [MiniCPM5-2B draft family](../../draftfit/research/experiments/minicpm5_family/README.md) | Pinned family experiments; measured exports have token parity gaps. |
| [Qwen3.5 hybrid text capture](../../draftfit/research/experiments/qwen35_text/README.md) | Tiny hybrid decoder coverage; no family-wide parity or rollback claim. |
| [Xing4.0-29B-A4B hyperconnection capture](../../draftfit/research/experiments/xing4/README.md) | Mean-reduced taps and YaRN; real target and serving must agree. |
| [ZDTaichu5.0-9B text drafts](../../draftfit/research/experiments/taichu5/README.md) | Scratch and official retarget configs differ; full-weight and serving parity unverified. |
| [Hemmingway-1 text drafts](../../draftfit/research/experiments/hemmingway1/README.md) | Candidate metadata; original weights and hybrid state behavior unverified. |
| [Agnes-3.0-Flash text capture](../../draftfit/research/experiments/agnes3/README.md) | Text decoder only; original-model training and serving remain open. |
| [LFM2.5 dense and MoE warm starts](../../draftfit/research/experiments/lfm25/README.md) | Official draft tensors loaded; original-target capture and exact serving still need checks. |
| [Humanizer joint-family preparation](../../draftfit/research/experiments/humanizer/README.md) | Pinned DFlash seed and text-only data; preflight is not a trained artifact. |

The [pinned recent-target recipes](../RECENT_MODEL_EXPERIMENTS.md) and [Taichu/LFM preparation record](../RECENT_MODEL_NEXT_TARGETS.md) distinguish metadata checks, original draft weights and real target runs. For the separate MiniCPM5-1B scratch experiment, see the [EAGLE4 comparison and receipts](eagle4-minicpm.md).
