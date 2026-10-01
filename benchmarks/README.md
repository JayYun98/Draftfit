# Benchmarking for Speculative Decoding

## Overview

The primary user workflow is `draftfit benchmark`; see the
[performance guide](../docs/PERFORMANCE_GATE.md). This directory also contains
source-checkout experiments and benchmark harnesses.

### Domino trainer microbenchmark

From the repository root in a provisioned GPU environment, run
`python -m benchmarks.bench_domino_mfu`. It uses the checked-in Qwen3-8B Domino
configuration and synthetic inputs to measure trainer compute, not inference
speedup. Its MFU reference is H200 SXM BF16 dense peak; that percentage is not
a hardware-adjusted utilization metric for other GPUs.

### Kernel and loss microbenchmarks

From the repository root, use `python -m benchmarks.benchmark_loss` or
`python -m benchmarks.benchmark_flex_attention` in their CUDA environment.
These developer probes are not installed with the Draftfit wheel. The attention
probe also requires matplotlib. They measure kernels, not workload speedup.

### EAGLE3 benchmark harness

The commands below run from this `benchmarks/` directory.

We provided a unified script to test the performance of the Speculative Decoding with EAGLE3 algorithm on multiple datasets. You can follow the steps below to run the benchmarks.

## Run Benchmarks

### Launch SGLang and Benchmarker Concurrently

`bench_eagle3.py` can help you launch a SGLang server process and a Benchmarking process concurrently. In this way, you don't have to launch the SGLang server manually, this script will manually handle the SGLang launch under different speculative decoding configurations. Some important arguments are:
- `--model-path`: the path to the target model.
- `--speculative-draft-model-path`: the path to the draft model.
- `--port`: the port to launch the SGLang server.
- `--trust-remote-code`: trust the remote code.
- `--mem-fraction-static`: the memory fraction for the static memory.
- `--tp-size`: the tensor parallelism size.
- `--attention-backend`: the attention backend.
- `--config-list`: the list of speculative decoding configuration to test, the format is `<batch-size>,<num-steps>,<topk>,<num-draft-tokens>`.
- `--benchmark-list`: the list of benchmarks to test, the format is `<benchmark-name>:<num-prompts>:<subset>`.

```shell
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --attention-backend fa3 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench gsm8k:5 ceval:5:accountant \
    --dtype bfloat16
```

### Launch Benchmarker Independently

If you want to launch the SGLang server independently, you can use the following command.

```shell
# you can launch a server
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 1 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

Then we can start benchmarking. Note that you should use the same host and port as the one used in the SGLang server. Note that `--skip-launch-server` is required to skip the launch of the SGLang server.

```bash
python bench_eagle3.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --port 30000 \
        --config-list 1,3,1,4 \
        --benchmark-list mtbench:5 ceval:5:accountant gsm8k:5 humaneval:5 math500:5 mtbench:5 aime:1 \
        --skip-launch-server
```
