# Native teacher GPU campaign environment

Metadata checked 2026-09-14; no instance, install or model download performed.

## Image and hardware

Use `vllm/vllm-openai:v0.22.1`, Linux amd64. Docker Hub currently resolves its
amd64 image to `sha256:55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5`
(9.22 GB compressed). For immutable selection use
`vllm/vllm-openai@sha256:55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5`.
Override the image's API-server entrypoint with a shell for campaign setup.
[Registry metadata](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.22.1)

The tagged build defaults are Python 3.12, Ubuntu 22.04, CUDA 13.0.2;
its required Torch version is 2.11.0. Transformers is not pinned by upstream;
the project requires 5.8.1, which satisfies upstream's allowed range.
Record installed versions before testing; Dockerfile defaults are not a runtime measurement.
[Dockerfile](https://github.com/vllm-project/vllm/blob/v0.22.1/docker/Dockerfile),
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.22.1/requirements/cuda.txt),
[common requirements](https://github.com/vllm-project/vllm/blob/v0.22.1/requirements/common.txt)

Practical budget target: one host, two 24 GB RTX 3090/4090 GPUs, 32 GB RAM,
80 GB available disk (image layers, dependencies, model cache), NVIDIA driver
580 or newer. This is a workload sizing recommendation, not a current cheapest
offer quote. Compare total hourly cost and provisioning delay; no NVLink/RDMA
requirement for this single-host TCP feature-transfer gate. CUDA 13 requires
driver >=580; do not assume forward compatibility on consumer GPUs.
[NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## Preserve the image's backend

Run inside the selected image and repository checkout. These commands are a
runbook, not a claim they have been executed. Constraints keep existing binary
dependencies pinned and fail resolution if the requested combination conflicts.
The interpreter path below is discovered inside the running container. The
Dockerfile's `/opt/venv` default has not been verified by starting this image.

```sh
campaign_python=$(command -v python3)
campaign_python=$("$campaign_python" -c 'import sys; print(sys.executable)')
"$campaign_python" -c 'import sys; print(sys.executable, sys.version)'
uv pip freeze --python "$campaign_python" > image-packages.txt
rg -v '^transformers==' image-packages.txt > image-constraints.txt
uv pip install --python "$campaign_python" -c image-constraints.txt \
  -e . 'vllm==0.22.1' 'transformers==5.8.1' \
  'mooncake-transfer-engine-cuda13==0.3.13.post1'
uv pip check --python "$campaign_python"
"$campaign_python" -c 'import torch,transformers,vllm; print(torch.__version__,torch.version.cuda,transformers.__version__,vllm.__version__); assert torch.cuda.device_count()==2'
"$campaign_python" -c 'from mooncake.store import MooncakeDistributedStore,ReplicateConfig; c=ReplicateConfig(); assert hasattr(c,"with_hard_pin"); c.with_hard_pin=True; assert all(hasattr(MooncakeDistributedStore,n) for n in ("put_from","get_into","register_buffer","unregister_buffer"))'
command -v mooncake_master
```

If `rg` is absent, use `grep -v` for the one-line constraints filter. Do not run
`uv sync` against the CPU lock or install stock SGLang into this image. If an
ordinary `mooncake-transfer-engine` distribution is already present, resolve that
conflict first: both packages install the same `mooncake` namespace.

CUDA13 wheel: `mooncake-transfer-engine-cuda13==0.3.13.post1`, cp312 Linux x86_64
SHA256 `155bcb1b02d07f4b78268641880ada0558abeac1e5db0624f879f35bba4c8cd6`.
The matching tagged API documents `with_hard_pin`, `put_from`, and `get_into`;
an actual registered-buffer round trip against the running master remains a gate.
[PyPI metadata](https://pypi.org/pypi/mooncake-transfer-engine-cuda13/0.3.13.post1/json),
[tagged Mooncake API](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/docs/source/api-reference/python/mooncake-store.md)

## Campaign sequence

Use Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`.
First test raw Mooncake hard-pin put/get/remove. Then compare HF and native vLLM
features for identical tokens/taps in the same environment, followed by each
teacher's actual two-GPU capture-to-draft update gate. GPU 0 owns the teacher;
GPU 1 owns the trainer. Run the two backends sequentially with unique namespaces
and fresh control/output directories. Keep prompt length and batch size small
initially (128 tokens, batch 1); increase only after correctness passes.

### Implemented two-GPU functional gate (not yet GPU-executed)

`run_owned_teacher_gpu.py` requires an already-running Mooncake master. Set
`MOONCAKE_METADATA_SERVER`, `MOONCAKE_MASTER_SERVER_ADDR`, and `MOONCAKE_PROTOCOL=tcp`
to that master's actual endpoints before executing. The script creates its own
teacher process and unique store namespace; it does not create the master or
run the managed training supervisor. Both GPUs must remain visible to its main
process. It currently uses four repeated prompts capped at 96 tokens.

```sh
"$campaign_python" scripts/gates/run_owned_teacher_gpu.py \
  --backend transformers --teacher-python "$campaign_python" \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output ./evidence/native-hf --steps 20 --algorithms dspark dflash2
"$campaign_python" scripts/gates/run_owned_teacher_gpu.py \
  --backend vllm --teacher-python "$campaign_python" \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output ./evidence/native-vllm --steps 20 --algorithms dspark dflash2 \
  --teacher-memory-fraction 0.4 --startup-timeout 900
```

Each output directory must be new. Separate teacher environments can be supplied
through `--teacher-python`; both environments need compatible Mooncake bindings.
Inspect `result.json` and `teacher.log`, including cleanup errors. This functional
gate does not establish serving speedup, GPU kernel overlap or held-out quality.

### Separate managed supervisor gate (local plan validated; GPU not executed)

The checked-in `scripts/gates/fixtures/owned-managed-{transformers,vllm}.yaml`
configs use the same four conversation rows, two DSpark updates, target taps
`[3,24]`, two draft layers, batch 1 and length 128. They select `tp_size: 1`,
capture device `["0"]`, trainer device `["1"]`, and Mooncake TCP. Run from the
repository root. Unlike the functional gate, the managed supervisor creates its
own master: stop a prior standalone gate's master before reusing the default
ports. Both configs were validated through the real CLI `--plan` with
`HF_HUB_OFFLINE=1` and cached target metadata, without mocks or GPU execution.

```sh
"$campaign_python" -m specforge.cli train -c scripts/gates/fixtures/owned-managed-transformers.yaml --plan
"$campaign_python" -m specforge.cli train -c scripts/gates/fixtures/owned-managed-transformers.yaml
"$campaign_python" -m specforge.cli train -c scripts/gates/fixtures/owned-managed-vllm.yaml --plan
"$campaign_python" -m specforge.cli train -c scripts/gates/fixtures/owned-managed-vllm.yaml
```

The model ID/revision is real; weights are resolved on the GPU host. Relative
paths assume the repository root. Output directories under `outputs/owned-managed-*`
must be fresh; preserve prior results and change run/control/output paths for a
rerun. No model weight download was performed for local plan validation.
Do not treat a successful low-level optimizer loop or local launch plan as proof
of managed supervisor integration: require both updates, checkpoint output,
successful terminal status and cleaned-up teacher/master processes on the GPU host.
SGLang's preserved path needs its existing patched image only if regression
evidence points to it; installing that stack is not a prerequisite for the new
native teacher campaign.
