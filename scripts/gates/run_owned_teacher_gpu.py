"""Real two-GPU owned teacher -> Mooncake -> draft functional gate.

Requires a running Mooncake master configured through MOONCAKE_* environment
variables and hard-pin capable bindings in both Python environments. This gate
does not rent hardware, install packages, or certify serving acceleration.
Use vllm==0.22.1 in --teacher-python for the vllm backend. Run once per backend.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("transformers", "vllm"))
    parser.add_argument("--teacher-python", default=sys.executable)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--revision", required=True, help="immutable Hugging Face commit SHA"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="require deterministic trainer operations for exact resume",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("dspark", "dflash2"),
        default=["dspark", "dflash2"],
    )
    parser.add_argument("--teacher-memory-fraction", type=float, default=0.4)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a resolved immutable 40-character commit SHA")
    if not 20 <= args.steps <= 100:
        parser.error("--steps must be between 20 and 100")
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error("--algorithms must not repeat")
    if not 0 < args.teacher_memory_fraction < 1 or args.startup_timeout <= 0:
        parser.error("invalid memory fraction or timeout")
    return args


def training_inputs(tensors, algorithm, device):
    inputs = {
        key: tensors[key].to(device)
        for key in ("input_ids", "hidden_states", "loss_mask")
    }
    if algorithm == "dspark":
        inputs["target_last_hidden_states"] = tensors["target"].to(device)
    return inputs


def stop_owned_group(process):
    """Reap only the session created for this gate, including orphaned workers."""

    def alive():
        process.poll()  # Reap the leader even when workers outlive it.
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    for sig, timeout in ((signal.SIGTERM, 45), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait(timeout=1)
            return
        deadline = time.monotonic() + timeout
        while alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not alive():
            return
    raise RuntimeError(
        "owned teacher process group remains after bounded SIGKILL cleanup"
    )


def main(argv=None):
    args = arguments(argv)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    from draftfit.algorithms.common.dflash_family_model import (
        OnlineDFlash2Model,
        OnlineDSparkModel,
    )
    from draftfit.export.to_hf import export_to_hf
    from draftfit.inference.adapters.server_capture import (
        ServerCaptureSchema,
        TeacherServerCaptureAdapter,
    )
    from draftfit.inference.capture import CaptureConfig
    from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel
    from draftfit.modeling.draft.dspark import DSparkDraftModel
    from draftfit.offline_capture.transformers import OfflineTransformersCapture
    from draftfit.runtime.contracts import PromptTask, SampleRef
    from draftfit.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "real gate requires two visible CUDA GPUs; no CPU/mock passing path"
        )
    if args.output.exists():
        raise ValueError("--output must be new to avoid overwriting previous evidence")
    args.output.mkdir(parents=True)
    teacher_versions = json.loads(
        subprocess.check_output(
            [
                args.teacher_python,
                "-c",
                "import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in "
                + repr(
                    ["torch", "transformers"]
                    + (["vllm"] if args.backend == "vllm" else [])
                )
                + "}))",
            ],
            text=True,
        )
    )
    if args.backend == "vllm" and teacher_versions["vllm"] != "0.22.1":
        raise RuntimeError("vLLM teacher must use the audited vllm==0.22.1 runtime")
    torch.cuda.set_device(1)
    torch.manual_seed(7)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    reference_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=dtype, attn_implementation="sdpa"
        )
        .to("cuda:1")
        .eval()
        .requires_grad_(False)
    )
    if reference_model.config.model_type != "qwen3":
        raise ValueError("this bounded gate only covers dense Qwen3")
    taps = [3, reference_model.config.num_hidden_layers - 4]
    reference = OfflineTransformersCapture(reference_model)
    reference.set_capture_layers(taps, capture_method="dflash")
    texts = [
        "Explain why the sky looks blue in clear weather and why sunsets often look red.",
        "Describe how a seed grows into a plant, including the roles of water, roots and sunlight.",
        "Explain what happens when ice melts and why temperature changes can change matter's state.",
        "Describe a simple method for checking whether a number is prime and illustrate it with seventeen.",
    ]
    prompts = []
    for text in texts:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": text},
                {
                    "role": "assistant",
                    "content": "Let us explain this carefully. " + text,
                },
            ],
            tokenize=False,
            enable_thinking=False,
        )
        ids = tokenizer(rendered, add_special_tokens=False).input_ids[:96]
        if len(ids) < 16:
            raise ValueError("fixture prompt unexpectedly short")
        prompts.append(ids)
    run_id = "owned-gpu-" + uuid.uuid4().hex[:12]
    setup = dict(
        local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
        metadata_server=os.environ.get(
            "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
        ),
        master_server_addr=os.environ.get(
            "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
        ),
        protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
        rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
        global_segment_size=int(
            os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 28)
        ),
        local_buffer_size=int(os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 28)),
    )
    store = MooncakeFeatureStore(store_id=run_id, setup_kwargs=setup)
    if getattr(store._put_config, "with_hard_pin", False) is not True:
        raise RuntimeError("real gate requires hard-pin capable Mooncake")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        args.teacher_python,
        "-m",
        "draftfit.inference.teacher_server",
        "--target-backend",
        args.backend,
        "--model-path",
        args.model,
        "--revision",
        args.revision,
        "--dtype",
        args.dtype,
        "--capture-method",
        "dflash",
        "--aux-layer-ids",
        *map(str, taps),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "256",
        "--device",
        "cuda:0",
        "--gpu-memory-utilization",
        str(args.teacher_memory_fraction),
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": visible,
        "DISAGG_STORE_ID": run_id,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    report = dict(
        passed=False,
        backend=args.backend,
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        scope="two-GPU functional transport/training, four repeated prompts; no serving speedup claim",
        teacher_gpu=0,
        trainer_gpu=1,
        torch=torch.__version__,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        teacher_command=command,
        teacher_versions=teacher_versions,
        real_mooncake=True,
        algorithms={},
        gpu_kernel_overlap_verified=False,
    )
    process = None
    try:
        with (args.output / "teacher.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError("teacher failed to start; inspect teacher.log")
            try:
                with urllib.request.urlopen(url + "/health", timeout=3) as response:
                    report["teacher_health"] = json.load(response)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("teacher readiness timed out")
                time.sleep(1)
        schema = ServerCaptureSchema(
            "hidden_states",
            "target",
            (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
        )
        contract = CaptureConfig.from_strategy(
            required_features={"hidden_states", "target", "input_ids", "loss_mask"},
            aux_hidden_state_layer_ids=tuple(taps),
            target_repr="hidden_state",
            target_hidden_size=reference_model.config.hidden_size,
            target_vocab_size=reference_model.config.vocab_size,
            draft_vocab_size=reference_model.config.vocab_size,
        )
        adapter = TeacherServerCaptureAdapter(
            url,
            store,
            run_id=run_id,
            algorithm="dflash2",
            schema=schema,
            backend=args.backend,
            target_model_version=args.model,
            target_revision=args.revision,
        )

        def fetch(index):
            started = time.monotonic()
            ids = prompts[index % len(prompts)]
            task = PromptTask(
                task_id=f"sample-{index}",
                run_id=run_id,
                source_id="functional-gate",
                payload={"input_ids": ids, "loss_mask": [1] * len(ids)},
                max_length=256,
            )
            (ref,) = adapter.produce_refs([task], capture=contract)
            if not isinstance(ref, SampleRef):
                raise RuntimeError(f"capture failed: {ref}")
            try:
                tensors, handle = store.get(ref, device="cpu")
                store.release(handle)
            except BaseException:
                store.abort(ref.sample_id, reason="gpu-gate-fetch-failed")
                raise
            return tensors, (started, time.monotonic())

        parity = {}
        report["parity"] = parity
        for prompt_index in range(len(prompts)):
            probe, _ = fetch(prompt_index)
            expected = reference.capture(
                input_ids=probe["input_ids"],
                attention_mask=torch.ones_like(probe["input_ids"]),
                loss_mask=probe["loss_mask"],
            )
            for name, expected_tensor in (
                ("hidden_states", expected.hidden_states),
                ("target", expected.last_hidden_states),
            ):
                actual, desired = probe[name].float(), expected_tensor.cpu().float()
                atol, rtol = (
                    (0.125, 0.025) if args.backend == "vllm" else (0.015625, 0.005)
                )
                key = f"prompt_{prompt_index}/{name}"
                parity[key] = dict(
                    max_abs=float((actual - desired).abs().max()),
                    atol=atol,
                    rtol=rtol,
                    finite=bool(torch.isfinite(actual).all()),
                    within_tolerance=bool(
                        torch.allclose(actual, desired, atol=atol, rtol=rtol)
                    ),
                )
                segments = (
                    zip(
                        taps,
                        actual.split(reference_model.config.hidden_size, dim=-1),
                        desired.split(reference_model.config.hidden_size, dim=-1),
                    )
                    if name == "hidden_states"
                    else [("final_norm", actual, desired)]
                )
                parity[key]["layers"] = {}
                for tap, observed, wanted in segments:
                    parity[key]["layers"][str(tap)] = dict(
                        max_expected=float(wanted.abs().max()),
                        max_abs=float((observed - wanted).abs().max()),
                        relative_l2=float(
                            (observed - wanted).norm() / wanted.norm().clamp_min(1e-30)
                        ),
                        cosine=float(
                            torch.nn.functional.cosine_similarity(
                                observed.flatten(), wanted.flatten(), dim=0
                            )
                        ),
                    )
                if not parity[key]["finite"] or not parity[key]["within_tolerance"]:
                    torch.save(
                        dict(
                            actual=actual,
                            expected=desired,
                            input_ids=probe["input_ids"],
                            taps=taps,
                            feature=name,
                            prompt_index=prompt_index,
                        ),
                        args.output / "parity-failure.pt",
                    )
                    raise AssertionError(f"{key} teacher/HF parity failed")
        index = len(prompts)
        for algorithm in args.algorithms:
            values = reference_model.config.to_dict()
            values.update(
                num_hidden_layers=1,
                layer_types=["full_attention"],
                block_size=4,
                target_num_hidden_layers=reference_model.config.num_hidden_layers,
                num_target_layers=reference_model.config.num_hidden_layers,
                target_layer_ids=taps,
                mask_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            if algorithm == "dflash2":
                values.update(
                    architectures=["DFlash2DraftModel"],
                    conv_group_size=64,
                    selector_rank=32,
                    selector_top_k=16,
                )
                draft = DFlash2DraftModel(DFlash2Config(**values))
                wrapper_type = OnlineDFlash2Model
            else:
                values.update(
                    architectures=["DSparkDraftModel"],
                    dflash_config=dict(
                        target_layer_ids=taps,
                        projector_type="dspark",
                        markov_rank=32,
                        markov_head_type="vanilla",
                        mask_token_id=values["mask_token_id"],
                        enable_confidence_head=True,
                        confidence_head_alpha=1.0,
                    ),
                )
                draft = DSparkDraftModel(Qwen3Config(**values))
                wrapper_type = OnlineDSparkModel
            draft = draft.to(device="cuda:1", dtype=dtype)
            wrapper = wrapper_type(
                draft,
                reference_model.lm_head,
                reference_model.get_input_embeddings(),
                mask_token_id=values["mask_token_id"],
                block_size=4,
                attention_backend="sdpa",
                num_anchors=4,
                anchor_sampling="uniform",
                objective_chunk_blocks=4,
            )
            optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-4)
            initial = {
                name: tensor.detach().cpu().clone()
                for name, tensor in draft.named_parameters()
            }
            losses, overlaps = [], 0

            def step(tensors):
                tensors = training_inputs(tensors, algorithm, "cuda:1")
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = wrapper(**tensors)
                if not torch.isfinite(loss):
                    raise AssertionError("nonfinite training loss")
                loss.backward()
                gradients = [p.grad for p in draft.parameters() if p.grad is not None]
                if (
                    not gradients
                    or not all(torch.isfinite(g).all() for g in gradients)
                    or not any(g.count_nonzero() for g in gradients)
                ):
                    raise AssertionError("invalid or zero draft gradients")
                optimizer.step()
                torch.cuda.synchronize(1)
                return float(loss.detach())

            current, _ = fetch(index)
            index += 1
            with ThreadPoolExecutor(max_workers=1) as pool:
                for number in range(args.steps):
                    pending = pool.submit(fetch, index)
                    index += 1
                    start = time.monotonic()
                    value = step(current)
                    end = time.monotonic()
                    current, interval = pending.result()
                    overlaps += int(max(start, interval[0]) < min(end, interval[1]))
                    losses.append(value)
                    print(
                        json.dumps(
                            dict(algorithm=algorithm, step=number + 1, loss=value)
                        ),
                        flush=True,
                    )
            if not overlaps:
                raise AssertionError("no observed capture/training interval overlap")
            if not any(
                not torch.equal(initial[n], p.detach().cpu())
                for n, p in draft.named_parameters()
            ):
                raise AssertionError("draft parameters did not update")
            directory = args.output / algorithm
            directory.mkdir()
            checkpoint = directory / "training_state.pt"
            torch.save(
                dict(
                    strategy=algorithm,
                    draft_state_dict=draft.state_dict(),
                    optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(1),
                    step=args.steps,
                ),
                checkpoint,
            )
            expected_loss = step(current)
            expected_state = {
                n: t.detach().cpu().clone() for n, t in draft.state_dict().items()
            }
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            draft.load_state_dict(state["draft_state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], 1)
            resumed_loss = step(current)
            resume_exact = expected_loss == resumed_loss and all(
                torch.equal(expected_state[n], t.cpu())
                for n, t in draft.state_dict().items()
            )
            report.setdefault("resume_diagnostics", {})[algorithm] = dict(
                expected_loss=expected_loss,
                resumed_loss=resumed_loss,
                max_weight_difference=max(
                    float(
                        (expected_state[n].float() - t.detach().cpu().float())
                        .abs()
                        .max()
                    )
                    for n, t in draft.state_dict().items()
                ),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            )
            if not resume_exact:
                raise AssertionError("same-state next-step resume diverged")
            config_path = directory / "draft_config.json"
            draft.config.to_json_file(config_path)
            export_to_hf(str(checkpoint), str(config_path), str(directory / "hf"))
            restored = type(draft).from_pretrained(directory / "hf", dtype=dtype)
            if not all(
                torch.equal(t.cpu(), restored.state_dict()[n])
                for n, t in state["draft_state_dict"].items()
            ):
                raise AssertionError("export/reload changed draft weights")
            report["algorithms"][algorithm] = dict(
                steps=args.steps,
                losses=losses,
                capture_http_training_wall_overlap_intervals=overlaps,
                next_step_replays=2,
                resume_exact=True,
                hf_reload_exact=True,
            )
            del wrapper, draft, optimizer, restored, initial, state, expected_state
            torch.cuda.empty_cache()
        report["passed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if process is not None:
                stop_owned_group(process)
        except Exception as exc:
            report["passed"] = False
            report["process_cleanup_error"] = str(exc)
        try:
            store.discard_external_attempts(reason="gpu-gate-terminal")
            store.drain_pending_removals(retry_interval_s=0.1)
        except Exception as exc:
            report["passed"] = False
            report["cleanup_error"] = str(exc)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    if not report["passed"]:
        raise RuntimeError("GPU gate cleanup failed; inspect result.json")


if __name__ == "__main__":

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    main()
