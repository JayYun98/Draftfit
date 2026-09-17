"""Native Ling GPU parity gate (single GPU, TP=1, static full verify windows).

Capture twice with identical native server arguments, adding only
--enable-linear-replayssm-spec --linear-replayssm-cache-len 16 for replay:
  python scripts/validate_ling_replay.py capture legacy --port 30100 -- <server args>
  python scripts/validate_ling_replay.py capture replay --port 30100 -- <server args>
  python scripts/validate_ling_replay.py compare legacy replay

Both runs require --mamba-ssm-dtype float32, static ragged DSPARK, triton linear
attention, a GPU-compatible full attention backend, disabled radix cache, and the same local target/draft.
Use CUDA_VISIBLE_DEVICES=0 and PYTHONPATH=<pinned-sglang>/python. Run sequentially
after stopping the target server. This intentionally synchronizes/copies complete
committed states: it is a correctness gate, not a benchmark. Capture directories
must not exist. Missing hooks, insufficient accept/reject coverage, nonfinite
states, differing round boundaries or outputs fail closed. No GPU pass on CPU.
Raw captures default to 512 MiB per run (--max-capture-bytes to change).
Exceeding the selected budget fails the gate.
Requests default to 24 new tokens; every round is saved once and compared in order.
"""

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MAX_CAPTURE_BYTES = 512 * 1024 * 1024
CAPTURE_RID = "ling-replay-validation"
DEFAULT_PROMPT = "Explain how a hash table resolves collisions, with a worked example and Python code."


def is_capture_batch(batch):
    return len(batch.reqs) == 1 and batch.reqs[0].rid == CAPTURE_RID


def install_hook():
    """Install before imports in every spawned worker, without importing CUDA."""
    module_name = "sglang.srt.speculative.dspark_components.dspark_worker_v2"

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original_exec = spec.loader.exec_module

            def execute(module):
                original_exec(module)
                cls = module.DSparkWorkerV2
                original = cls._commit_target_mamba_states_after_verify
                round_index = 0
                capture_bytes = 0

                def commit(self, **kwargs):
                    nonlocal capture_bytes, round_index
                    import torch

                    if not is_capture_batch(kwargs["batch"]):
                        return original(self, **kwargs)
                    diagnostic = (
                        round_index == 0
                        and os.environ.get("LING_REPLAY_CAPTURE_INITIAL") == "1"
                    )
                    if not diagnostic:
                        # Even a slot-validity .any() would synchronize CUDA.
                        # Keep all tensor inspection AFTER the real commit.
                        result = original(self, **kwargs)
                    if not self._need_mamba_verify_commit:
                        raise RuntimeError("Ling mamba commit path disabled")
                    backend = (
                        self.target_worker.model_runner.attn_backend.linear_attn_backend
                    )
                    n = kwargs["commit_lens"].numel()
                    slots = backend.forward_metadata.mamba_cache_indices[:n].long()
                    if n != 1 or (slots < 0).any():
                        raise RuntimeError("validator requires one real request, TP=1")
                    pool = backend.req_to_token_pool.mamba_pool
                    state = (
                        backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                    )
                    if state.temporal.dtype != torch.float32:
                        raise RuntimeError("both modes require float32 persistent SSM")
                    # Diagnostic only: this adds a pre-commit CUDA sync and may
                    # mask a race. Final gates must also run without this option.
                    initial = (
                        state.temporal.index_select(1, slots).cpu().clone()
                        if diagnostic
                        else None
                    )
                    if diagnostic:
                        result = original(self, **kwargs)
                    destination = Path(os.environ["LING_REPLAY_CAPTURE"])
                    tensors = [state.temporal, *state.conv]
                    next_bytes = sum(
                        x.numel() // x.shape[1] * n * x.element_size() for x in tensors
                    )
                    if initial is not None:
                        next_bytes += initial.numel() * initial.element_size()
                    budget = int(
                        os.environ.get(
                            "LING_REPLAY_MAX_CAPTURE_BYTES", MAX_CAPTURE_BYTES
                        )
                    )
                    if capture_bytes + next_bytes > budget:
                        (destination / "FAILED").write_text(
                            f"raw committed state capture exceeds {budget} bytes"
                        )
                        raise RuntimeError(
                            "raw state capture exceeds budget; parity unproven"
                        )
                    capture_bytes += next_bytes
                    record = {
                        "rid": CAPTURE_RID,
                        "pre": kwargs["seq_lens_pre_verify"].cpu().tolist(),
                        "post": kwargs["seq_lens_post_verify"].cpu().tolist(),
                        "commit": kwargs["commit_lens"].cpu().tolist(),
                        "window": int(self.verify_num_draft_tokens),
                        "replay": bool(
                            pool.replayssm_spec_fold and pool.replayssm_is_kda
                        ),
                        "temporal": state.temporal.index_select(1, slots).cpu().clone(),
                        "conv": [
                            x.index_select(1, slots).cpu().clone() for x in state.conv
                        ],
                    }
                    if initial is not None:
                        record["initial_temporal"] = initial
                    with (
                        destination / f"commits-{os.getpid()}-{round_index:06d}.pt"
                    ).open("xb") as output:
                        torch.save(record, output)
                    round_index += 1
                    return result

                cls._commit_target_mamba_states_after_verify = commit

            spec.loader.exec_module = execute
            return spec

    sys.meta_path.insert(0, Finder())


def check_coverage(rounds):
    if len(rounds) < 3:
        raise ValueError("need multiple nonterminal verify rounds")
    # Exclude the final round, where max_new_tokens can trim acceptance.
    active = rounds[:-1]
    if not any(1 < r["commit"][0] for r in active):
        raise ValueError("no actual accepted draft tokens")
    if not any(r["commit"][0] < r["window"] for r in active):
        raise ValueError("no rejected draft tokens in nonterminal rounds")


def capture(
    directory,
    port,
    server_args,
    max_capture_bytes=MAX_CAPTURE_BYTES,
    max_new_tokens=24,
    prompt=DEFAULT_PROMPT,
    capture_initial=False,
    capture_states=True,
):
    import socket

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dspark.launch_plan import _terminate_processes

    if max_capture_bytes <= 0:
        raise ValueError("max-capture-bytes must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    if "--port" in server_args:
        raise ValueError("pass --port to capture, not server arguments")
    (directory / "args.json").write_text(json.dumps(server_args))
    request = {
        "rid": CAPTURE_RID,
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    (directory / "request.json").write_text(json.dumps(request))
    (directory / "capture-mode.json").write_text(
        json.dumps(
            {"capture_states": capture_states, "capture_initial": capture_initial}
        )
    )
    with tempfile.TemporaryDirectory(prefix="ling-replay-hook-") as hookdir:
        hook = Path(hookdir) / "sitecustomize.py"
        if capture_states:
            hook.write_text(
                f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r})['install_hook']()\n"
            )
        env = dict(
            os.environ,
            LING_REPLAY_CAPTURE=str(directory),
            LING_REPLAY_MAX_CAPTURE_BYTES=str(max_capture_bytes),
            LING_REPLAY_CAPTURE_INITIAL="1" if capture_initial else "0",
        )
        env["PYTHONPATH"] = hookdir + os.pathsep + env.get("PYTHONPATH", "")
        with (directory / "server.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    *server_args,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 900
                while True:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            f"server failed or timed out; see {directory / 'server.log'}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=3
                        ):
                            break
                    except OSError:
                        time.sleep(1)
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/generate",
                    data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=900) as response:
                    output = json.load(response)
                (directory / "output.json").write_text(json.dumps(output))
            finally:
                # Include the owned session even if its leader exits during cleanup.
                _terminate_processes([proc], exited_group_leaders=[proc])


def compare_state(x, y, *, round_index, name, atol, rtol):
    import torch

    context = f"round {round_index}: {name}"
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError(f"{context}: nonfinite committed state")
    try:
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(f"{context}: {exc}") from exc


def compare(legacy, replay, atol, rtol):
    import torch

    arguments = [
        json.loads((Path(d) / "args.json").read_text()) for d in (legacy, replay)
    ]
    baseline, candidate = arguments
    expected = list(baseline)
    expected += ["--enable-linear-replayssm-spec", "--linear-replayssm-cache-len", "16"]
    if candidate != expected:
        raise ValueError(
            "replay args must equal legacy args plus replay-spec/cache-len 16 flags"
        )
    if not any(
        baseline[i : i + 2] == ["--mamba-ssm-dtype", "float32"]
        for i in range(len(baseline))
    ):
        raise ValueError("baseline must explicitly select float32 mamba state")
    if "--disable-radix-cache" not in baseline:
        raise ValueError(
            "disable radix cache: tracked checkpoint slots are outside this gate"
        )
    requests = [
        json.loads((Path(d) / "request.json").read_text()) for d in (legacy, replay)
    ]
    if requests[0] != requests[1]:
        raise ValueError("capture requests differ")
    runs = []
    for directory in (Path(legacy), Path(replay)):
        if (directory / "FAILED").exists():
            raise ValueError((directory / "FAILED").read_text())
        files = sorted(directory.glob("commits-*.pt"))
        workers = set()
        for index, file in enumerate(files):
            parts = file.stem.split("-")
            if len(parts) != 3 or not parts[1].isdigit() or parts[2] != f"{index:06d}":
                raise ValueError(
                    "capture indices must be contiguous from zero, with one worker"
                )
            workers.add(parts[1])
        if len(workers) != 1:
            raise ValueError("need exactly one hooked worker capture per run")
        runs.append(files)
    a, b = runs
    if len(a) != len(b):
        raise ValueError("verify round boundaries diverged")
    coverage = [[], []]
    for index, (left_path, right_path) in enumerate(zip(a, b)):
        left = torch.load(left_path, map_location="cpu", weights_only=True)
        right = torch.load(right_path, map_location="cpu", weights_only=True)
        for records, record in zip(coverage, (left, right)):
            if record.get("rid") != CAPTURE_RID:
                raise ValueError("capture must belong to the validation request")
            records.append({"commit": record["commit"], "window": record["window"]})
        if left["replay"] or not right["replay"]:
            raise ValueError("expected legacy then actual KDA ReplaySSM")
        for key in ("pre", "post", "commit", "window"):
            if left[key] != right[key]:
                raise ValueError(f"round {index}: {key} diverged")
        if len(left["conv"]) != len(right["conv"]):
            raise ValueError("convolution state count differs")
        if ("initial_temporal" in left) != ("initial_temporal" in right):
            raise ValueError("initial state capture coverage differs")
        if "initial_temporal" in left:
            compare_state(
                left["initial_temporal"],
                right["initial_temporal"],
                round_index=index,
                name="initial_temporal",
                atol=atol,
                rtol=rtol,
            )
        names = ["temporal", *[f"conv[{i}]" for i in range(len(left["conv"]))]]
        for name, x, y in zip(
            names,
            [left["temporal"], *left["conv"]],
            [right["temporal"], *right["conv"]],
        ):
            compare_state(x, y, round_index=index, name=name, atol=atol, rtol=rtol)
    for records in coverage:
        check_coverage(records)
    outputs = [
        json.loads((Path(d) / "output.json").read_text()) for d in (legacy, replay)
    ]
    if any("output_ids" not in o for o in outputs):
        raise ValueError("server did not expose output_ids; token parity unproven")
    if outputs[0]["output_ids"] != outputs[1]["output_ids"]:
        raise ValueError("generated token IDs differ")
    result = {
        "passed": True,
        "rounds": len(a),
        "atol": atol,
        "rtol": rtol,
        "scope": "single-request full committed temporal/conv and output token parity",
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("directory")
    cap.add_argument("--port", type=int, default=30100)
    cap.add_argument("--max-capture-bytes", type=int, default=MAX_CAPTURE_BYTES)
    cap.add_argument("--max-new-tokens", type=int, default=24)
    cap.add_argument("--prompt", default=DEFAULT_PROMPT)
    cap.add_argument(
        "--capture-initial",
        action="store_true",
        help="diagnostic first pre-commit state; adds a CUDA synchronization",
    )
    cap.add_argument(
        "--tokens-only",
        action="store_true",
        help="run without state hooks; token-only evidence cannot pass the state comparator",
    )
    cmp = sub.add_parser("compare")
    cmp.add_argument("legacy")
    cmp.add_argument("replay")
    cmp.add_argument("--atol", type=float, default=1e-5)
    cmp.add_argument("--rtol", type=float, default=1e-4)
    args, extra = parser.parse_known_args()
    if args.command == "capture":
        capture(
            args.directory,
            args.port,
            extra,
            args.max_capture_bytes,
            args.max_new_tokens,
            args.prompt,
            args.capture_initial,
            not args.tokens_only,
        )
    else:
        if extra:
            parser.error(f"unexpected arguments: {extra}")
        compare(args.legacy, args.replay, args.atol, args.rtol)
