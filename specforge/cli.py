# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The single public SpecForge training entry point.

``specforge train --config run.yaml [section.field=value ...]`` builds the
validated :class:`~specforge.config.Config`, assembles the models, and runs
training through the DataFlow launch builders — the same wiring the
programmatic path uses, behind one typed config.

``deployment.trainer`` defines the process topology. The CLI self-launches
multi-rank workers and recognizes an existing torchrun worker environment
without nesting another launcher.

Model/data assembly lives in :mod:`specforge.training.assembly`; this module is
deliberately limited to command parsing and distributed process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
from contextlib import contextmanager
from typing import Iterator, List, Optional

from specforge.config import load_config

class _WorkerTermination(BaseException):
    """Translate a process signal into normal Python stack unwinding."""

    def __init__(self, signum: int):
        self.signum = signum


@contextmanager
def _worker_signal_unwind() -> Iterator[None]:
    """Make worker termination run training and distributed cleanup blocks.

    Managed supervisors terminate worker process groups with SIGTERM.  Python's
    default SIGTERM action exits immediately, bypassing ``finally`` blocks.  The
    first managed signal is therefore raised as a ``BaseException``; subsequent
    signals are ignored while cleanup runs, after which the original handlers
    are restored.  A supervising parent may still enforce its grace period with
    SIGKILL if cleanup cannot finish.
    """
    managed_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        managed_signals.append(signal.SIGHUP)
    previous_handlers = {}

    def unwind(signum, _frame):
        for installed in previous_handlers:
            signal.signal(installed, signal.SIG_IGN)
        raise _WorkerTermination(signum)

    try:
        for signum in managed_signals:
            try:
                previous_handlers[signum] = signal.signal(signum, unwind)
            except ValueError:
                # Embedded callers may execute the CLI from a non-main thread,
                # where Python does not permit signal handler installation.
                for installed, handler in previous_handlers.items():
                    signal.signal(installed, handler)
                previous_handlers.clear()
                break
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _bootstrap_single_process_env() -> None:
    """Provide ``env://`` rendezvous values for a direct one-GPU invocation."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    present = [name for name in required if name in os.environ]
    if present:
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise ValueError(
                "distributed environment is incomplete; present="
                f"{present}, missing={missing}. Launch with torchrun or unset the "
                "partial distributed variables for a one-process run."
            )
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    os.environ.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )


def _validate_world_size(cfg: Config, world_size: int) -> None:
    cfg.validate_world_size(world_size)


def _train(resolved) -> int:
    from accelerate.utils import set_seed

    cfg = resolved.config
    # Make the typed recipe authoritative for the backend's existing FSDP
    # sharding seam in both direct and managed-local worker processes.
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    os.environ["SPECFORGE_FSDP_VERSION"] = cfg.training.fsdp_version
    set_seed(cfg.training.seed)
    if cfg.training.role == "producer":
        # A server-capture/offline-ingest producer owns no trainer process
        # group and must not initialize CUDA merely to publish feature refs.
        from specforge.application import build_application_run

        return build_application_run(resolved).run()

    from specforge.distributed import destroy_distributed, init_distributed

    _bootstrap_single_process_env()
    _validate_world_size(cfg, int(os.environ["WORLD_SIZE"]))
    init_distributed(
        timeout=cfg.training.dist_timeout,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    try:
        import torch.distributed as dist

        _validate_world_size(cfg, dist.get_world_size())
        from specforge.application import build_application_run

        return build_application_run(resolved).run()
    finally:
        destroy_distributed()


def _config_for_role(cfg: Config, role: str) -> Config:
    """Resolve a launch role without changing the persisted run config.

    A shared disaggregated config may contain trainer-only state used by the
    consumer child.  The capture-only producer must ignore that state when the
    launcher derives its role from the shared config.
    """
    from specforge.config import Config

    raw = cfg.model_dump()
    raw["training"]["role"] = role
    disaggregated = raw["deployment"].get("disaggregated")
    if disaggregated is not None and disaggregated.get("managed_local") is not None:
        # This field describes services owned by the parent supervisor.  A role
        # child consumes the already-derived environment and must not attempt to
        # validate or own that stack again.
        disaggregated["managed_local"] = None
    if role == "producer":
        raw["profiling"]["enabled"] = False
    return Config.model_validate(raw)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dspark")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train a draft model from a typed config")
    train.add_argument("-c", "--config", required=True, help="YAML or JSON run config")
    train.add_argument(
        "--role",
        choices=("auto", "all", "producer", "consumer", "both"),
        default="auto",
        help=(
            "launch selection (default: offline local all or online/disaggregated "
            "producer+consumer)"
        ),
    )
    train.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="node-local rank for an explicit multi-node trainer launch",
    )
    train.add_argument(
        "--plan",
        action="store_true",
        help="print the resolved process plan without starting workers",
    )
    train.add_argument(
        "overrides",
        nargs="*",
        help="dotted overrides, e.g. training.learning_rate=1e-4",
    )
    sub.add_parser("algorithms", help="list registered draft methods and feature requirements")
    data = sub.add_parser("data", help="prepare local user data for training")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_prepare = data_sub.add_parser(
        "prepare",
        help="normalize OpenAI messages or ShareGPT rows to canonical JSONL",
    )
    data_prepare.add_argument(
        "--input",
        "--data-path",
        dest="input_path",
        required=True,
        help="local .json or .jsonl input file",
    )
    data_prepare.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        required=True,
        help="fresh canonical training .jsonl output file",
    )
    from specforge.data.prepare import LOCAL_DATA_FORMATS

    data_prepare.add_argument(
        "--format",
        "--data-format",
        dest="data_format",
        choices=LOCAL_DATA_FORMATS,
        default="auto",
        help="input format (default: auto-detect)",
    )
    data_prepare.add_argument(
        "--split-eval",
        action="store_true",
        help="write a deterministic five-percent held-out sibling JSONL",
    )
    data_prepare.add_argument(
        "--eval-output",
        help="optional evaluation output path (requires --split-eval)",
    )
    data_prepare.add_argument(
        "--seed",
        type=int,
        default=42,
        help="split seed (default: 42)",
    )
    data_prepare.add_argument(
        "--max-rows",
        "--sample-size",
        dest="max_rows",
        type=int,
        help="cap input rows before conversion",
    )
    target = sub.add_parser("target", help="inspect and prepare a target without downloading weights")
    target_sub = target.add_subparsers(dest="target_command", required=True)
    inspect_target = target_sub.add_parser(
        "inspect", help="classify architecture/state and suggest DSpark validation gates"
    )
    inspect_target.add_argument("source", help="Hugging Face repo id or local model directory")
    inspect_target.add_argument("--revision", default="main")
    inspect_target.add_argument("--local-only", action="store_true")
    prepare_target = target_sub.add_parser("prepare", help="create a new editable draft-training project from metadata")
    prepare_target.add_argument("source", help="Hugging Face target repo or local model directory")
    prepare_target.add_argument("--output-dir", required=True)
    prepare_target.add_argument("--strategy", default="dspark")
    prepare_target.add_argument("--revision", default="main")
    prepare_target.add_argument("--local-only", action="store_true")
    prepare_data = prepare_target.add_mutually_exclusive_group(required=True)
    prepare_data.add_argument("--train-data", help="raw conversation file; creates GPU0 server/GPU1 trainer config")
    prepare_data.add_argument("--hidden-states", help="offline feature directory, which can be populated later")
    prepare_target.add_argument("--draft-config", help="custom draft JSON or pretrained draft repo")
    prepare_target.add_argument("--draft-checkpoint", help="weights-only initialization; never restores optimizer state")
    prepare_target.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE", help="typed config override, repeatable")
    scaffold_target = target_sub.add_parser(
        "scaffold", help="write a typed offline run config from inspection JSON"
    )
    scaffold_target.add_argument("--inspection", required=True)
    scaffold_target.add_argument("--output", required=True)
    scaffold_target.add_argument("--strategy", default="dspark")
    scaffold_target.add_argument("--data-root", default="./cache/hidden_states")
    scaffold_target.add_argument("--output-root", default="./outputs")
    validate = sub.add_parser("validate", help="run dependency-free parity gates")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    parity = validate_sub.add_parser("parity", help="compare offline and online feature manifests")
    parity.add_argument("--offline", required=True)
    parity.add_argument("--online", required=True)
    tokens = validate_sub.add_parser("tokens", help="compare token-id JSON arrays")
    tokens.add_argument("--expected", required=True)
    tokens.add_argument("--actual", required=True)
    state = validate_sub.add_parser("state", help="compare state snapshot JSON objects")
    state.add_argument("--expected", required=True)
    state.add_argument("--actual", required=True)
    replay = validate_sub.add_parser(
        "replay", help="atomically compare target state snapshots and token prefixes"
    )
    replay.add_argument("--expected", required=True)
    replay.add_argument("--actual", required=True)
    acceptance = validate_sub.add_parser(
        "acceptance", help="validate an aggregated speculative-serving acceptance report"
    )
    acceptance.add_argument("--summary", required=True, help="aggregated serving-gate JSON")
    acceptance.add_argument("--min-train-acc-len", type=float)
    acceptance.add_argument("--min-holdout-acc-len", type=float)
    acceptance.add_argument(
        "--min-holdout-pos2",
        type=float,
        help="minimum acceptance at the second proposed-token position",
    )
    acceptance.add_argument("--expected-train-requests", type=int)
    acceptance.add_argument("--expected-holdout-requests", type=int)
    sweep = sub.add_parser("sweep", help="write a deterministic model-option sweep")
    sweep.add_argument("--model-spec", required=True, help="JSON from target inspect")
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--max-runs", type=int, default=24)
    distributed = sub.add_parser("distributed", help="plan a TorchSpec-style launch")
    distributed_sub = distributed.add_subparsers(dest="distributed_command", required=True)
    distributed_plan = distributed_sub.add_parser("plan")
    distributed_plan.add_argument("--nodes", type=int, default=1)
    distributed_plan.add_argument("--gpus-per-node", type=int, default=1)
    distributed_plan.add_argument("--master-addr", default="127.0.0.1")
    distributed_plan.add_argument("--master-port", type=int, default=29500)
    distributed_plan.add_argument("--fsdp-sharding", default="full_shard")
    distributed_plan.add_argument("--fsdp-version", choices=("v1", "v2"), default="v2")
    distributed_plan.add_argument("--checkpoint-root")
    distributed_plan.add_argument("--max-checkpoints", type=int, default=0)
    export = sub.add_parser(
        "export", help="materialize a runtime checkpoint as a model directory"
    )
    export.add_argument("--to", choices=("hf", "sglang"), required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--draft-config", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--vocab-mapping", default=None)
    export.add_argument(
        "--embedding-source",
        default=None,
        help="target model path supplying a frozen embedding for HF export",
    )
    export.add_argument("--embedding-key", default="model.embed_tokens.weight")
    benchmark = sub.add_parser(
        "benchmark",
        help="benchmark a running SGLang server",
        description=(
            "Measure throughput and optional speculative-decoding telemetry from "
            "a running SGLang server."
        ),
    )
    benchmark.add_argument("--model", required=True)
    benchmark_source = benchmark.add_mutually_exclusive_group(required=True)
    benchmark_source.add_argument(
        "--dataset",
        choices=("gsm8k", "math500", "humaneval", "mbpp", "mt-bench"),
    )
    benchmark_source.add_argument(
        "--data-path",
        dest="messages_jsonl",
        help="local held-out messages JSONL instead of a hosted benchmark preset",
    )
    benchmark.add_argument("--max-new-tokens", type=int, default=2048)
    benchmark.add_argument("--temperature", type=float, default=0.0)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--top-k", type=int, default=1)
    benchmark.add_argument("--max-samples", type=int)
    benchmark.add_argument("--num-prompts", type=int, default=1024)
    benchmark.add_argument("--concurrency", type=int, default=1)
    benchmark.add_argument("--base-url", default="http://127.0.0.1:30000")
    benchmark.add_argument("--timeout-seconds", type=int, default=3600)
    benchmark.add_argument("--enable-thinking", action="store_true")
    benchmark.add_argument("--trust-remote-code", action="store_true")
    benchmark.add_argument("--output-json")
    artifact = sub.add_parser("artifact", help="write reproducible run-artifact metadata")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    artifact_manifest = artifact_sub.add_parser("manifest")
    artifact_manifest.add_argument("--output", required=True)
    artifact_manifest.add_argument("--run-id", required=True)
    artifact_manifest.add_argument("--target-model", required=True)
    artifact_manifest.add_argument("--target-revision", required=True)
    artifact_manifest.add_argument("--tokenizer-version")
    artifact_manifest.add_argument("--backend-revision")
    artifact_manifest.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="local config/checkpoint/data source to record (repeatable)",
    )
    artifact_manifest.add_argument(
        "--validation",
        action="append",
        default=[],
        metavar="JSON_PATH",
        help="validation JSON artifact to hash and record (repeatable)",
    )
    args = parser.parse_args(argv)

    if args.command == "algorithms":
        from specforge.target_project import algorithm_catalog
        print(json.dumps(algorithm_catalog(), indent=2))
        return 0
    if args.command == "data":
        from specforge.data.prepare import prepare_dataset

        try:
            result = prepare_dataset(
                args.input_path,
                args.output_path,
                data_format=args.data_format,
                split_eval=args.split_eval,
                eval_output_path=args.eval_output,
                seed=args.seed,
                max_rows=args.max_rows,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "target":
        if args.target_command == "prepare":
            from specforge.target_project import prepare_project
            try:
                result = prepare_project(
                    args.source, args.output_dir, strategy=args.strategy,
                    revision=args.revision, local_only=args.local_only,
                    train_data=args.train_data, hidden_states=args.hidden_states,
                    draft_config=args.draft_config, draft_checkpoint=args.draft_checkpoint,
                    overrides=args.overrides,
                )
            except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
                parser.error(str(exc))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.target_command == "inspect":
            from specforge.target_inspector import inspect_target as inspect

            try:
                result = inspect(args.source, revision=args.revision, local_only=args.local_only)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
            json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        from specforge.target_spec import TargetSpec

        try:
            with open(args.inspection, encoding="utf-8") as handle:
                inspection = json.load(handle)
            spec = TargetSpec.from_inspection(inspection)
            output, sidecar = spec.write_scaffold(
                args.output,
                strategy=args.strategy,
                data_root=args.data_root,
                output_root=args.output_root,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "config": output,
                    "target_spec": sidecar,
                    "support": spec.support,
                    "required_gates": list(
                        inspection.get("recommendations", {}).get("required_gates", [])
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "validate":
        from specforge.inference.parity import (
            compare_feature_manifests,
            compare_state_replay,
            compare_state_snapshots,
            compare_token_ids,
        )

        def _load(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    return json.load(handle)
            except OSError as exc:
                parser.error(f"cannot read {path!r}: {exc}")
            except json.JSONDecodeError as exc:
                parser.error(f"invalid JSON in {path!r}: {exc}")

        if args.validate_command == "acceptance":
            from specforge.inference.acceptance import validate_acceptance_summary

            result = validate_acceptance_summary(
                _load(args.summary),
                min_train_acc_len=args.min_train_acc_len,
                min_holdout_acc_len=args.min_holdout_acc_len,
                min_holdout_pos2=args.min_holdout_pos2,
                expected_train_requests=args.expected_train_requests,
                expected_holdout_requests=args.expected_holdout_requests,
            )
        elif args.validate_command == "parity":
            result = compare_feature_manifests(_load(args.offline), _load(args.online))
        elif args.validate_command == "tokens":
            result = compare_token_ids(_load(args.expected), _load(args.actual))
        elif args.validate_command == "replay":
            result = compare_state_replay(_load(args.expected), _load(args.actual))
        else:
            result = compare_state_snapshots(_load(args.expected), _load(args.actual))
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        # Validation commands are intended for CI and launch gates: a
        # structurally valid but mismatching artifact must fail the process.
        return 0 if result.get("passed") else 1
    if args.command == "sweep":
        from specforge.sweep import build_sweep

        try:
            with open(args.model_spec, encoding="utf-8") as handle:
                model_spec = json.load(handle)
            rows = build_sweep(model_spec, max_runs=args.max_runs)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"runs": len(rows), "output": args.output}, ensure_ascii=False))
        return 0
    if args.command == "distributed":
        from specforge.torchspec_bridge import TorchSpecLaunch

        launch = TorchSpecLaunch(
            nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
            master_addr=args.master_addr,
            master_port=args.master_port,
            fsdp_sharding=args.fsdp_sharding,
            fsdp_version=args.fsdp_version,
            checkpoint_root=args.checkpoint_root,
            max_checkpoints=args.max_checkpoints,
        )
        json.dump({"argv": launch.torchrun_argv(), **launch.manifest()}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "artifact":
        from specforge.artifacts import build_artifact_manifest, write_artifact_manifest

        sources = {}
        for item in args.source:
            if "=" not in item or item.startswith("="):
                parser.error(f"--source must use NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            if not name or not path or name in sources:
                parser.error(f"invalid or duplicate --source: {item!r}")
            sources[name] = path
        try:
            manifest = build_artifact_manifest(
                run_id=args.run_id,
                target_model=args.target_model,
                target_revision=args.target_revision,
                tokenizer_version=args.tokenizer_version,
                backend_revision=args.backend_revision,
                sources=sources,
                validation_paths=args.validation,
            )
            write_artifact_manifest(args.output, manifest)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(json.dumps({"artifact_hash": manifest["artifact_hash"], "output": args.output}))
        return 0
    if args.command == "train":
        cfg = load_config(args.config, args.overrides)
        from specforge.application import bind_run, resolve_run
        from specforge.launch_plan import build_launch_plan, run_commands

        resolved = resolve_run(cfg)
        plan = build_launch_plan(
            resolved.config,
            algorithm=resolved.algorithm,
            config_path=args.config,
            overrides=args.overrides,
            requested_role=args.role,
            node_rank=args.node_rank,
        )
        if args.plan:
            print(plan.render())
            return 0
        if plan.kind == "worker":
            for key, value in plan.worker_env.items():
                if value is None:
                    # CommandSpec.env contract: None unsets the variable.
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            role_config = _config_for_role(resolved.config, plan.role)
            try:
                with _worker_signal_unwind():
                    _train(bind_run(role_config, resolved.algorithm))
            except _WorkerTermination as received:
                return 128 + received.signum
            return 0
        return run_commands(plan)
    if args.command == "benchmark":
        from specforge.benchmarks.sglang import run

        return run(args)
    if args.to == "hf":
        from specforge.export.to_hf import export_to_hf

        export_to_hf(
            args.checkpoint,
            args.draft_config,
            args.output_dir,
            vocab_mapping_path=args.vocab_mapping,
            embedding_source=args.embedding_source,
            embedding_key=args.embedding_key,
        )
    else:
        if args.embedding_source is not None:
            parser.error("--embedding-source is only valid with --to hf")
        from specforge.export.to_sglang import export_to_sglang

        export_to_sglang(
            args.checkpoint,
            args.draft_config,
            args.output_dir,
            vocab_mapping_path=args.vocab_mapping,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
