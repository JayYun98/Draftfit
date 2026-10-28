"""Bounded real-teacher MiniCPM comparison on an already allocated GPU.

This source-only experiment uses production draft models and objectives. It
never stops other processes, provisions GPUs or replaces the host environment.
Generation timing from the reference decoder is not engine throughput evidence.
"""

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    Qwen3Config,
)

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDSparkModel,
)
from draftfit.algorithms.eagle4.model import Eagle4TrainingModel
from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from draftfit.modeling.draft.dspark import DSparkDraftModel
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from draftfit.modeling.target.target_head import TargetHead
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.optimizer import BF16Optimizer
from draftfit.runtime.contracts import TrainBatch
from draftfit.training.strategies.base import Eagle3TrainStrategy

TARGET_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
LAYERS = [1, 6, 11, 16, 20]
EAGLE_LAYERS = [1, 11, 20]
NAMES = ("dspark", "dflash2", "eagle4")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parameter_hashes(draft):
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, value in draft.named_parameters()
    }


def memory_guard(args):
    free, total = torch.cuda.mem_get_info()
    reserve = getattr(args, "reserve_mib", 1536)
    if free < (args.memory_mib + reserve) * 2**20:
        raise RuntimeError(
            f"Insufficient spare VRAM: {free // 2**20} MiB; need budget plus {reserve} MiB reserve"
        )
    torch.cuda.set_per_process_memory_fraction(args.memory_mib * 2**20 / total)


@torch.inference_mode()
def capture_example(
    target, capture, prompt, reference_ids, max_length, *, source="target"
):
    """Capture target continuations in the same chat mode as generation evaluation."""
    device = target.get_input_embeddings().weight.device
    if not prompt or len(prompt) >= max_length:
        raise ValueError("The prompt must leave room for a continuation")
    if source == "target":
        tokens = torch.tensor([prompt], device=device)
        tokens = target.generate(
            tokens,
            attention_mask=torch.ones_like(tokens),
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_length - len(prompt),
        )
    elif source == "reference":
        if reference_ids[: len(prompt)] != prompt:
            raise ValueError("Reference tokens must preserve the generation prompt")
        tokens = torch.tensor([reference_ids[:max_length]], device=device)
    else:
        raise ValueError("Capture source must be target or reference")
    if tokens.shape[1] <= len(prompt) + 1:
        raise ValueError("The continuation has no supervised next-token positions")
    mask = torch.ones_like(tokens)
    mask[:, : len(prompt)] = 0
    mask[:, -1] = 0
    result = capture.capture(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        loss_mask=mask,
    )
    return {
        "input_ids": tokens.cpu(),
        "loss_mask": mask.cpu(),
        "hidden_states": result.hidden_states.cpu(),
        "target_last_hidden_states": result.last_hidden_states.cpu(),
        "prompt_ids": torch.tensor([prompt]),
    }


def prepare(args):
    from datasets import load_dataset

    root = args.output
    if (root / "capture.json").exists():
        return
    previous = (
        json.loads((args.warm_start / "capture.json").read_text())
        if args.warm_start
        else None
    )
    if previous and any(
        previous[key] != value
        for key, value in {
            "dataset_revision": DATASET_REVISION,
            "target_revision": TARGET_REVISION,
            "max_length": args.max_length,
            "layers": LAYERS,
        }.items()
    ):
        raise ValueError("Warm-start capture settings differ from this comparison")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    capture = OfflineTransformersCapture(target)
    capture.set_capture_layers(LAYERS, capture_method="dspark")
    torch.save(
        {
            "embedding": target.get_input_embeddings().weight.detach().cpu(),
            "head": target.lm_head.weight.detach().cpu(),
        },
        root / "target-tables.pt",
    )
    dataset = "HuggingFaceH4/ultrachat_200k"
    revision = DATASET_REVISION
    identities = {}
    with torch.inference_mode():
        for split, count in (
            ("train_sft", args.train_samples),
            ("test_sft", args.eval_samples),
        ):
            folder = root / split
            folder.mkdir(exist_ok=True)
            rows = load_dataset(
                dataset, split=split, revision=revision, streaming=True, token=False
            )
            selected = []
            for row in rows:
                messages = row["messages"][:2]
                if (
                    len(messages) != 2
                    or messages[0]["role"] != "user"
                    or messages[1]["role"] != "assistant"
                ):
                    continue
                prompt = tokenizer.apply_chat_template(
                    messages[:1],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                )
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=False,
                )
                if len(prompt) > args.max_length - 24 or ids[: len(prompt)] != prompt:
                    continue
                ids = ids[: args.max_length]
                if len(ids) - len(prompt) < 16:
                    continue
                sample = str(row.get("prompt_id", row.get("id", len(selected))))
                if previous:
                    identities_before = previous["splits"][split]
                    if (
                        len(selected) >= len(identities_before)
                        or sample != identities_before[len(selected)]["sample"]
                    ):
                        raise ValueError("Warm-start source conversations changed")
                    old = torch.load(
                        args.warm_start / split / f"{len(selected):05}.pt",
                        weights_only=True,
                        mmap=True,
                        map_location="cpu",
                    )
                    if old["prompt_ids"].tolist() != [prompt]:
                        raise ValueError("Warm-start generation prompt tokens changed")
                    del old
                record = capture_example(
                    target,
                    capture,
                    prompt,
                    ids,
                    args.max_length,
                    source=args.capture_source,
                )
                path = folder / f"{len(selected):05}.pt"
                torch.save(record, path)
                selected.append(
                    {
                        "sample": sample,
                        "sha256": digest(path),
                        "tokens": record["input_ids"].shape[1],
                    }
                )
                if len(selected) % 32 == 0:
                    print(
                        json.dumps({"capture": split, "samples": len(selected)}),
                        flush=True,
                    )
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise RuntimeError(
                    f"Only {len(selected)} usable {split} samples, expected {count}"
                )
            identities[split] = selected
            if previous and len(selected) != len(previous["splits"][split]):
                raise ValueError("Warm-start source conversation count changed")
    write_json(
        root / "capture.json",
        {
            "dataset": dataset,
            "dataset_revision": revision,
            "target_revision": TARGET_REVISION,
            "layers": LAYERS,
            "target_tables_sha256": digest(root / "target-tables.pt"),
            "max_length": args.max_length,
            "continuation_source": args.capture_source,
            "source_capture_sha256": (
                digest(args.warm_start / "capture.json") if args.warm_start else None
            ),
            "generation": (
                {"do_sample": False, "dtype": "bfloat16", "use_cache": True}
                if args.capture_source == "target"
                else None
            ),
            "splits": identities,
        },
    )
    del target, capture
    gc.collect()
    torch.cuda.empty_cache()


def make_models(args, name, tables):
    target = AutoConfig.from_pretrained(args.target, local_files_only=True)
    common = {
        key: getattr(target, key)
        for key in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "max_position_embeddings",
        )
    }
    rope_theta = (getattr(target, "rope_parameters", None) or {}).get(
        "rope_theta", getattr(target, "rope_theta", 5000000.0)
    )
    common.update(
        num_hidden_layers={"dspark": 5, "dflash2": 3, "eagle4": 1}[name],
        tie_word_embeddings=False,
        rope_theta=rope_theta,
        pad_token_id=target.pad_token_id,
        attention_dropout=0.0,
    )
    embed = torch.nn.Embedding.from_pretrained(tables["embedding"], freeze=True).cuda()
    head = torch.nn.Linear(target.hidden_size, target.vocab_size, bias=False)
    head.weight = torch.nn.Parameter(tables["head"].cuda(), requires_grad=False)
    if name == "eagle4":
        config = LlamaConfig(
            **common,
            architectures=["LlamaForCausalLMEagle4"],
            fc_norm=True,
            norm_output=True,
            sliding_window=4096,
            use_sliding_window=True,
            eagle_config={"eagle_aux_hidden_state_layer_ids": EAGLE_LAYERS},
            efficient_lm_head_config={
                "type": "low_rank",
                "rank": 128,
                "num_candidates": 128,
            },
        )
        draft = LlamaForCausalLMEagle4(config).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            draft.embed_tokens.weight.copy_(embed.weight)
            draft.lm_head.weight.copy_(head.weight)
        wrapper = Eagle4TrainingModel(draft, length=7, attention_backend="sdpa")
        target_head = TargetHead(args.target).to(device="cuda", dtype=torch.bfloat16)
        target_head.fc = head
        strategy = Eagle3TrainStrategy(wrapper, target_head=target_head)
    else:
        if name == "dflash2":
            config = DFlash2Config(
                **common,
                architectures=["DFlash2DraftModel"],
                target_layer_ids=LAYERS,
                target_num_hidden_layers=24,
                target_hidden_size=target.hidden_size,
                block_size=7,
                mask_token_id=75982,
                conv_kernel_size=2,
                conv_group_size=16,
                selector_rank=256,
                selector_top_k=16,
                fc_norm=False,
            )
            draft = DFlash2DraftModel(config)
            wrapper_type = OnlineDFlash2Model
        else:
            config = Qwen3Config(**common, architectures=["DSparkDraftModel"])
            config.num_target_layers = 24
            config.block_size = 7
            config.dflash_config = {
                "target_layer_ids": LAYERS,
                "block_size": 7,
                "mask_token_id": 75982,
                "projector_type": "dspark",
                "markov_rank": 256,
                "enable_confidence_head": True,
                "confidence_head_alpha": 1.0,
                "sample_from_anchor": True,
            }
            draft = DSparkDraftModel(config)
            wrapper_type = OnlineDSparkModel
        draft = draft.to(device="cuda", dtype=torch.bfloat16)
        wrapper = wrapper_type(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embed,
            mask_token_id=75982,
            block_size=7,
            attention_backend="sdpa",
            num_anchors=512,
            anchor_sampling="random",
            objective_chunk_blocks=8,
            loss_decay_gamma=7.0 if name == "dflash2" else None,
        )
        strategy = None
    return draft, wrapper, strategy


def forward(name, wrapper, strategy, row):
    tensors = {key: value.cuda() for key, value in row.items() if key != "prompt_ids"}
    if name == "eagle4":
        hidden = tensors["hidden_states"].reshape(
            *tensors["hidden_states"].shape[:-1], len(LAYERS), -1
        )
        hidden = hidden[..., [LAYERS.index(i) for i in EAGLE_LAYERS], :].flatten(-2)
        tensors = {
            "input_ids": tensors["input_ids"],
            "loss_mask": tensors["loss_mask"],
            "hidden_state": hidden,
            "target": tensors["target_last_hidden_states"],
            "attention_mask": torch.ones_like(tensors["input_ids"]),
        }
        return strategy.forward_loss(
            TrainBatch(["sample"], "eagle4", tensors, {"target_repr": "hidden_state"})
        ).loss
    if name == "dflash2":
        tensors.pop("target_last_hidden_states")
    return wrapper(**tensors)[0]


def make_optimizer(args, name, draft, stage):
    candidate = stage == "candidates"
    optimizer = BF16Optimizer(
        draft,
        lr=1e-4 if name == "eagle4" else 6e-4,
        total_steps=args.candidate_steps if candidate else args.steps,
        warmup_ratio=0.015 if name == "eagle4" else 0.04,
        max_grad_norm=0.5 if name == "eagle4" else 1.0,
        offload_master=args.optimizer_cpu_offload,
    )
    if not args.optimizer_cpu_offload:
        for group in optimizer.optimizer.param_groups:
            group.update(fused=True, foreach=False)
    return optimizer


def save_checkpoint(path, draft, optimizer, step, name, stage):
    payload = {
        "strategy": name,
        "global_step": step,
        "stage": stage,
        "draft_state_dict": {
            k: v.detach().cpu() for k, v in draft.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def warm_start(draft, source, name):
    """Load a completed dense draft; the new run owns a fresh optimizer schedule."""
    if source is None:
        return None
    result = json.loads((source / name / "result.json").read_text())
    export = source / name / ("dense-export" if name == "eagle4" else "export")
    weights = {p.name: digest(p) for p in export.glob("*.safetensors")}
    if not weights or (name != "eagle4" and weights != result["export_sha256"]):
        raise ValueError("Warm-start export weights differ from the completed run")
    with torch.random.fork_rng(devices=[]):
        loaded, info = AutoDraftModel.from_pretrained(
            export,
            local_files_only=True,
            device_map="cpu",
            torch_dtype=next(draft.parameters()).dtype,
            output_loading_info=True,
        )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Warm-start export has missing or incompatible tensors")
    if loaded.config.to_dict() != draft.config.to_dict():
        # Serialized configs add metadata; compare the actual architecture fields.
        ignored = {"_name_or_path", "transformers_version", "dtype", "torch_dtype"}
        expected = {k: v for k, v in draft.config.to_dict().items() if k not in ignored}
        actual = {k: v for k, v in loaded.config.to_dict().items() if k not in ignored}
        # AutoDraftModelConfig materializes the full-vocabulary default on reload.
        for config in (expected, actual):
            if config.get("draft_vocab_size") is None:
                config["draft_vocab_size"] = config["vocab_size"]
        if actual != expected:
            raise ValueError("Warm-start draft configuration differs from this arm")
    draft.load_state_dict(loaded.state_dict())
    del loaded
    if name == "eagle4" and parameter_hashes(draft) != json.loads(
        (source / name / "dense-parameter-hashes.json").read_text()
    ):
        raise ValueError("Warm-start EAGLE4 dense tensors changed after training")
    return {
        "export": str(export.resolve()),
        "weights_sha256": weights,
        "previous_dense_steps": result.get(
            "cumulative_dense_steps", result["dense_steps"]
        ),
        "optimizer": "reset; model warm start, not an optimizer-state resume",
    }


def train(args, name):
    root = args.output / name
    root.mkdir(exist_ok=True)
    if (root / "result.json").exists():
        return
    metadata = json.loads((args.output / "capture.json").read_text())
    if (
        metadata["dataset_revision"] != DATASET_REVISION
        or metadata["layers"] != LAYERS
        or metadata["target_revision"] != TARGET_REVISION
        or metadata["max_length"] != args.max_length
        or metadata.get("continuation_source", "reference") != args.capture_source
    ):
        raise ValueError("Capture settings differ from the comparison contract")
    paths = sorted((args.output / "train_sft").glob("*.pt"))
    identities = metadata["splits"]["train_sft"]
    if len(paths) != args.train_samples or len(identities) != len(paths):
        raise ValueError("Training sample count differs from the capture manifest")
    if any(
        digest(path) != identity["sha256"] for path, identity in zip(paths, identities)
    ):
        raise ValueError("Training feature identity changed")
    tables_sha = metadata.get("target_tables_sha256")
    if (
        tables_sha is not None
        and digest(args.output / "target-tables.pt") != tables_sha
    ):
        raise ValueError("Frozen target tables changed")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tables = torch.load(args.output / "target-tables.pt", weights_only=True)
    draft, wrapper, strategy = make_models(args, name, tables)
    initialized_from = warm_start(draft, args.warm_start, name)
    draft.config.save_pretrained(root)
    initial_hashes = parameter_hashes(draft)
    dense_trainable = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    dense_hashes = None
    rows = [torch.load(p, weights_only=True) for p in paths]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    optimizer = make_optimizer(args, name, draft, "draft")
    start = 0
    stage = "draft"
    checkpoint = root / "training_state.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        draft.load_state_dict(saved["draft_state_dict"])
        start, stage = saved["global_step"], saved["stage"]
        if name == "eagle4":
            draft.set_training_stage(stage)
            optimizer = make_optimizer(args, name, draft, stage)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved
    started = time.monotonic()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    total_steps = args.steps + (args.candidate_steps if name == "eagle4" else 0)
    for step in range(start, min(total_steps, args.stop_at_step or total_steps)):
        if name == "eagle4" and step == args.steps and stage == "draft":
            dense_hashes = parameter_hashes(draft)
            write_json(root / "dense-parameter-hashes.json", dense_hashes)
            draft.save_pretrained(root / "dense-export")
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            draft.initialize_candidate_head(randomized=True)
            draft.set_training_stage("candidates")
            stage = "candidates"
            optimizer = make_optimizer(args, name, draft, stage)
        draft.zero_grad(set_to_none=True)
        loss = forward(name, wrapper, strategy, rows[order[step % len(order)]])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {name} loss at step {step}")
        loss.backward()
        norm = optimizer.step()
        if not torch.isfinite(norm):
            raise RuntimeError(f"Non-finite {name} gradient at step {step}")
        losses.append(float(loss.detach()))
        if step == start or (step + 1) % 50 == 0:
            record = {
                "algorithm": name,
                "step": step + 1,
                "stage": stage,
                "loss": losses[-1],
                "grad_norm": float(norm),
                "learning_rate": optimizer.get_learning_rate(),
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            with open(root / "metrics.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if (
            (step + 1) % 1000 == 0
            or step + 1 == total_steps
            or step + 1 == args.stop_at_step
        ):
            save_checkpoint(checkpoint, draft, optimizer, step + 1, name, stage)
    if args.stop_at_step and args.stop_at_step < total_steps:
        return
    export = root / "export"
    draft.save_pretrained(export)
    expected = {key: value.detach().cpu() for key, value in draft.state_dict().items()}
    reloaded = AutoDraftModel.from_pretrained(export)
    reload_equal = all(
        torch.equal(value, reloaded.state_dict()[key].cpu())
        for key, value in expected.items()
    )
    if not reload_equal:
        raise RuntimeError(f"{name} export reload mismatch")
    final_hashes = parameter_hashes(draft)
    if name == "eagle4":
        if dense_hashes is None:
            dense_hashes = json.loads(
                (root / "dense-parameter-hashes.json").read_text()
            )
        frozen_equal = all(
            final_hashes[k] == v
            for k, v in dense_hashes.items()
            if not k.startswith("efficient_lm_head")
        )
        if not frozen_equal:
            raise RuntimeError("Candidate training changed the frozen dense draft")
    else:
        dense_hashes = final_hashes
        frozen_equal = None
    changed_dense = [
        k
        for k in dense_hashes
        if dense_hashes[k] != initial_hashes[k]
        and not k.startswith("efficient_lm_head")
    ]
    if start == 0 and not changed_dense:
        raise RuntimeError("Dense draft parameters did not change")
    result = {
        "algorithm": name,
        "dense_trainable_parameters": dense_trainable,
        "changed_dense_parameters": changed_dense,
        "candidate_stage_preserved_dense_parameters": frozen_equal,
        "optimizer_fused": not args.optimizer_cpu_offload,
        "steps": total_steps,
        "dense_steps": args.steps,
        "warm_start": initialized_from,
        "cumulative_dense_steps": args.steps
        + (initialized_from["previous_dense_steps"] if initialized_from else 0),
        "candidate_steps": args.candidate_steps if name == "eagle4" else 0,
        "optimizer": "BF16Optimizer with FP32 master parameters and cosine warmup",
        "trainable_parameters": sum(
            p.numel() for p in draft.parameters() if p.requires_grad
        ),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "elapsed_seconds": time.monotonic() - started,
        "export_reload_equal": reload_equal,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint_sha256": digest(checkpoint),
        "resumed_from_step": start,
        "export_sha256": {p.name: digest(p) for p in export.glob("*.safetensors")},
    }
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--candidate-steps", type=int, default=1000)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument(
        "--capture-source",
        choices=("target", "reference"),
        default="target",
        help="Use target-generated answers; reference reproduces the old raw-SFT control",
    )
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--optimizer-cpu-offload", action="store_true")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--memory-mib", type=int, default=5600)
    parser.add_argument("--reserve-mib", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--phase", choices=("capture", "train", "all"), default="all")
    parser.add_argument("--algorithm", choices=NAMES)
    args = parser.parse_args()
    if args.steps < 1 or args.candidate_steps < 1:
        parser.error("steps and candidate-steps must be positive")
    if (
        min(args.train_samples, args.eval_samples, args.memory_mib, args.reserve_mib)
        < 1
        or args.max_length < 40
    ):
        parser.error(
            "sample counts and memory must be positive; max-length must be at least 40"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    memory_guard(args)
    invocation = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    invocation["runner_sha256"] = digest(Path(__file__))
    if args.warm_start:
        invocation["warm_start_result_sha256"] = {
            name: digest(args.warm_start / name / "result.json") for name in NAMES
        }
    previous = args.output / "invocation.json"
    if previous.exists():
        recorded = json.loads(previous.read_text())
        changed = [
            key
            for key in invocation
            if key not in {"phase", "algorithm", "stop_at_step"}
            and recorded.get(key) != invocation[key]
        ]
        if changed:
            raise ValueError(
                f"Existing output has different settings: {changed}; use a new directory"
            )
    else:
        write_json(previous, invocation)
    if args.phase in {"capture", "all"}:
        prepare(args)
    if args.phase in {"train", "all"}:
        for name in [args.algorithm] if args.algorithm else NAMES:
            train(args, name)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
