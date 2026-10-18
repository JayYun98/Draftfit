"""Single-process offline or online training of DFlash-family objectives on tied draft weights.

Each architecture retains its real forward (DFlash2 has convolution that the
other two do not), while identically named backbone parameters are the *same*
Parameter objects and receive one combined optimizer update.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from draftfit.family import FamilyStage


_FAMILY = {"dspark", "dflash", "dflash2"}
_GEOMETRY = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "block_size",
    "max_position_embeddings", "rms_norm_eps", "hidden_act", "attention_bias",
    "attention_dropout", "layer_types", "sliding_window", "use_sliding_window",
    "max_window_layers", "rope_parameters", "rope_scaling",
    "rope_is_neox_style", "partial_rotary_factor", "target_hidden_size",
    "draft_vocab_size",
)
_SCHEDULE = (
    "batch_size", "accumulation_steps", "num_epochs", "max_steps",
    "total_steps", "seed", "learning_rate", "lr_scheduler", "warmup_ratio",
    "max_grad_norm", "optimizer_cpu_offload", "save_interval",
    "eval_interval", "log_interval", "max_checkpoints",
    "joint_feature_queue_bytes", "joint_feature_queue_batches", "joint_verification",
    "joint_control", "joint_target_device",
)
_PRIVATE = ("markov_head.", "confidence_head.", "candidate_selector.")


def _backbone_key(name: str) -> bool:
    return not name.startswith(_PRIVATE) and not any(
        part in name for part in (".attention_conv.", ".mlp_conv.")
    )


def _semantic_fingerprint(stages: list[FamilyStage], online_source=None, *, fixed_only=False) -> str:
    """Bind objectives and source identity while allowing a new output root."""
    from draftfit.runtime.data_plane.offline_reader import list_feature_files
    from draftfit.training.provenance import model_source_identity

    records = []
    for stage in stages:
        cfg = stage.config.model_dump(mode="json")
        cfg.pop("output_dir", None)
        cfg.pop("run_id", None)
        # Serialization policy does not change training or exact resume identity.
        cfg["training"].pop("joint_emit_arm_checkpoints", None)
        cfg["model"].pop("draft_model_config", None)
        cfg["training"].pop("resume_from", None)
        verification = cfg["training"].get("joint_verification")
        if verification and not verification.get("existing_drafts"):
            verification.pop("existing_drafts", None)
        # Disabled extensions preserve the fingerprint of pre-queue checkpoints.
        if not cfg["training"].get("joint_feature_queue_bytes"):
            cfg["training"].pop("joint_feature_queue_bytes", None)
            cfg["training"].pop("joint_feature_queue_batches", None)
        if cfg["training"].get("joint_verification") is None:
            cfg["training"].pop("joint_verification", None)
        if cfg["training"].get("joint_control") is None:
            cfg["training"].pop("joint_control", None)
        if cfg["training"].get("joint_target_device") is None:
            cfg["training"].pop("joint_target_device", None)
        if fixed_only:
            # The epoch count is a per-shard data budget; total_steps is the
            # fixed optimizer/scheduler horizon across all phases.
            cfg["training"].pop("num_epochs", None)
            cfg["data"].pop("prompts_path" if cfg["training"].get("joint_online_capture")
                            else "hidden_states_path", None)
        if not cfg["training"].get("joint_online_capture"):
            cfg["training"].pop("joint_online_capture", None)
        records.append({
            "name": stage.name,
            "config": cfg,
            "draft": stage.draft_config.to_dict(),
        })
    base = stages[0].config
    sources = []
    identities = []
    manifests = []
    if online_source is not None:
        sources = online_source.fingerprint_sources()
        if fixed_only:
            sources = {"eval": sources["eval"]}
    for path in (() if online_source is not None else (base.data.hidden_states_path, base.data.eval_hidden_states_path)):
        files = list_feature_files(path)
        directory = Path(path).parent if Path(path).is_file() else Path(path)
        manifest = directory / "capture_manifest.json"
        manifests.append(hashlib.sha256(manifest.read_bytes()).hexdigest()
                         if manifest.is_file() else None)
        if not files:
            raise ValueError(f"joint family has no offline features at {path}")
        identities.append({(os.stat(name).st_dev, os.stat(name).st_ino) for name in files})
        sources.append([(name, os.stat(name).st_size, os.stat(name).st_mtime_ns)
                        for name in files])
    if identities and identities[0] & identities[1]:
        raise ValueError("joint family train and held-out feature files overlap")
    if fixed_only and online_source is None:
        sources, manifests = sources[1:], manifests[1:]
    data = json.dumps({"execution_version": 2, "stages": records, "sources": sources,
                       "target_identity": model_source_identity(base.model.target_model_path),
                       "capture_manifests": manifests},
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def validate_joint_family(stages: list[FamilyStage]) -> None:
    """Cheap preflight; no model, tokenizer, feature tensor or output is opened."""
    if len(stages) < 2 or len({s.name for s in stages}) != len(stages):
        raise ValueError("joint family needs at least two uniquely named arms")
    if any(s.config.training.strategy not in _FAMILY for s in stages):
        raise ValueError("joint family supports dspark, dflash and dflash2 only")
    if any(s.parent is not None for s in stages):
        raise ValueError("joint family cannot use sequential parent handoffs")
    first = stages[0]
    base = first.config
    source = base.model.draft_checkpoint_path
    seeded = [bool(stage.config.model.draft_checkpoint_path) for stage in stages]
    if any(seeded) and not all(seeded):
        raise ValueError("joint family requires all arms to have a seed checkpoint or all scratch")
    if base.training.resume_from:
        raise ValueError("use joint resume_from, not training.resume_from")
    online = base.training.joint_online_capture
    target_device = base.training.joint_target_device
    if target_device is not None:
        import torch
        if (not torch.cuda.is_available()
                or torch.cuda.device_count() <= int(target_device.split(":")[1])):
            raise ValueError(f"joint_target_device {target_device} is not visible")
    if not online and (not base.data.hidden_states_path or not base.data.eval_hidden_states_path):
        raise ValueError("joint family requires train and held-out offline features")
    train_source = base.data.prompts_path if online else base.data.hidden_states_path
    eval_source = base.data.eval_data_path if online else base.data.eval_hidden_states_path
    if Path(train_source).resolve() == Path(eval_source).resolve():
        raise ValueError("joint family train and held-out features must differ")
    for stage in stages:
        cfg = stage.config
        t = cfg.training
        if (cfg.training.joint_online_capture != online
                or cfg.mode != ("online" if online else "offline")
                or cfg.deployment.mode != "local_colocated"):
            raise ValueError("joint family requires local colocated offline training")
        if (cfg.deployment.trainer.nnodes != 1
                or cfg.deployment.trainer.nproc_per_node != 1
                or t.tp_size != 1 or t.sp_ulysses_size != 1 or t.sp_ring_size != 1
                or any(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))):
            raise ValueError("joint family requires one process and one training rank")
        if online and (cfg.data.max_prompts is not None or cfg.model.shard_target_output):
            raise ValueError("joint online does not support max_prompts or shard_target_output")
        if t.dspark_head_only:
            raise ValueError("joint family cannot freeze the shared backbone for head-only training")
        if (t.fsdp_sharding != "NO_SHARD" or t.role != "all"
                or t.attention_backend not in ("sdpa", "eager", "flex_attention")
                or cfg.profiling.enabled or cfg.tracking.report_to != "none"
                or cfg.data.dataloader_num_workers not in (None, 0)):
            raise ValueError("joint family supports unsharded local training without profiling, external tracking or loader workers")
        if t.resume_from:
            raise ValueError("use joint resume_from, not training.resume_from")
        if t.attention_backend == "usp" or cfg.model.use_liger_kernel:
            raise ValueError("joint family does not support USP or Liger kernels")
        if t.accumulation_steps < 1 or t.batch_size < 1:
            raise ValueError("joint family needs positive batch and accumulation sizes")
        if getattr(t, "joint_emit_arm_checkpoints", True) != getattr(
                base.training, "joint_emit_arm_checkpoints", True):
            raise ValueError("joint family has differing training.joint_emit_arm_checkpoints")
        for field in _SCHEDULE:
            if getattr(t, field) != getattr(base.training, field):
                raise ValueError(f"joint family has differing training.{field}")
        for field in ("target_model_path", "target_revision", "torch_dtype",
                      "mask_token_id", "embedding_key", "lm_head_key", "target_backend",
                      "trust_remote_code", "hf_attn_implementation", "target_layer_ids"):
            if getattr(cfg.model, field) != getattr(base.model, field):
                raise ValueError(f"joint family has differing model.{field}")
        for field in ("hidden_states_path", "eval_hidden_states_path", "max_length", "prompts_path", "eval_data_path"):
            if getattr(cfg.data, field) != getattr(base.data, field):
                raise ValueError(f"joint family has differing data.{field}")
        for field in _GEOMETRY:
            if getattr(stage.draft_config, field, None) != getattr(first.draft_config, field, None):
                raise ValueError(f"joint family has incompatible backbone {field}")
        def rope_theta(draft):
            parameters = getattr(draft, "rope_parameters", None)
            if isinstance(parameters, dict) and "rope_theta" in parameters:
                return parameters["rope_theta"]
            return getattr(draft, "rope_theta", 10000.0)
        if rope_theta(stage.draft_config) != rope_theta(first.draft_config):
            raise ValueError("joint family has incompatible backbone rope_theta")
        left = getattr(first.draft_config, "dflash_config", None) or {}
        right = getattr(stage.draft_config, "dflash_config", None) or {}
        for field in ("target_layer_ids", "mask_token_id", "shift_label",
                      "pure_draft_prefix_len", "sliding_window_non_causal"):
            if left.get(field) != right.get(field):
                raise ValueError(f"joint family has incompatible draft {field}")
        if cfg.training.strategy in ("dflash", "dflash2") and right.get("sample_from_anchor") is not False:
            raise ValueError("joint DFlash/DFlash2 requires sample_from_anchor=false")
        if source and left.get("sample_from_anchor", False) and cfg.training.strategy in ("dflash", "dflash2") and not t.retrain_anchor_alignment:
            raise ValueError("DSpark anchor-aligned seed needs retrain_anchor_alignment=true")


def _tie_backbones(bundles, *, scratch: bool) -> int:
    """Tie common parameters by identity, never silently merge unequal seeds."""
    import torch

    owner = bundles[0].draft_model
    reference = dict(owner.named_parameters())
    shared = {name for name in reference if _backbone_key(name)}
    for bundle in bundles[1:]:
        other = dict(bundle.draft_model.named_parameters())
        common = {name for name in other if _backbone_key(name)}
        if not common or common != shared:
            raise ValueError("joint family backbone parameter sets differ")
        for name in sorted(common):
            current, original = other[name], reference[name]
            if current.shape != original.shape or current.dtype != original.dtype:
                raise ValueError(f"joint family backbone parameter {name} differs in shape/dtype")
            if not scratch and not torch.equal(current.detach(), original.detach()):
                raise ValueError(f"joint family seed has unequal shared weight {name}")
            parent, _, leaf = name.rpartition(".")
            setattr(bundle.draft_model.get_submodule(parent), leaf, original)
    return len(shared)


def _rng_state():
    import torch

    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
    }


def _joint_draft_state_dicts(stages, bundles):
    """Transfer each shared tensor once; torch.save preserves CPU storage aliases.

    keep_vars retains Parameter identity across tied arms. Do not merge equal
    private parameters or independent mutable buffers by value or key name.
    """
    cpu_by_identity = {}
    states = {}
    for stage, bundle in zip(stages, bundles):
        state = {}
        for name, tensor in bundle.draft_model.state_dict(keep_vars=True).items():
            identity = id(tensor)
            if identity not in cpu_by_identity:
                cpu_by_identity[identity] = tensor.detach().cpu()
            state[name] = cpu_by_identity[identity]
        states[stage.name] = state
    return states


def _restore_rng(state):
    import torch

    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])


def _policy_best_steps(control):
    """Retention exceptions from a caller-authenticated control receipt."""
    steps = set()
    for decision in control["decisions"].values():
        for metric in ("loss", "acceptance", "speed"):
            best = decision.get("best", {}).get(metric)
            if best is not None:
                step = best["artifact"]["step"]
                if type(step) is not int or not 0 <= step <= control["step"]:
                    raise ValueError("joint policy best checkpoint step is invalid")
                steps.add(step)
    return steps


def _atomic_torch_save(payload, path: Path):
    import torch

    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _make_loader(path, stage, *, needs_dspark: bool, epoch: int, shuffle: bool, drop_last: bool):
    from draftfit.algorithms.common.dflash_family_data import (
        build_collator, build_dspark_collator, normalize_offline_sample,
        normalize_dspark_offline_sample,
    )
    from draftfit.launch import _shard_offline_refs
    from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
    from draftfit.runtime.data_plane.feature_store import LocalFeatureStore
    from draftfit.runtime.data_plane.offline_reader import OfflineManifestReader

    cfg = stage.config
    keys = ("input_ids", "loss_mask", "hidden_states")
    if needs_dspark:
        keys += ("target_last_hidden_states",)
    strategy = "dspark" if needs_dspark else "dflash"
    reader = OfflineManifestReader(
        path, run_id="joint", strategy=strategy, feature_keys=keys,
        target_repr="hidden_state" if needs_dspark else None,
        max_len=cfg.data.max_length,
    )
    refs = reader.read()
    if not refs:
        raise ValueError(f"joint family has no offline features at {path}")
    if shuffle:
        refs = _shard_offline_refs(refs, use_usp_preprocess=False,
                                   seed=cfg.training.seed, epoch=epoch)
    return FeatureDataLoader(
        LocalFeatureStore("joint"), refs=refs,
        batch_size=cfg.training.batch_size,
        collate_fn=build_dspark_collator() if needs_dspark else build_collator(),
        per_sample_transform=(
            (lambda raw: normalize_dspark_offline_sample(raw, cfg.data.max_length))
            if needs_dspark else
            (lambda raw: normalize_offline_sample(raw, cfg.data.max_length))
        ),
        drop_last=drop_last, strategy=strategy, num_workers=0,
    ), len(refs)


def _evaluate(stages, strategies, bundles, online_source=None):
    import torch
    from draftfit.runtime.contracts import TrainBatch
    from draftfit.training.evaluation import Evaluator
    from draftfit.training.strategies.base import StepOutput

    def compact(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {name: compact(item) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(compact(item) for item in value)
        return value

    saved_rng = _rng_state()
    needs_dspark = any(s.config.training.strategy == "dspark" for s in stages)
    if online_source is not None:
        loader, _ = online_source.loader(stages[0], epoch=0, shuffle=False,
                                         drop_last=False, evaluation=True)
    else:
        loader, _ = _make_loader(stages[0].config.data.eval_hidden_states_path,
                                 stages[0], needs_dspark=needs_dspark,
                                 epoch=0, shuffle=False, drop_last=False)
    for bundle in bundles:
        bundle.model.eval()
    # Evaluator owns denominator-weighted loss, accuracy and family diagnostic
    # reduction. Keep only detached scalar/position terms and a token count per
    # batch; the full feature tensors are discarded after each shared read.
    records = {stage.name: [] for stage in stages}
    try:
        with torch.no_grad():
            for batch in loader:
                token_count = batch.tensors["loss_mask"].sum().detach().cpu().reshape(1)
                for stage, strategy in zip(stages, strategies):
                    out = strategy.forward_loss(batch)
                    summary = StepOutput(
                        loss=compact(out.loss), metrics=compact(out.metrics),
                        ratio_metrics=compact(out.ratio_metrics),
                        loss_terms=compact(out.loss_terms),
                    )
                    records[stage.name].append(TrainBatch(
                        sample_ids=[], strategy=stage.config.training.strategy,
                        tensors={"loss_mask": token_count},
                        metadata={"joint_output": summary},
                    ))
    finally:
        for bundle in bundles:
            bundle.model.train()
        _restore_rng(saved_rng)
    return {name: Evaluator().run(lambda item: item.metadata["joint_output"], rows)
            for name, rows in records.items()}


def _backward_window(stages, strategies, bundles, batches):
    """Mean of arm objectives, each normalized across its entire window.

    Hold one CPU feature window, FP32 arm and combined gradient buffers. An
    arm's graph is released after each microbatch. Private gradient buffers can
    be transferred to the combined result without a copy. Sharing parameters does not mean sharing
    objective denominators (DSpark has a different loss contract).
    """
    import math
    import torch
    combined = {}
    metrics = {}
    for stage, strategy, bundle in zip(stages, strategies, bundles):
        numerator_total = denominator_total = accuracy_total = 0.0
        uses_terms = None
        arm_gradients = {}
        for batch in batches:
            out = strategy.forward_loss(batch)
            has_terms = out.loss_terms is not None
            if uses_terms is not None and uses_terms != has_terms:
                raise ValueError("joint objective changed its loss normalization within a window")
            uses_terms = has_terms
            if has_terms:
                numerator, denominator = out.loss_terms
                if numerator.numel() != 1 or denominator.numel() != 1:
                    raise ValueError("joint loss_terms must contain scalar tensors")
                denominator = float(denominator.detach())
                if not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError("joint loss denominator must be finite and positive")
            else:
                numerator, denominator = out.loss, 1.0
            value = float(numerator.detach())
            if not math.isfinite(value):
                raise ValueError("joint loss numerator must be finite")
            numerator_total += value
            denominator_total += denominator
            accuracy = out.metrics.get("accuracy")
            accuracy_total += float(accuracy) if accuracy is not None else 0.0
            numerator.backward()
            # BF16 .grad accumulation can stop incorporating small increments
            # in long windows. Widen each microgradient before adding it.
            for parameter in bundle.draft_model.parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().to(torch.float32)
                parameter.grad = None
                if parameter in arm_gradients:
                    arm_gradients[parameter].add_(gradient)
                else:
                    arm_gradients[parameter] = gradient
        scale = 1.0 / (len(stages) * denominator_total)
        for parameter, gradient in arm_gradients.items():
            gradient.mul_(scale)
            if parameter in combined:
                combined[parameter].add_(gradient)
            else:
                combined[parameter] = gradient
        metrics[stage.name] = [numerator_total / denominator_total,
                               accuracy_total / len(batches)]
    for parameter, gradient in combined.items():
        parameter.grad = gradient.to(parameter.dtype)
    return metrics


def run_joint_family(root, stages: list[FamilyStage], *, resume_from=None,
                     stop_at_step=None, deadline_unix=None, data_phase=0,
                     milestone_observer=None, milestone_runtime=False) -> dict:
    """Train all arms from one shared batch stream and one optimizer.

    `root` may already contain the per-arm configs written by the family CLI.
    Resume accepts a previous root/joint-stepN.pt and writes to a *new* root.
    Opt-in milestone_runtime supplies a bounded observer with a model-memory
    handoff factory. The default resident native verification is unchanged.
    """
    if type(milestone_runtime) is not bool or (milestone_runtime and milestone_observer is None):
        raise ValueError("milestone_runtime requires a milestone observer and explicit boolean")
    if deadline_unix is not None and (isinstance(deadline_unix, bool)
                                      or not isinstance(deadline_unix, Real)
                                      or not math.isfinite(deadline_unix)):
        raise ValueError("deadline_unix must be a finite Unix timestamp")
    if type(data_phase) is not int or data_phase < 0 or (data_phase and resume_from is None):
        raise ValueError("data_phase must be a nonnegative integer; later phases require resume_from")
    validate_joint_family(stages)
    import torch
    from torch import nn
    from draftfit.algorithms.builtin import builtin_algorithm_registry
    from draftfit.optimizer import BF16Optimizer
    from draftfit.training.assembly import build_model_bundle
    from draftfit.training.checkpoint import CheckpointManager
    from draftfit.training.schedule import resolve_total_steps, validate_fixed_accumulation_plan

    root = Path(root)
    cfg = stages[0].config
    if milestone_runtime and not cfg.training.joint_online_capture:
        raise ValueError("milestone_runtime requires an online frozen target")
    needs_dspark = any(s.config.training.strategy == "dspark" for s in stages)
    online_source = None
    if cfg.training.joint_online_capture:
        from draftfit.training.joint_online import JointOnlineSource
        online_source = JointOnlineSource(stages[0], needs_dspark=needs_dspark)

    try:
        def training_loader(epoch):
            if online_source is not None:
                return online_source.loader(stages[0], epoch=epoch, shuffle=True, drop_last=True)
            return _make_loader(cfg.data.hidden_states_path, stages[0],
                                needs_dspark=needs_dspark, epoch=epoch, shuffle=True, drop_last=True)

        _, num_samples = training_loader(0)
        t = cfg.training
        if milestone_observer is not None and (not callable(milestone_observer)
                or t.joint_verification is None or t.joint_control is not None):
            raise ValueError("milestone_observer requires generation verification and owns policy instead of joint_control")
        validate_fixed_accumulation_plan(
            num_samples=num_samples, batch_size=t.batch_size,
            accumulation_steps=t.accumulation_steps, num_epochs=t.num_epochs,
            max_steps=t.max_steps,
        )
        available_steps = ((num_samples // t.batch_size) * t.num_epochs) // t.accumulation_steps
        if available_steps < 1:
            raise ValueError("joint family data produces no optimizer step")
        stop_step = t.max_steps or available_steps
        if t.max_steps is not None and t.max_steps > available_steps:
            raise ValueError(f"joint family max_steps={t.max_steps} exceeds {available_steps} available optimizer steps")
        horizon = resolve_total_steps(
            total_steps=t.total_steps, max_steps=t.max_steps, num_samples=num_samples,
            batch_size=t.batch_size, accumulation_steps=t.accumulation_steps,
            num_epochs=t.num_epochs,
        )
        # Operational boundary only: keep the full schedule/fingerprint for exact resume.
        if stop_at_step is not None and (type(stop_at_step) is not int or stop_at_step < 1):
            raise ValueError("stop_at_step must be within the configured training horizon")
        fingerprint = _semantic_fingerprint(stages, online_source)
        fixed_fingerprint = _semantic_fingerprint(stages, online_source, fixed_only=True)
        torch.manual_seed(t.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(t.seed)
        random.seed(t.seed)
        registry = builtin_algorithm_registry()
        bundles = []
        shared_count = 0
        # Per-arm exports preserve private heads. Their loaded shared tensors
        # must match exactly before creating the optimizer or running training.
        scratch = not all(stage.config.model.draft_checkpoint_path for stage in stages)
        for stage in stages:
            bundle = build_model_bundle(stage.config,
                                        algorithm=registry.resolve(stage.config.training.strategy))
            if bundles:
                shared_count = _tie_backbones(
                    (bundles[0], bundle), scratch=scratch,
                )
                bundle.model.lm_head = bundles[0].model.lm_head
                bundle.model.embed_tokens = bundles[0].model.embed_tokens
                bundle.target_head = bundles[0].target_head
            bundles.append(bundle)
        existing_draft_sources = {}
        if t.joint_verification and t.joint_verification.existing_drafts:
            from draftfit.training.joint_verification import preflight_existing_drafts
            settings = t.joint_verification.model_dump()
            if deadline_unix is not None: settings["deadline_unix"] = deadline_unix
            existing_draft_sources = preflight_existing_drafts(stages, bundles, settings)
        if not shared_count:
            raise ValueError("joint family has no shared trainable backbone parameters")
        for bundle in bundles:
            bundle.model.train()
        strategies = [registry.resolve(stage.config.training.strategy).providers.step.build(bundle.model)
                      for stage, bundle in zip(stages, bundles)]
        drafts = nn.ModuleList([bundle.draft_model for bundle in bundles])
        unique = list(drafts.parameters())
        if len(unique) != len({id(p) for p in unique}):
            raise AssertionError("optimizer contains duplicate parameters")
        optimizer = BF16Optimizer(
            drafts, lr=t.learning_rate, max_grad_norm=t.max_grad_norm,
            total_steps=horizon, warmup_ratio=t.warmup_ratio,
            lr_scheduler=t.lr_scheduler, offload_master=t.optimizer_cpu_offload,
        )
        from draftfit.algorithms.common.providers import MODEL_PROVENANCE_CONTRACT_KEY
        provenance = dict(bundles[0].strategy_kwargs.resume_contract[MODEL_PROVENANCE_CONTRACT_KEY])
        target_revision = provenance.get("target_revision")
        start_step = start_epoch = start_batch = phase_start_step = 0
        policy_history = {}
        policy_best_steps = set()
        if resume_from is not None:
            state = torch.load(resume_from, map_location="cpu", weights_only=True)
            if state.get("existing_draft_sources", {}) != existing_draft_sources:
                raise ValueError("existing comparison draft changed since checkpoint")
            if state.get("schema") != "draftfit-joint-v2":
                raise ValueError("joint exact resume requires draftfit-joint-v2; use a per-arm export for weights-only continual from older runs")
            if state.get("target_revision") != target_revision:
                raise ValueError("joint checkpoint resolved target revision differs")
            previous_phase = state.get("data_phase", 0)
            new_phase = data_phase == previous_phase + 1
            if data_phase != previous_phase and not new_phase:
                raise ValueError("joint checkpoint data_phase must stay the same or advance by one")
            if new_phase:
                if state.get("fixed_fingerprint") != fixed_fingerprint:
                    raise ValueError("joint checkpoint fixed eval, architecture or objectives differ")
                if state.get("fingerprint") == fingerprint:
                    raise ValueError("new data_phase requires a different training shard")
            elif state.get("fingerprint") != fingerprint:
                raise ValueError("joint checkpoint does not match these objectives or feature sources")
            if state.get("horizon") != horizon or state.get("accumulation_steps") != t.accumulation_steps:
                raise ValueError("joint checkpoint schedule differs")
            if set(state.get("drafts", {})) != {stage.name for stage in stages}:
                raise ValueError("joint checkpoint arm set differs")
            first_state = state["drafts"][stages[0].name]
            shared_names = {name for name in first_state if _backbone_key(name)}
            for stage in stages[1:]:
                arm_state = state["drafts"][stage.name]
                if any(name not in arm_state or not torch.equal(first_state[name], arm_state[name])
                       for name in shared_names):
                    raise ValueError("joint checkpoint contains unequal shared backbone weights")
            start_step = int(state["step"])
            start_epoch = int(state["epoch"])
            start_batch = int(state["next_batch"])
            phase_start_step = int(state.get("phase_start_step", 0))
            prior_batches = state.get("batches_per_epoch") if new_phase else num_samples // t.batch_size
            prior_epochs = state.get("phase_num_epochs") if new_phase else t.num_epochs
            if (type(prior_batches) is not int or prior_batches < 1
                    or type(prior_epochs) is not int or prior_epochs < 1
                    or not 0 <= phase_start_step <= start_step <= horizon
                    or not 0 <= start_epoch <= prior_epochs
                    or not 0 <= start_batch < prior_batches
                    or (start_epoch == prior_epochs and start_batch != 0)
                    or start_epoch * prior_batches + start_batch
                       != (start_step - phase_start_step) * t.accumulation_steps):
                raise ValueError("joint checkpoint cursor is invalid")
            optimizer_state = state.get("optimizer")
            if not isinstance(optimizer_state, dict) or optimizer_state.get("fp32_params") is None:
                raise ValueError("joint checkpoint requires fp32_params for exact resume")
            for stage, bundle in zip(stages, bundles):
                bundle.draft_model.load_state_dict(state["drafts"][stage.name], strict=True)
            optimizer.load_state_dict(optimizer_state)
            _restore_rng(state["rng"])
            policy_history = state.get("policy_history", {})
            policy_best_steps = set(state.get("policy_best_steps", []))
            if t.joint_control:
                # A checkpoint may precede a failed evaluation and have no
                # same-step receipt. Keep its last authenticated selection.
                policy_best_steps = set(state.get("policy_best_steps", []))
                sidecar = Path(resume_from).parent / f"control-step{start_step}.json"
                if sidecar.exists():
                    control_state = json.loads(sidecar.read_text())
                    digest = hashlib.sha256()
                    with Path(resume_from).open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if control_state.get("artifact_sha256") != digest.hexdigest():
                        raise ValueError("joint control history checkpoint hash mismatch")
                    policy_history = control_state["history"]
                    policy_best_steps = _policy_best_steps(control_state)
            if new_phase:
                phase_start_step = start_step
                start_epoch = start_batch = 0
        phase_capacity = (num_samples // t.batch_size * t.num_epochs) // t.accumulation_steps
        if data_phase:
            stop_step = min(horizon, phase_start_step + phase_capacity)
        if stop_at_step is not None:
            if not start_step <= stop_at_step <= stop_step:
                raise ValueError("stop_at_step must be within the configured training horizon")
            stop_step = stop_at_step
        policy_next_verify = None
        if t.joint_control:
            from draftfit.training.joint_control import admit_initial
            budget = json.loads(Path(t.joint_control.budget_path).read_text())
            stop_step, policy_next_verify = admit_initial(t.joint_control.model_dump(),
                budget, start_step, stop_step, t.joint_verification.interval)

        root.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            output = Path(stage.config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            stage.draft_config.to_json_file(str(output / "config.json"))
        results = [{"name": stage.name, "strategy": stage.config.training.strategy,
                    "output_dir": stage.config.output_dir, "status": "running"}
                   for stage in stages]

        from draftfit.training.joint_monitor import JointMonitor
        monitor = JointMonitor(root, stages, start_step, resume_from)

        def checkpoint(step, epoch, next_batch, eval_metrics):
            # Arm checkpoints use the existing weights-only reader/export schema.
            for stage, bundle, result in zip(stages, bundles, results):
                output = Path(stage.config.output_dir)
                if getattr(t, "joint_emit_arm_checkpoints", True):
                    state = {
                        "draft_state_dict": {k: v.detach().cpu() for k, v in bundle.draft_model.state_dict().items()},
                        "global_step": step, "epoch": epoch, "epoch_batch": next_batch,
                        "epoch_samples": next_batch * t.batch_size,
                        "strategy": stage.config.training.strategy, "run_id": stage.name,
                        "checkpoint_schema_version": 1, "world_size": 1,
                        "dataset_size": num_samples, "batch_size": t.batch_size,
                        "accumulation_steps": t.accumulation_steps, "num_epochs": t.num_epochs,
                        "effective_total_steps": horizon,
                        "tp_size": 1, "sp_ulysses_size": 1, "sp_ring_size": 1,
                        **dict(bundle.strategy_kwargs.resume_contract),
                    }
                    manager = CheckpointManager(str(output), stage.name, max_checkpoints=t.max_checkpoints)
                    directory = Path(manager.save(state, step))
                    stage.draft_config.to_json_file(str(directory / "config.json"))
                    result["latest"] = str(output / f"{stage.name}-latest")
                    result["arm_checkpoint"] = str(directory)
                result["global_step"] = step
                if stage.name in eval_metrics:
                    result["metrics"] = eval_metrics[stage.name]
            joint = {
                "schema": "draftfit-joint-v2", "fingerprint": fingerprint,
                "fixed_fingerprint": fixed_fingerprint, "data_phase": data_phase,
                "phase_start_step": phase_start_step,
                "batches_per_epoch": num_samples // t.batch_size,
                "phase_num_epochs": t.num_epochs,
                "target_revision": target_revision,
                "existing_draft_sources": existing_draft_sources,
                "step": step, "epoch": epoch, "next_batch": next_batch,
                "horizon": horizon, "accumulation_steps": t.accumulation_steps,
                "drafts": _joint_draft_state_dicts(stages, bundles),
                "optimizer": optimizer.state_dict(), "rng": _rng_state(),
                "consumed_samples": (step - phase_start_step) * t.batch_size * t.accumulation_steps,
                "policy_history": policy_history,
                "policy_best_steps": sorted(policy_best_steps),
            }
            joint_path = root / f"joint-step{step}.pt"
            _atomic_torch_save(joint, joint_path)
            if milestone_observer is not None:
                # Durable byte-integrity receipt precedes observer/evaluation;
                # even an unobserved seed checkpoint can be safely resumed.
                with joint_path.open("rb") as stream:
                    checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                checksum = joint_path.with_suffix(".sha256")
                temporary_checksum = checksum.with_suffix(".sha256.tmp")
                with temporary_checksum.open("w") as stream:
                    stream.write(checkpoint_sha + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_checksum.replace(checksum)
            for result in results:
                result.update(joint_checkpoint=str(joint_path),
                              emit_arm_checkpoints=getattr(t, "joint_emit_arm_checkpoints", True))
            if t.max_checkpoints:
                saved = sorted(root.glob("joint-step*.pt"),
                               key=lambda p: int(p.stem.removeprefix("joint-step")))
                protected = Path(resume_from).resolve() if resume_from else None
                for stale in saved[:-t.max_checkpoints]:
                    if (stale.resolve() != protected
                            and int(stale.stem.removeprefix("joint-step")) not in policy_best_steps):
                        stale.unlink()
                        stale.with_suffix(".sha256").unlink(missing_ok=True)
            monitor.checkpoint(step, results)
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")

        def deadline_reached():
            return deadline_unix is not None and time.time() >= deadline_unix

        # An expired budget still produces a resumable seed checkpoint.
        initial_metrics = ({} if deadline_reached() or (t.joint_control and start_step >= stop_step) else
                           _evaluate(stages, strategies, bundles, online_source))
        def record_eval_capture(step):
            if online_source is not None:
                monitor.emit("evaluation_capture", step, stages[0].name,
                    capture_calls=online_source.capture_calls,
                    capture_seconds=online_source.capture_seconds,
                    cuda_capture_peak_allocated_bytes=online_source.capture_peak_allocated_bytes,
                    cuda_capture_peak_reserved_bytes=online_source.capture_peak_reserved_bytes)
        if initial_metrics:
            record_eval_capture(start_step)
            for stage in stages:
                monitor.emit("evaluation", start_step, stage.name, **initial_metrics[stage.name])

        stopped_for_deadline = deadline_reached() and start_step < stop_step
        if stopped_for_deadline:
            for result in results:
                result.update(status="stopped", reason="deadline")
        # The source checkpoint remains the durable retry point until this run
        # completes an optimizer step. Avoid duplicating its large weights at start.
        initial_saved = resume_from is None or stopped_for_deadline or start_step >= stop_step
        if initial_saved:
            checkpoint(start_step, start_epoch, start_batch, initial_metrics)

        stopped_for_policy = bool(t.joint_control and start_step >= stop_step)
        observer_reason = None

        def notify_observer(report, metrics):
            nonlocal policy_best_steps, observer_reason
            if milestone_observer is None:
                return False
            runtime = {}
            if milestone_runtime:
                from draftfit.training.serving_memory import release_training_models
                # Capture is synchronous and all backward work is committed at
                # this boundary. Queued CPU features and cursor remain intact.
                runtime["memory_handoff"] = lambda: release_training_models(
                    bundles, optimizer, online_source._teacher)
            decision = milestone_observer(step=report["step"],
                checkpoint=Path(report["checkpoint_path"]), report=report,
                metrics=metrics, stages=stages, **runtime)
            if not isinstance(decision, dict) or type(decision.get("stop")) is not bool:
                raise ValueError("milestone observer must return a boolean stop decision")
            protected = decision.get("protect_steps", [])
            if any(type(n) is not int or not 0 <= n <= report["step"] for n in protected):
                raise ValueError("milestone observer returned invalid protected checkpoints")
            policy_best_steps = set(protected)
            observer_reason = decision.get("reason")
            monitor.emit("milestone_observer", report["step"], stages[0].name,
                         decision=decision, checkpoint_sha256=report["artifact_sha256"])
            return decision["stop"]

        # Replay against the loaded teacher, settings and held-out prompts before
        # another update. A matching checkpoint SHA authenticates weights, not
        # a sidecar's runtime, prompt list or measurements. Never read that cache.
        if resume_from is not None and milestone_observer is not None and not deadline_reached():
            from draftfit.training.joint_verification import evaluate_generation
            source_checkpoint = Path(resume_from).resolve()
            with source_checkpoint.open("rb") as stream:
                source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            settings = t.joint_verification.model_dump()
            if deadline_unix is not None:
                settings["deadline_unix"] = deadline_unix
            # evaluate_generation suspends/restores optimizer, gradients and RNG
            # even on failure. Do not perform an optimizer update during replay.
            report = evaluate_generation(stages, bundles, online_source, optimizer, start_step, settings)
            prompt_count = min(settings["max_prompts"], len(online_source._rows[True]))
            expected_pairs = {(trial, index) for trial in range(settings["trials"])
                              for index in range(prompt_count)}

            def paired_runs(arm):
                rows = arm["runs"]
                if not isinstance(rows, list) or len(rows) != len(expected_pairs):
                    raise ValueError("replayed generation has missing/extra requests")
                indexed = {}
                for row in rows:
                    trial, index = row["trial"], row["row_index"]
                    if type(trial) is not int or type(index) is not int:
                        raise ValueError("replayed generation has invalid request indices")
                    key = trial, index
                    if key not in expected_pairs or key in indexed:
                        raise ValueError("replayed generation has duplicate/unmatched requests")
                    ids, count = row["token_ids"], row["generated_tokens"]
                    if (not isinstance(ids, list) or type(count) is not int
                            or not 0 < count <= settings["max_new_tokens"] or count != len(ids)
                            or any(type(token) is not int or token < 0 for token in ids)):
                        raise ValueError("replayed generation has invalid output token counts")
                    indexed[key] = row
                return indexed

            try:
                if (report["schema"] != "draftfit-joint-generation-v1"
                        or type(report["step"]) is not int or report["step"] != start_step
                        or report["production_engine_speed"] is not False):
                    raise ValueError("replayed generation has invalid native identity")
                prompts = report["bindings"]["prompts"]
                if (len(prompts) != prompt_count or any(type(p["row_index"]) is not int
                        or p["row_index"] != i for i, p in enumerate(prompts))):
                    raise ValueError("replayed generation has invalid prompt indices")
                names = [candidate["name"] for candidate in report["candidates"]]
                if len(names) != len(stages) or set(names) != {stage.name for stage in stages}:
                    raise ValueError("replayed generation has missing/duplicate/extra candidates")
                baseline = paired_runs(report["baseline"])
                for candidate in report["candidates"]:
                    rows = paired_runs(candidate)
                    if (candidate["tokenparity"] is not True or any(
                            row.get("tokenparity") is not True
                            or row["token_ids"] != baseline[key]["token_ids"]
                            for key, row in rows.items())):
                        raise ValueError("replayed generation token parity failed")
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError("replayed generation report is malformed") from exc
            if report["tokenparity"] is not True:
                raise ValueError("resumed generation token parity failed; checkpoint preserved")
            report.update(checkpoint_path=str(source_checkpoint), artifact_sha256=source_sha,
                          capture_fingerprint=fixed_fingerprint)
            (root / f"verification-step{start_step}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n")
            stopped_for_policy = notify_observer(report, initial_metrics)
        step = start_step
        cursor_epoch, cursor_batch = start_epoch, start_batch
        window = []
        window_started = time.perf_counter()
        device = next(bundles[0].draft_model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        last_saved = start_step if initial_saved else None
        if resume_from is not None and start_step >= stop_step:
            metrics = initial_metrics
            for stage in stages:
                with (Path(stage.config.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": start_step, **metrics.get(stage.name, {})},
                                            allow_nan=False) + "\n")
            last_saved = start_step
        for epoch in range(start_epoch, t.num_epochs):
            if step >= stop_step or stopped_for_deadline or stopped_for_policy:
                break
            if deadline_reached():
                stopped_for_deadline = True
                break
            loader, _ = training_loader(epoch)
            offset = start_batch if epoch == start_epoch else 0
            loader.seek(offset)
            batches = iter(loader)
            for batch_index in range(offset, num_samples // t.batch_size):
                if not window and deadline_reached():
                    stopped_for_deadline = True
                    break
                batch = next(batches)
                window.append(batch)
                if len(window) < t.accumulation_steps:
                    continue
                device = next(bundles[0].draft_model.parameters()).device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                step_lr = optimizer.optimizer.param_groups[0]["lr"]
                tokens = sum(float(b.tensors["loss_mask"].sum()) for b in window)
                accum_metrics = _backward_window(stages, strategies, bundles, window)
                grad_norm = float(optimizer.step())
                if online_source is not None:
                    online_source.consume(window)
                window.clear()
                del batch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                train_seconds = time.perf_counter() - started
                step += 1
                stopped_for_deadline = step < stop_step and deadline_reached()
                wall_seconds = time.perf_counter() - window_started
                if online_source is not None:
                    monitor.emit("capture", step, stages[0].name,
                        capture_calls=online_source.capture_calls,
                        capture_seconds=online_source.capture_seconds,
                        cuda_capture_peak_allocated_bytes=online_source.capture_peak_allocated_bytes,
                        cuda_capture_peak_reserved_bytes=online_source.capture_peak_reserved_bytes,
                        queue=(online_source.queue_metrics() if t.joint_feature_queue_bytes else None),
                        consumed_samples=online_source.consumed_samples,
                        step_with_capture_seconds=wall_seconds,
                        samples_per_second_with_capture=t.batch_size*t.accumulation_steps/wall_seconds)
                for stage in stages:
                    monitor.train(step, stage.name, final=(step >= stop_step or
                                  stopped_for_deadline or deadline_unix is not None),
                        train_loss=accum_metrics[stage.name][0],
                        train_accuracy=accum_metrics[stage.name][1], grad_norm=grad_norm,
                        joint_step_seconds=train_seconds,
                        joint_samples_per_second=t.batch_size * t.accumulation_steps / train_seconds,
                        joint_valid_tokens_per_second=tokens / train_seconds,
                        learning_rate=step_lr,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch == num_samples // t.batch_size:
                    next_epoch, next_batch = epoch + 1, 0
                cursor_epoch, cursor_batch = next_epoch, next_batch
                do_eval = not stopped_for_deadline and bool(
                    (t.eval_interval and step % t.eval_interval == 0) or step >= stop_step)
                do_verify = not stopped_for_deadline and t.joint_verification is not None and (
                    (step >= policy_next_verify if policy_next_verify is not None
                     else step % t.joint_verification.interval == 0) or step >= stop_step)
                do_eval = do_eval or do_verify
                # Capture/evaluation can fail or exhaust the device. Commit the
                # consumed cursor and optimizer before entering that boundary.
                save_due = bool((t.save_interval and step % t.save_interval == 0)
                                or step >= stop_step or stopped_for_deadline or do_eval)
                if save_due:
                    checkpoint(step, next_epoch, next_batch, {})
                    last_saved = step
                metrics = _evaluate(stages, strategies, bundles, online_source) if do_eval else {}
                if do_eval:
                    record_eval_capture(step)
                for name, values in metrics.items():
                    monitor.emit("evaluation", step, name, **values)
                for stage in stages:
                    if step % t.log_interval == 0 or do_eval:
                        row = {"step": step,
                               "train/loss": accum_metrics[stage.name][0],
                               "train/acc": accum_metrics[stage.name][1],
                               "train/grad_norm": grad_norm,
                               **metrics.get(stage.name, {})}
                        with (Path(stage.config.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                if stopped_for_deadline:
                    for result in results:
                        result.update(status="stopped", reason="deadline")
                # Metrics are a small sidecar; never serialize the large state
                # a second time merely to attach a completed evaluation.
                for result in results:
                    if result["name"] in metrics:
                        result["metrics"] = metrics[result["name"]]
                if save_due:
                    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                if do_verify and not deadline_reached():
                    from draftfit.training.joint_verification import evaluate_generation
                    settings = t.joint_verification.model_dump()
                    if deadline_unix is not None:
                        settings["deadline_unix"] = deadline_unix
                    joint_path = root / f"joint-step{step}.pt"
                    digest = hashlib.sha256()
                    with joint_path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    try:
                        report = evaluate_generation(stages, bundles, online_source,
                                                     optimizer, step, settings)
                    except TimeoutError:
                        stopped_for_deadline = True
                    else:
                        report.update(checkpoint_path=str(joint_path),
                                      artifact_sha256=digest.hexdigest(),
                                      capture_fingerprint=fixed_fingerprint)
                        (root / f"verification-step{step}.json").write_text(
                            json.dumps(report, indent=2, allow_nan=False) + "\n")
                        for candidate in report["candidates"]:
                            monitor.emit("generation_verification", step, candidate["name"],
                                artifact_sha256=report["artifact_sha256"],
                                production_engine_speed=False,
                                timing_scope=report["timing_scope"],
                                accepted_tokens=candidate["accepted_tokens"],
                                proposed_tokens=candidate["proposed_tokens"],
                                acceptance_rate=candidate["acceptance_rate"],
                                tokenparity=candidate["tokenparity"],
                                tok_s=candidate["tok_s"], baseline_tok_s=report["baseline"]["tok_s"],
                                verification_seconds=report["total_seconds"])
                        if not report["tokenparity"]:
                            raise ValueError("joint generation verification token parity failed; checkpoint preserved")
                        if milestone_observer is not None:
                            stopped_for_policy = notify_observer(report, metrics)
                        if t.joint_control:
                            from draftfit.training.joint_control import observe_generation
                            budget = json.loads(Path(t.joint_control.budget_path).read_text())
                            budget["steps_used"] = max(step, budget.get("steps_used", 0))
                            control = observe_generation(report, metrics, stages,
                                t.joint_control.model_dump(), policy_history, budget)
                            policy_history = control["history"]
                            control.update(artifact_sha256=report["artifact_sha256"], step=step)
                            (root / f"control-step{step}.json").write_text(
                                json.dumps(control, indent=2, allow_nan=False) + "\n")
                            policy_best_steps = _policy_best_steps(control)
                            for name, decision in control["decisions"].items():
                                monitor.emit("control_decision", step, name,
                                    decision=decision, action=decision["action"], reason=decision["reason"])
                            stopped_for_policy = control["stop"]
                            if not stopped_for_policy:
                                approved = min(d["next_steps"] for d in control["decisions"].values())
                                if approved < 1:
                                    raise ValueError("joint policy admitted an empty continuation interval")
                                policy_next_verify = step + min(approved, t.joint_verification.interval)
                window_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if step >= stop_step or stopped_for_deadline or stopped_for_policy:
                    break
        if window:
            raise ValueError("joint family ended with incomplete gradient accumulation")
        if stopped_for_deadline:
            for result in results:
                result.update(status="stopped", reason="deadline")
            if step != last_saved:
                checkpoint(step, cursor_epoch, cursor_batch, {})
        elif step != last_saved:
            if resume_from is not None and step == start_step and not new_phase:
                # A replayed stop performed no update. Keep the authoritative
                # bytes/SHA so another retry reuses the committed decision.
                for result in results:
                    result.update(global_step=step,
                                  joint_checkpoint=str(Path(resume_from).resolve()),
                                  emit_arm_checkpoints=getattr(t, "joint_emit_arm_checkpoints", True))
            else:
                checkpoint(step, cursor_epoch, cursor_batch, {})
        if stopped_for_policy:
            for result in results:
                result.update(status="stopped", reason="policy_boundary")
        elif not stopped_for_deadline:
            for result in results:
                result["status"] = "complete"
        if online_source is not None:
            online_source.close_queues()
            monitor.emit("queue_closed", step, stages[0].name, **online_source.queue_metrics())
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        outcome = {"output_dir": str(root), "execution": "joint", "jobs": 1,
                   "shared_backbone_parameters": shared_count, "runs": results}
        if stopped_for_deadline:
            outcome.update(status="stopped", reason="deadline")
        elif stopped_for_policy:
            outcome.update(status="stopped", reason="policy_boundary")
        if milestone_observer is not None:
            outcome["observer_reason"] = observer_reason
        return outcome

    finally:
        if online_source is not None:
            online_source.close_queues()


__all__ = ["validate_joint_family", "run_joint_family"]
