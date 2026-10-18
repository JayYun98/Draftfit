"""Same-module greedy generation checks, not a production serving benchmark.

The frozen online teacher and current family drafts are reused. Both baseline
and verification recompute full prefixes without a KV cache. Acceptance counts
only proposals committed after comparing them with causal target predictions;
neither labels, teacher-forced accuracy nor confidence scores count as accepts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


@contextmanager
def suspend_training_state(bundles, optimizer, teacher=None, *, cpu_offload=True):
    """Suspend gradients and offload Adam/master tensors, preserving identities.

    No model module or model Parameter is replaced or moved: shared backbone and frozen
    teacher weights stay resident. Tensor storage of optimizer state is moved
    in place and restored in finally, including on forward/timeout failures.
    Caller must quiesce any feature producer before entering this context.
    """
    roots = [bundle.model for bundle in bundles]
    roots += [bundle.draft_model for bundle in bundles]
    roots += [
        bundle.target_head
        for bundle in bundles
        if isinstance(getattr(bundle, "target_head", None), torch.nn.Module)
    ]
    if teacher is not None:
        roots.append(teacher._model)
    modules = {id(module): module for root in roots for module in root.modules()}
    modes = [(module, module.training) for module in modules.values()]
    # Rotary implementations can replace even nonpersistent buffers on forward.
    buffers = [
        (
            module,
            name,
            value,
            value.detach().cpu().clone() if value is not None else None,
        )
        for module in modules.values()
        for name, value in module._buffers.items()
    ]
    rotary_state = [
        (module, name, getattr(module, name))
        for module in modules.values()
        for name in ("max_seq_len_cached", "attention_scaling")
        if hasattr(module, name) and not isinstance(getattr(module, name), torch.Tensor)
    ]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_rng = (
        torch.mps.get_rng_state()
        if any(p.device.type == "mps" for root in roots for p in root.parameters())
        else None
    )
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    masters = list(getattr(optimizer, "fp32_params", []))
    parameters = {id(p): p for root in roots for p in root.parameters()}
    parameters.update(
        {id(p): p for group in raw_optimizer.param_groups for p in group["params"]}
    )
    gradients = [
        (p, p.grad, p.grad.device if p.grad is not None else None)
        for p in parameters.values()
    ]
    moved, seen = [], set()

    def offload(tensor):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            if cpu_offload and tensor.device.type != "cpu":
                device = tensor.device
                tensor.data = tensor.detach().to("cpu")
                moved.append((tensor, device))

    try:
        for parameter, gradient, _ in gradients:
            if gradient is not None:
                offload(gradient)
            parameter.grad = None
        for tensor in [
            *masters,
            *_tensor_leaves(raw_optimizer.state),
            *_tensor_leaves(getattr(optimizer, "last_grad_norm", None)),
        ]:
            offload(tensor)
        for module, _ in modes:
            module.training = False
        with torch.no_grad():
            yield
    finally:
        try:
            for tensor, device in reversed(moved):
                tensor.data = tensor.to(device)
            for parameter, gradient, _ in gradients:
                parameter.grad = gradient
            for module, name, original, saved in buffers:
                if original is not None:
                    original.data = saved.to(
                        device=original.device, dtype=original.dtype
                    )
                module._buffers[name] = original
            for module, name, original in rotary_state:
                setattr(module, name, original)
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)


def _settings(config):
    value = dict(config.model_dump() if hasattr(config, "model_dump") else config)
    defaults = dict(
        interval=250,
        max_prompts=2,
        max_new_tokens=8,
        draft_tokens=2,
        trials=1,
        offload_optimizer=True,
        existing_drafts={},
    )
    unknown = (
        set(value)
        - set(defaults)
        - {"stop_token_ids", "prompt_tokens", "deadline_unix"}
    )
    if unknown:
        raise ValueError(f"unknown generation verification settings: {sorted(unknown)}")
    defaults.update(value)
    for name in ("interval", "max_prompts", "max_new_tokens", "draft_tokens", "trials"):
        if type(defaults[name]) is not int or defaults[name] < 1:
            raise ValueError(
                f"generation verification {name} must be a positive integer"
            )
    if type(defaults["offload_optimizer"]) is not bool:
        raise ValueError("offload_optimizer must be boolean")
    from draftfit.config.schema import JointVerificationConfig

    defaults["existing_drafts"] = JointVerificationConfig.validate_existing_drafts(
        defaults["existing_drafts"]
    )
    if defaults["existing_drafts"] and not defaults["offload_optimizer"]:
        raise ValueError("existing_drafts requires offload_optimizer=true")
    prompt_tokens = defaults.get("prompt_tokens")
    if prompt_tokens is not None and (
        type(prompt_tokens) is not int or prompt_tokens < 1
    ):
        raise ValueError("prompt_tokens must be a positive integer")
    stops = defaults.get("stop_token_ids", ())
    if not isinstance(stops, (tuple, list)) or any(
        type(v) is not int or v < 0 for v in stops
    ):
        raise ValueError("stop_token_ids must be nonnegative token IDs")
    defaults["stop_token_ids"] = list(stops)
    deadline = defaults.get("deadline_unix")
    if deadline is not None and (
        type(deadline) not in (float, int) or not math.isfinite(deadline)
    ):
        raise ValueError("deadline_unix must be finite")
    return defaults


def _check_deadline(config):
    if (
        config.get("deadline_unix") is not None
        and time.time() >= config["deadline_unix"]
    ):
        raise TimeoutError("generation verification deadline reached")


def _json_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _export_binding(path, config):
    """Inventory every file, including auxiliary metadata, without loading weights."""
    files = {}
    root = Path(path)
    for item in sorted(root.rglob("*")):
        _check_deadline(config)
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(
                "existing draft exports must contain regular files, not links or special files"
            )
        if item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    _check_deadline(config)
                    digest.update(chunk)
            files[item.relative_to(root).as_posix()] = digest.hexdigest()
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name in files:
            index = json.loads((root / index_name).read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    "existing draft shard index requires a nonempty weight_map"
                )
            if any(
                not isinstance(name, str) or name not in files
                for name in weight_map.values()
            ):
                raise ValueError(
                    "existing draft shards must be bound files inside the export"
                )
    return dict(path=str(root), files=files, sha256=_json_sha(files))


def _load_existing_draft(bundle, path):
    """Independent CPU-only HF load; never load a teacher or training optimizer."""
    from draftfit.algorithms.common.dflash_family_model import (
        OnlineDFlash2Model,
        OnlineDFlashModel,
        OnlineDSparkModel,
    )
    from draftfit.modeling.auto import AutoDraftModel
    from draftfit.training.model_loading import load_draft_config_source

    if type(bundle.model) not in (
        OnlineDFlashModel,
        OnlineDFlash2Model,
        OnlineDSparkModel,
    ):
        raise ValueError("unsupported existing draft model wrapper")
    current = bundle.draft_model
    source_config = load_draft_config_source(path, trust_remote_code=False)
    if source_config.architectures != [type(current).__name__]:
        raise ValueError("existing draft architecture differs from corresponding arm")
    # Tensor shapes alone do not bind capture taps, masks, RoPE or target logits.
    for field in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "block_size",
        "layer_types",
        "sliding_window",
        "is_causal",
        "rope_parameters",
        "rms_norm_eps",
        "hidden_act",
        "attention_bias",
        "attention_dropout",
        "input_embedding_scale",
        "output_multiplier",
        "final_logit_softcapping",
        "dflash_config",
    ):
        if getattr(source_config, field, None) != getattr(current.config, field, None):
            raise ValueError(f"existing draft has incompatible {field}")
    source_config._attn_implementation = current.config._attn_implementation
    # HF missing-weight initialization can consume RNG even on a rejected load.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        draft, info = AutoDraftModel.from_pretrained(
            path,
            config=source_config,
            local_files_only=True,
            trust_remote_code=False,
            device_map="cpu",
            torch_dtype=next(current.parameters()).dtype,
            output_loading_info=True,
            weights_only=True,
        )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(field):
            raise ValueError(f"existing draft has incompatible {field}: {info[field]}")
    expected, actual = current.state_dict(), draft.state_dict()
    if (
        type(draft) is not type(current)
        or expected.keys() != actual.keys()
        or any(expected[k].shape != actual[k].shape for k in expected)
    ):
        raise ValueError("existing draft tensor shapes differ from corresponding arm")
    if any(
        t.device.type != "cpu" or not torch.isfinite(t).all() for t in actual.values()
    ):
        raise ValueError("existing draft requires finite CPU-loaded weights")
    if (
        tuple(draft.target_layer_ids) != tuple(current.target_layer_ids)
        or draft.mask_token_id != current.mask_token_id
    ):
        raise ValueError("existing draft target capture taps or mask token differ")
    return draft.requires_grad_(False).eval()


def preflight_existing_drafts(stages, bundles, config):
    """Call before the first update; returns immutable-source receipts to persist.

    Loads/discards one CPU draft at a time, validates exact arm architecture,
    shape and decoding semantics, and leaves training objects/RNG untouched.
    The controller owns cross-milestone/resume fingerprint binding; evaluation
    additionally rejects source changes during each comparison.
    """
    config = _settings(config)
    if len(stages) != len(bundles) or len({stage.name for stage in stages}) != len(
        stages
    ):
        raise ValueError("existing drafts require distinct paired stages and bundles")
    by_name = {stage.name: bundle for stage, bundle in zip(stages, bundles)}
    unknown = set(config["existing_drafts"]) - set(by_name)
    if unknown:
        raise ValueError(f"existing drafts name unknown arms: {sorted(unknown)}")
    receipts = {}
    for name, path in config["existing_drafts"].items():
        binding = _export_binding(path, config)
        draft = _load_existing_draft(by_name[name], path)
        del draft
        if _export_binding(path, config) != binding:
            raise ValueError("existing draft export changed during preflight")
        receipts[name] = binding
    return receipts


@contextmanager
def _existing_bundle(current, binding, config):
    if _export_binding(binding["path"], config) != binding:
        raise ValueError("existing draft export changed after preflight")
    draft = _load_existing_draft(current, binding["path"])
    try:
        if _export_binding(binding["path"], config) != binding:
            raise ValueError("existing draft export changed during loading")
        # Move ONLY this independent draft. A recursive wrapper.to/eval would
        # traverse the resident shared target pieces as well.
        draft.to(next(current.draft_model.parameters()).device)
        model = type(current.model)(
            draft_model=draft,
            target_lm_head=current.model.lm_head,
            target_embed_tokens=current.model.embed_tokens,
            mask_token_id=draft.mask_token_id,
            block_size=draft.block_size,
            attention_backend=draft.config._attn_implementation,
        )
        model.training = False
        yield replace(
            current, model=model, draft_model=draft, draft_config=draft.config
        )
        if _export_binding(binding["path"], config) != binding:
            raise ValueError("existing draft export changed during comparison")
    finally:
        draft.to("cpu")


def _clock(*devices):
    for device in dict.fromkeys(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    return time.perf_counter()


def _capture(teacher, ids, config):
    _check_deadline(config)
    return teacher.capture(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.zeros_like(ids),
    )


def _project(bundle, hidden):
    # The frozen target head belongs to the bundle, even when the teacher is remote.
    device = next(bundle.model.parameters()).device
    logits = bundle.model._compute_logits(hidden.to(device))
    if not torch.isfinite(logits).all():
        raise ValueError("generation verification got nonfinite logits")
    return logits


def _propose(bundle, features, anchor, count):
    """One real masked block; DSpark Markov and DFlash2 selector are active."""
    from draftfit.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
    from draftfit.modeling.draft.dflash2 import DFlash2DraftModel

    draft = bundle.draft_model
    device = next(draft.parameters()).device
    features = replace(
        features,
        hidden_states=features.hidden_states.to(device),
        input_ids=features.input_ids.to(device),
    )
    anchor = anchor.to(device)
    length, block_size = features.input_ids.shape[1], draft.block_size
    block = torch.full(
        (1, block_size), draft.mask_token_id, dtype=torch.long, device=device
    )
    block[:, 0] = anchor
    options = bundle.model._block_mask_options()
    mask_args = dict(
        anchor_positions=torch.tensor([[length]], device=device),
        block_keep_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
        S=length,
        block_size=block_size,
        device=device,
    )
    mask = create_dflash_sdpa_mask(**mask_args, **options)
    if draft.sliding_window is not None:
        mask = {
            "full_attention": mask,
            "sliding_attention": create_dflash_sdpa_mask(
                **mask_args,
                sliding_window=draft.sliding_window,
                **bundle.model._sliding_block_mask_options(),
            ),
        }
    hidden = draft(
        target_hidden=features.hidden_states,
        noise_embedding=bundle.model.embed_tokens(block),
        position_ids=torch.arange(length + block_size, device=device).unsqueeze(0),
        attention_mask=mask,
        use_cache=False,
        is_causal=False,
    )
    if not torch.isfinite(hidden).all():
        raise ValueError("generation verification got nonfinite draft hidden states")
    if isinstance(draft, DFlash2DraftModel):
        logits = _project(bundle, hidden[:, 1:])
        predecessor, selected = anchor, []
        for index in range(count):
            scores, candidates = draft.score_candidates(
                hidden[:, index + 1], logits[:, index], predecessor
            )
            if not torch.isfinite(scores).all():
                raise ValueError(
                    "generation verification got nonfinite selector scores"
                )
            predecessor = candidates.gather(
                -1, scores.argmax(-1, keepdim=True)
            ).squeeze(-1)
            selected.append(predecessor)
        return torch.stack(selected, dim=1)
    # Existing serving proposal boundary uses greedy Markov decoding by default.
    target = SimpleNamespace(lm_head=lambda value: _project(bundle, value))
    return draft._sample_draft_tokens(target, hidden, block)[:, :count]


def _generate(teacher, bundle, prompt, config, *, speculative):
    device = teacher._model.get_input_embeddings().weight.device
    bundle_device = next(bundle.model.parameters()).device
    prefix = prompt.to(device).unsqueeze(0)
    initial_length = prefix.shape[1]
    generated, accepted, proposed, rejects, rounds = [], 0, 0, 0, 0
    proposal_seconds = verification_seconds = 0.0
    traces = []
    stops = set(config["stop_token_ids"])
    start = _clock(device, bundle_device)

    def append(token):
        nonlocal prefix
        token = int(token)
        generated.append(token)
        prefix = torch.cat((prefix, prefix.new_tensor([[token]])), dim=1)
        return token in stops or len(generated) >= config["max_new_tokens"]

    while len(generated) < config["max_new_tokens"]:
        _check_deadline(config)
        features = _capture(teacher, prefix, config)
        anchor = _project(bundle, features.last_hidden_states[:, -1:]).argmax(-1)[:, 0]
        rounds += 1
        if not speculative:
            if append(anchor.item()):
                break
            continue
        count = min(
            config["draft_tokens"], config["max_new_tokens"] - len(generated) - 1
        )
        if count == 0 or anchor.item() in stops:
            append(anchor.item())
            break
        tick = _clock(device, bundle_device)
        proposal = _propose(bundle, features, anchor, count)
        proposal_seconds += _clock(device, bundle_device) - tick
        del features
        proposed += count
        # A causal full-prefix target pass verifies a *generated* draft suffix.
        verify_ids = torch.cat(
            (prefix, anchor.reshape(1, 1).to(device), proposal.to(device)), dim=1
        )
        tick = _clock(device, bundle_device)
        verified = _capture(teacher, verify_ids, config)
        posterior = _project(
            bundle, verified.last_hidden_states[:, prefix.shape[1] :]
        ).argmax(-1)[0]
        verification_seconds += _clock(device, bundle_device) - tick
        del verified, verify_ids
        trace = dict(
            proposed_token_ids=proposal[0].tolist(),
            target_token_ids=posterior.tolist(),
            accepted_tokens=0,
            rejected=False,
        )
        done = append(anchor.item())
        for index, token in enumerate(proposal[0].tolist()):
            if done:
                break
            if token != int(posterior[index]):
                rejects += 1
                trace["rejected"] = True
                done = append(posterior[index].item())
                break
            accepted += 1
            trace["accepted_tokens"] += 1
            done = append(token)
        else:
            if not done:
                done = append(posterior[count].item())
        traces.append(trace)
        if done:
            break
    seconds = _clock(device, bundle_device) - start
    return dict(
        token_ids=generated,
        output_token_ids=prefix[0].tolist(),
        prompt_tokens=initial_length,
        generated_tokens=len(generated),
        accepted_tokens=accepted,
        proposed_tokens=proposed,
        rejected_tokens=proposed - accepted,
        reject_events=rejects,
        rounds=rounds,
        seconds=seconds,
        tok_s=len(generated) / seconds,
        proposal_seconds=proposal_seconds,
        verification_seconds=verification_seconds,
        traces=traces,
    )


def _preflight(stages, bundles, source, config):
    from draftfit.modeling.draft.dflash import DFlashDraftModel
    from draftfit.modeling.draft.dflash2 import DFlash2DraftModel
    from draftfit.modeling.draft.dspark import DSparkDraftModel
    from draftfit.offline_capture.transformers import OfflineTransformersCapture

    teacher = getattr(source, "_teacher", None)
    if not isinstance(teacher, OfflineTransformersCapture):
        raise ValueError(
            "generation verification requires the already-loaded online Transformers teacher"
        )
    family = teacher._model.config.model_type
    if family not in {"llama", "qwen2", "qwen3", "gemma4_unified_text"}:
        raise ValueError(
            f"unsupported generation verification target: {family}; "
            "only Llama/Qwen2/Qwen3/Gemma4Unified reference decoding is supported"
        )
    if (
        family != "gemma4_unified_text"
        and getattr(teacher._model.config, "sliding_window", None) is not None
    ):
        raise ValueError(
            "generation verification requires a dense full-attention target"
        )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("generation verification requires a single process")
    if (
        not stages
        or len(stages) != len(bundles)
        or len({s.name for s in stages}) != len(stages)
    ):
        raise ValueError(
            "generation verification requires distinct paired stages and bundles"
        )
    classes = {
        "dflash": DFlashDraftModel,
        "dspark": DSparkDraftModel,
        "dflash2": DFlash2DraftModel,
    }
    if any(parameter.requires_grad for parameter in teacher._model.parameters()):
        raise ValueError("generation verification teacher must be frozen")
    if teacher.capture_layers is None:
        raise ValueError("generation verification teacher capture layers are unset")
    for stage, bundle in zip(stages, bundles):
        strategy, draft = stage.config.training.strategy, bundle.draft_model
        if type(draft) is not classes.get(strategy):
            raise ValueError(
                f"unsupported generation verification draft for {strategy}"
            )
        if draft.config._attn_implementation not in {"sdpa", "eager"}:
            raise ValueError(
                "generation verification supports eager/SDPA draft attention only"
            )
        if config["draft_tokens"] >= draft.block_size:
            raise ValueError("draft_tokens must be smaller than every draft block_size")
        if tuple(teacher.capture_layers) != tuple(draft.target_layer_ids):
            raise ValueError("generation verification target capture taps differ")
        if any(token >= bundle.target_vocab_size for token in config["stop_token_ids"]):
            raise ValueError(
                "generation verification stop token is outside the target vocabulary"
            )
        bundle_device = next(bundle.model.parameters()).device
        if any(p.device != bundle_device for p in bundle.model.parameters()):
            raise ValueError(
                "generation verification bundle modules must share a device"
            )
        if any(p.requires_grad for p in bundle.model.lm_head.parameters()):
            raise ValueError("generation verification target LM head must be frozen")
        if (
            bundle.model.lm_head is not bundles[0].model.lm_head
            or bundle.model.embed_tokens is not bundles[0].model.embed_tokens
        ):
            raise ValueError(
                "generation verification families must share frozen target pieces"
            )
    return teacher


def _compare(stage, bundle, teacher, prompts, baseline_runs, config, domains, step):
    runs = []
    for index, baseline in enumerate(baseline_runs):
        run = _generate(
            teacher, bundle, prompts[baseline["row_index"]], config, speculative=True
        )
        run.update(
            trial=baseline["trial"],
            row_index=baseline["row_index"],
            request_id=baseline["request_id"],
            baseline_pair=dict(
                run_index=index,
                request_id=baseline["request_id"],
                output_sha256=baseline["output_sha256"],
            ),
            tokenparity=(
                run["output_token_ids"] == baseline["output_token_ids"]
                and run["token_ids"] == baseline["token_ids"]
            ),
        )
        runs.append(run)
    seconds = sum(run["seconds"] for run in runs)
    proposed = sum(run["proposed_tokens"] for run in runs)
    accepted = sum(run["accepted_tokens"] for run in runs)
    return dict(
        name=stage.name,
        strategy=stage.config.training.strategy,
        preset="greedy-full-prefix-reference",
        bindings=dict(
            draft_architecture=type(bundle.draft_model).__name__,
            block_size=bundle.draft_model.block_size,
            draft_config=bundle.draft_config.to_dict(),
        ),
        accepted_tokens=accepted,
        proposed_tokens=proposed,
        acceptance_rate=accepted / proposed if proposed else None,
        rejected_tokens=proposed - accepted,
        reject_events=sum(run["reject_events"] for run in runs),
        tokenparity=all(run["tokenparity"] for run in runs),
        seconds=seconds,
        tok_s=sum(run["generated_tokens"] for run in runs) / seconds,
        domains=sorted(set(domains)),
        step=step,
        runs=runs,
    )


def evaluate_generation(stages, bundles, online_source, optimizer, step, config):
    """Return JSON-safe full-generation timings, exact parity and real accepts.

    Parent owns interval scheduling, queue quiescence, persistence and deadlines.
    Config accepts the parent's JointVerificationConfig.model_dump(). Optional
    prompt_tokens/stop_token_ids/deadline_unix bind an explicit bounded workload.
    With no prompt override, use tokens before the first supervised token (or
    a disclosed <=4-token prefix when the row has no unsupervised prefix).
    """
    config = _settings(config)
    if type(step) is not int or step < 0:
        raise ValueError("generation verification step must be nonnegative")
    teacher = _preflight(stages, bundles, online_source, config)
    family = teacher._model.config.model_type
    embedding = teacher._model.get_input_embeddings().weight
    runtime = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=str(embedding.device),
        dtype=str(embedding.dtype),
    )
    rows = online_source._rows[True][: config["max_prompts"]]
    if not rows:
        raise ValueError("generation verification requires held-out token rows")
    prompts, bindings, domains = [], [], []
    limit = getattr(teacher._model.config, "max_position_embeddings", None)
    for index, row in enumerate(rows):
        ids, mask = row["input_ids"], row["loss_mask"]
        if ids.ndim != 1 or ids.numel() == 0 or mask.shape != ids.shape:
            raise ValueError("generation verification needs nonempty 1-D token rows")
        supervised = (mask > 0).nonzero().flatten()
        boundary = int(supervised[0]) if supervised.numel() else 0
        count = config.get("prompt_tokens") or (
            boundary if boundary > 0 else min(4, ids.numel())
        )
        if count > ids.numel():
            raise ValueError("prompt_tokens exceeds held-out row length")
        if limit is not None and count + config["max_new_tokens"] > limit:
            raise ValueError(
                "generation verification workload exceeds target context limit"
            )
        domain = row.get("domain", "unlabeled")
        if not isinstance(domain, str):
            raise ValueError("generation verification row domain must be a string")
        prompt = ids[:count].detach().cpu().clone()
        prompts.append(prompt)
        domains.append(domain)
        bindings.append(
            dict(
                row_index=index,
                domain=domain,
                prompt_token_ids=prompt.tolist(),
                prompt_sha256=hashlib.sha256(
                    json.dumps(prompt.tolist()).encode()
                ).hexdigest(),
                prompt_selection=(
                    "explicit"
                    if config.get("prompt_tokens")
                    else "unsupervised_prefix" if boundary > 0 else "bounded_prefix"
                ),
            )
        )
    baseline_runs, candidates, existing_drafts = [], [], []
    start = time.perf_counter()
    with suspend_training_state(
        bundles, optimizer, teacher, cpu_offload=config["offload_optimizer"]
    ):
        sources = preflight_existing_drafts(stages, bundles, config)
        for trial in range(config["trials"]):
            for row_index, prompt in enumerate(prompts):
                baseline = _generate(
                    teacher, bundles[0], prompt, config, speculative=False
                )
                request = dict(
                    step=step,
                    trial=trial,
                    row_index=row_index,
                    prompt=bindings[row_index],
                    max_new_tokens=config["max_new_tokens"],
                    stop_token_ids=config["stop_token_ids"],
                    target_model_type=family,
                    target_revision=stages[0].config.model.target_revision,
                    target_model_path=stages[0].config.model.target_model_path,
                )
                baseline_runs.append(
                    dict(
                        trial=trial,
                        row_index=row_index,
                        request_id=_json_sha(request),
                        request=request,
                        output_sha256=_json_sha(baseline["output_token_ids"]),
                        **baseline,
                    )
                )
        for stage, bundle in zip(stages, bundles):
            candidates.append(
                _compare(
                    stage,
                    bundle,
                    teacher,
                    prompts,
                    baseline_runs,
                    config,
                    domains,
                    step,
                )
            )
        for stage, bundle in zip(stages, bundles):
            if stage.name in sources:
                with _existing_bundle(bundle, sources[stage.name], config) as old:
                    result = _compare(
                        stage,
                        old,
                        teacher,
                        prompts,
                        baseline_runs,
                        config,
                        domains,
                        step,
                    )
                    result["eligible_for_selection"] = False
                    result["bindings"]["source"] = sources[stage.name]
                    existing_drafts.append(result)
        # Catch changes to an earlier export while a later reference runs.
        for binding in sources.values():
            if _export_binding(binding["path"], config) != binding:
                raise ValueError("existing draft export changed during comparison")
    seconds = sum(run["seconds"] for run in baseline_runs)
    parity_failures = [
        dict(kind=kind, name=item["name"], **run["baseline_pair"])
        for kind, items in (
            ("candidate", candidates),
            ("existing_draft", existing_drafts),
        )
        for item in items
        for run in item["runs"]
        if not run["tokenparity"]
    ]
    return dict(
        schema="draftfit-joint-generation-v1",
        step=step,
        preset="greedy-full-prefix-reference",
        production_engine_speed=False,
        timing_scope="full_generation_including_prefill_no_kv_cache",
        acceptance_scope="committed_draft_suffix_excludes_target_anchor",
        total_seconds=time.perf_counter() - start,
        domains=sorted(set(domains)),
        bindings=dict(
            config=config,
            prompts=bindings,
            runtime=runtime,
            target_model_type=family,
            target_model_path=stages[0].config.model.target_model_path,
            target_revision=stages[0].config.model.target_revision,
            capture_layers=list(teacher.capture_layers),
        ),
        baseline=dict(
            name="target-only",
            seconds=seconds,
            step=step,
            domains=sorted(set(domains)),
            preset="greedy-full-prefix-reference",
            accepted_tokens=0,
            proposed_tokens=0,
            tok_s=sum(run["generated_tokens"] for run in baseline_runs) / seconds,
            runs=baseline_runs,
        ),
        candidates=candidates,
        existing_drafts=existing_drafts,
        parity_failures=parity_failures,
        tokenparity=not parity_failures,
    )


__all__ = ["evaluate_generation", "suspend_training_state", "preflight_existing_drafts"]
