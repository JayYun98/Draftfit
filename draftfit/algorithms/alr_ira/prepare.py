"""Reference target-rollout producer; IRA reuses the primary's computed features."""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import torch
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.inference.capture_manifest import build_capture_manifest, write_capture_manifest
from .providers import KEYS


@torch.no_grad()
def rollout_features(target, prefix, block_size, layer_ids, stop_token_ids=()):
    teacher = OfflineTransformersCapture(target)
    if teacher._model.config.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("ALR reference producer currently supports dense Llama/Qwen text targets")
    if any(i >= len(teacher._model.layers) for i in layer_ids):
        raise ValueError("target tap exceeds decoder depth")
    ids = prefix.clone()
    generated = 0
    # ponytail: full-prefix reference forwards cost O(block_size * prefix_length);
    # add state-safe cached rollouts after target feature parity is measured.
    for _ in range(block_size - 1):
        out = target(input_ids=ids, use_cache=False, return_dict=True)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, token], dim=1)
        generated += 1
        if int(token[0, 0]) in stop_token_ids:
            break
    states = {}; handles = []
    try:
        for index in layer_ids:
            def save(_module, _args, output, index=index):
                states[index] = teacher._capture_output(output)
            handles.append(teacher._model.layers[index].register_forward_hook(save))
        out = target(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return ids, torch.cat([states[i] for i in layer_ids], dim=-1), out.hidden_states[-1].detach(), generated


def prepare(config, *, target=None):
    from transformers import AutoModelForCausalLM
    destination = Path(config.output_dir)
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if target is None:
        target = AutoModelForCausalLM.from_pretrained(
            config.model.target_model_path, revision=config.model.target_revision,
            trust_remote_code=config.model.trust_remote_code,
            dtype=getattr(torch, config.model.torch_dtype), attn_implementation="eager",
        ).to(config.runtime.device)
    target.eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(config.runtime.seed)
    device = next(target.parameters()).device
    settings = config.training
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".alr-ira-", dir=destination.parent))
    count = rollouts = forward_calls = 0
    records = []
    try:
        with open(config.data.input_path, encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= config.data.max_samples:
                    break
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("input_ids"), list) or not isinstance(row.get("loss_mask"), list):
                    raise ValueError("ALR input rows require input_ids and assistant loss_mask arrays")
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                mask = torch.tensor(row["loss_mask"])
                if ids.ndim != 1 or ids.numel() < 2 or mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all():
                    raise ValueError("input IDs and binary loss mask must have matching nonempty sequence shape")
                if not all(type(token) is int and 0 <= token < target.config.vocab_size for token in row["input_ids"]):
                    raise ValueError("input_ids contain invalid target token IDs")
                candidates = torch.nonzero(mask[1:].bool(), as_tuple=True)[0]
                candidates = candidates[candidates + settings.block_size <= settings.max_length]
                chosen = candidates[torch.randperm(len(candidates), generator=generator)[:settings.anchors_per_sample]]
                for original_anchor in chosen.tolist():
                    prefix = ids[:original_anchor + 1].unsqueeze(0).to(device)
                    full_ids, features, final, generated = rollout_features(
                        target, prefix, settings.block_size, settings.target_layer_ids, settings.stop_token_ids,
                    )
                    rollouts += 1; forward_calls += generated + 1
                    anchors = [original_anchor]
                    if settings.in_rollout_anchors:
                        # Eq. 4: offsets 2..block_size-2, all within this same rollout.
                        offset = int(torch.randint(2, settings.block_size - 1, (), generator=generator))
                        if offset < generated:
                            anchors.append(original_anchor + offset)
                    for anchor in anchors:
                        loss = torch.zeros_like(full_ids[0], dtype=torch.float32)
                        loss[anchor + 1:] = 1
                        tensors = {
                            "input_ids": full_ids[0].cpu(), "loss_mask": loss.cpu(),
                            "hidden_states": features[0].cpu(), "target_last_hidden_states": final[0].cpu(),
                            "alr_anchor": torch.tensor([anchor]),
                        }
                        if not all(torch.isfinite(value).all() for value in tensors.values()):
                            raise ValueError("target rollout produced nonfinite features")
                        file = f"{count:08d}.ckpt"; torch.save(tensors, scratch / file)
                        records.append({"file": file, "corpus_id": row.get("id", index), "corpus_anchor": original_anchor,
                                        "anchor": anchor, "primary": len(records) if anchor == original_anchor else count - 1})
                        count += 1
        if not count:
            raise ValueError("no rollout yielded supervised tokens; inspect input masks/stop IDs")
        revision = config.model.target_revision or getattr(target.config, "_commit_hash", None)
        if revision is None:
            revision = "local-config-" + hashlib.sha256(target.config.to_json_string().encode()).hexdigest()
        manifest = build_capture_manifest(
            strategy="alr_ira", capture_method="dflash", capture_layers=settings.target_layer_ids,
            feature_names=KEYS, target_model=config.model.target_model_path, target_revision=revision,
            target_hidden_size=target.config.hidden_size, target_vocab_size=target.config.vocab_size,
            max_length=settings.max_length, capture_backend="transformers",
            capture_settings={"label_source": "target_rollout", **settings.model_dump()},
        )
        write_capture_manifest(str(scratch / "capture_manifest.json"), manifest)
        result = {"blocks": count, "rollouts": rollouts, "forward_calls": forward_calls,
                  "secondary_blocks": sum(record["anchor"] != record["corpus_anchor"] for record in records)}
        (scratch / "preparation.json").write_text(json.dumps({**result, "records": records}, indent=2)+"\n")
        scratch.rename(destination)
        return result
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
