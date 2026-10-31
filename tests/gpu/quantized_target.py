"""Bounded actual NF4/INT8 teacher → file → loader → draft training smoke.

Run in an isolated CUDA runtime with bitsandbytes and --output NEW_DIRECTORY.
Random tiny Llama/Qwen weights prove integration, not model quality or speedup.
The caller must own the shared GPU lease before starting this process.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.config import Config
from draftfit.inference.capture_manifest import write_capture_manifest
from draftfit.modeling.auto import AutoDraftModel
from draftfit.modeling.target.quantization import validate_quantized_feature_source
from draftfit.offline_capture.prepare import build_capture_plan_manifest
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
from draftfit.runtime.data_plane.feature_store import LocalFeatureStore
from draftfit.training.assembly import build_model_bundle
from draftfit.training.model_loading import resolve_draft_config


def digest(model):
    return hashlib.sha256(
        b"".join(
            p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            for p in model.parameters()
        )
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    assert torch.cuda.is_available()
    import bitsandbytes as bnb

    torch.cuda.set_per_process_memory_fraction(0.1)
    torch.manual_seed(91)
    results = []
    start = time.monotonic()
    for family, config_cls, model_cls in (
        ("llama", LlamaConfig, LlamaForCausalLM),
        ("qwen3", Qwen3Config, Qwen3ForCausalLM),
    ):
        target_path = root / family
        target_config = config_cls(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        target_config._attn_implementation = "eager"
        target = model_cls(target_config).to(dtype=torch.bfloat16).eval()
        target.save_pretrained(target_path)
        del target
        for recipe, quant_cls in (
            ("bitsandbytes-4bit", bnb.nn.Linear4bit),
            ("bitsandbytes-8bit", bnb.nn.Linear8bitLt),
        ):
            teacher = OfflineTransformersCapture.from_pretrained(
                str(target_path),
                device="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                quantization=recipe,
            )
            packed_count = sum(
                isinstance(m, quant_cls) for m in teacher._model.modules()
            )
            assert packed_count > 0
            teacher.set_capture_layers([0, 2], capture_method="dspark")
            ids = torch.arange(1, 25).view(1, -1)
            mask = torch.ones_like(ids)
            captured = teacher.capture(
                input_ids=ids, attention_mask=mask, loss_mask=mask
            )
            assert not captured.hidden_states.requires_grad
            # Saving/reopening packed decoder weights exercises the prequantized auto path.
            packed_path = root / f"{family}-{recipe}-packed"
            teacher._model.save_pretrained(packed_path)
            reopened = OfflineTransformersCapture.from_pretrained(
                str(packed_path),
                device="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            reopened.set_capture_layers([0, 2], capture_method="dspark")
            replay = reopened.capture(
                input_ids=ids, attention_mask=mask, loss_mask=mask
            )
            torch.testing.assert_close(
                captured.hidden_states, replay.hidden_states, rtol=0, atol=0
            )
            torch.testing.assert_close(
                captured.last_hidden_states, replay.last_hidden_states, rtol=0, atol=0
            )
            del reopened
            for method in ("dspark", "dflash2"):
                registration = builtin_algorithm_registry().resolve(method)
                provider = registration.providers.offline[0]
                feature_root = root / f"{family}-{recipe}-{method}-features"
                feature_root.mkdir()
                record = provider.capture_layout.materialize(
                    {
                        "input_ids": captured.input_ids[0],
                        "loss_mask": captured.loss_mask[0],
                        "aux_hidden_states": captured.hidden_states,
                        "last_hidden_states": captured.last_hidden_states,
                    }
                )
                torch.save(
                    {k: v.cpu() for k, v in record.items()},
                    feature_root / "sample.ckpt",
                )
                warm_source = None
                for initialization in ("from_scratch", "weights_only"):
                    run_root = root / f"{family}-{recipe}-{method}-{initialization}"
                    cfg = Config.model_validate(
                        {
                            "run_id": run_root.name,
                            "output_dir": str(run_root),
                            "model": {
                                "target_model_path": str(target_path),
                                "target_backend": "transformers",
                                "target_quantization": recipe,
                                "hf_attn_implementation": "eager",
                                "torch_dtype": "bfloat16",
                                "mask_token_id": 127,
                                "draft_num_hidden_layers": 1,
                                "draft_block_size": 4,
                                "target_layer_ids": [0, 2],
                                "draft_checkpoint_path": warm_source,
                            },
                            "data": {
                                "hidden_states_path": str(feature_root),
                                "max_length": 32,
                                "dataloader_num_workers": 0,
                            },
                            "training": {
                                "strategy": method,
                                "attention_backend": "sdpa",
                                "num_anchors": 2,
                                "anchor_sampling": "uniform",
                                "max_steps": 3,
                            },
                        }
                    )
                    draft_config = resolve_draft_config(
                        cfg,
                        provider=registration.providers.model.draft_config,
                        target_config=target_config,
                    )
                    manifest = build_capture_plan_manifest(
                        SimpleNamespace(
                            target_model_path=str(target_path),
                            target_backend="transformers",
                            target_quantization=recipe,
                            hf_attention_implementation="eager",
                            torch_dtype="bfloat16",
                            chat_template="qwen",
                            max_length=32,
                            is_preformatted=False,
                        ),
                        target_config,
                        SimpleNamespace(
                            strategy=method,
                            capture_method=provider.capture_layout.capture_method,
                            capture_layers=[0, 2],
                            draft_config=draft_config,
                            layout=provider.capture_layout,
                        ),
                    )
                    write_capture_manifest(
                        str(feature_root / "capture_manifest.json"), manifest
                    )
                    validate_quantized_feature_source(cfg, target_config)
                    bundle = build_model_bundle(cfg, algorithm=registration)
                    if warm_source:
                        loaded = AutoDraftModel.from_pretrained(
                            warm_source,
                            config=bundle.draft_config,
                            torch_dtype=torch.bfloat16,
                        )
                        assert digest(bundle.draft_model) == digest(loaded)
                        del loaded
                    reader = provider.build_reader(
                        str(feature_root),
                        run_id=run_root.name,
                        ttt_length=4,
                        max_len=32,
                    )
                    loader = FeatureDataLoader(
                        LocalFeatureStore(),
                        refs=reader.read(),
                        batch_size=1,
                        strategy=method,
                        device="cuda:0",
                        gc_interval_s=None,
                        per_sample_transform=provider.build_normalizer(max_len=32),
                        collate_fn=provider.build_collator(),
                    )
                    try:
                        batch = next(iter(loader)).tensors
                    finally:
                        loader.close()
                    before = digest(bundle.draft_model)
                    optimizer = torch.optim.AdamW(
                        [p for p in bundle.draft_model.parameters() if p.requires_grad],
                        lr=1e-3,
                    )
                    losses = []
                    for step in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss, _, _ = bundle.model(**batch)
                        assert torch.isfinite(loss)
                        loss.backward()
                        grads = [
                            p.grad
                            for p in bundle.draft_model.parameters()
                            if p.grad is not None
                        ]
                        assert grads and all(torch.isfinite(g).all() for g in grads)
                        assert any(torch.count_nonzero(g) for g in grads)
                        optimizer.step()
                        losses.append(loss.item())
                    after = digest(bundle.draft_model)
                    assert before != after
                    assert all(p.grad is None for p in teacher._model.parameters())
                    output = run_root / "draft"
                    bundle.draft_model.save_pretrained(output)
                    reloaded = AutoDraftModel.from_pretrained(
                        output, config=bundle.draft_config, torch_dtype=torch.bfloat16
                    )
                    assert digest(reloaded) == after
                    assert (
                        reloaded.config.draftfit_target_quantization["recipe"] == recipe
                    )
                    results.append(
                        {
                            "family": family,
                            "recipe": recipe,
                            "algorithm": method,
                            "initialization": initialization,
                            "packed_linears": packed_count,
                            "updates": 3,
                            "losses": losses,
                            "weights_changed": True,
                            "reload_weights_exact": True,
                            "prequantized_capture_exact": True,
                        }
                    )
                    warm_source = str(output)
                    del bundle, optimizer, reloaded, batch
                    gc.collect()
            del teacher, captured, replay
            gc.collect()
            torch.cuda.empty_cache()
    import transformers

    report = {
        "status": "PASS",
        "scope": "tiny random-weight integration, not quality or serving",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bnb.__version__,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.monotonic() - start,
        "results": results,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
