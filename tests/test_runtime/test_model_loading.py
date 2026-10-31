# coding=utf-8
"""Draft config sources, target-derived defaults, and weights-only loading."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.config import Config
from draftfit.training.model_loading import (
    load_draft_config_source,
    resolve_draft_config,
    warm_start_draft_model,
)


def _run_config(strategy: str, **model_overrides) -> Config:
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    if strategy == "eagle3":
        model["vocab_mapping_path"] = "/mapping.pt"
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"].update(
            {
                "attention_backend": "flex_attention",
                "batch_size": 1,
                "max_steps": 1,
            }
        )
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "/control",
                "backend": "mooncake",
                "server_urls": ["http://capture:30000"],
            },
        }
    return Config.model_validate(payload)


def _draft_config_provider(strategy: str):
    registration = builtin_algorithm_registry().resolve(strategy)
    return registration.providers.model.draft_config


def _target_config(*, layers: int = 12):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )


def _draft_payload(architecture: str, *, layers: int = 1, block_size=None):
    payload = {
        "architectures": [architecture],
        "model_type": "qwen3" if architecture == "DFlashDraftModel" else "llama",
        "vocab_size": 128,
        "draft_vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    if block_size is not None:
        payload.update(
            {
                "block_size": block_size,
                "num_target_layers": 12,
                "dflash_config": {},
            }
        )
    return payload


class DraftConfigResolutionTest(unittest.TestCase):
    def test_family_identity_conversion_does_not_alias_source(self):
        from draftfit.training.model_loading import convert_family_draft_config
        from tests.test_dflash2_integration import tiny_config

        source = tiny_config()
        converted = convert_family_draft_config(source, "dflash2")
        converted.dflash_config["mask_token_id"] = 1
        self.assertEqual(source.dflash_config["mask_token_id"], 63)

    def test_family_conversion_rejects_conflicting_mask_without_mutation(self):
        from draftfit.training.model_loading import convert_family_draft_config
        from tests.test_dflash2_integration import tiny_config

        source = tiny_config()
        with self.assertRaisesRegex(ValueError, "conflicting mask_token_id"):
            convert_family_draft_config(source, "dflash", mask_token_id=1)
        self.assertEqual(source.dflash_config["mask_token_id"], 63)

    def test_checkpoint_conversion_uses_explicit_mask(self):
        from draftfit.training.model_loading import convert_family_draft_config
        from tests.test_dflash2_integration import tiny_config

        source = convert_family_draft_config(tiny_config(), "dflash")
        source.dflash_config.pop("mask_token_id")
        with tempfile.TemporaryDirectory() as directory:
            source.to_json_file(os.path.join(directory, "config.json"))
            cfg = _run_config(
                "dflash2", draft_checkpoint_path=directory, mask_token_id=63
            )
            resolved = resolve_draft_config(
                cfg, provider=_draft_config_provider("dflash2")
            )
        self.assertEqual(resolved.dflash_config["mask_token_id"], 63)

    def test_dspark_head_only_warm_start_freezes_shared_backbone(self):
        from draftfit.algorithms.dspark.providers import build_draft
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.training.model_loading import _draft_config_from_dict

        payload = _draft_payload("DSparkDraftModel", block_size=4)
        payload.update(
            model_type="qwen3",
            draft_vocab_size=128,
            head_dim=8,
            layer_types=["full_attention"],
        )
        payload["dflash_config"] = {
            "target_layer_ids": [1, 5],
            "mask_token_id": 127,
            "markov_rank": 8,
            "projector_type": "dspark",
            "sample_from_anchor": True,
        }
        config = _draft_config_from_dict(payload)
        with tempfile.TemporaryDirectory() as directory:
            AutoDraftModel.from_config(config).save_pretrained(directory)
            cfg = _run_config("dspark", draft_checkpoint_path=directory)
            cfg = cfg.model_copy(
                update={
                    "training": cfg.training.model_copy(
                        update={"dspark_head_only": True}
                    )
                }
            )
            draft = build_draft(cfg, config)
            trainable = [
                name for name, param in draft.named_parameters() if param.requires_grad
            ]
            frozen = [
                name
                for name, param in draft.named_parameters()
                if not param.requires_grad
            ]
            self.assertTrue(trainable)
            self.assertTrue(frozen)
            self.assertTrue(
                all(
                    name.startswith(("markov_head.", "confidence_head."))
                    for name in trainable
                )
            )
            self.assertTrue(any(name.startswith("fc.") for name in frozen))

    def test_anchor_style_dspark_requires_explicit_retraining_for_family_transfer(self):
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.training.model_loading import (
            _draft_config_from_dict,
            convert_family_draft_config,
        )

        payload = _draft_payload("DSparkDraftModel", block_size=4)
        payload.update(
            model_type="qwen3",
            draft_vocab_size=128,
            head_dim=8,
            layer_types=["full_attention"],
        )
        payload["dflash_config"] = {
            "target_layer_ids": [1, 5],
            "mask_token_id": 127,
            "markov_rank": 8,
            "projector_type": "dspark",
            "sample_from_anchor": True,
        }
        source_config = _draft_config_from_dict(payload)
        with tempfile.TemporaryDirectory() as directory:
            source = AutoDraftModel.from_config(source_config)
            source.save_pretrained(directory)
            for strategy in ("dflash", "dflash2"):
                with self.subTest(strategy=strategy):
                    with self.assertRaisesRegex(ValueError, "sample_from_anchor=false"):
                        convert_family_draft_config(source_config, strategy)
                    cfg = _run_config(strategy, draft_checkpoint_path=directory)
                    with self.assertRaisesRegex(ValueError, "sample_from_anchor=false"):
                        resolve_draft_config(
                            cfg, provider=_draft_config_provider(strategy)
                        )
                    cfg = cfg.model_copy(
                        update={
                            "training": cfg.training.model_copy(
                                update={"retrain_anchor_alignment": True}
                            )
                        }
                    )
                    target_config = resolve_draft_config(
                        cfg, provider=_draft_config_provider(strategy)
                    )
                    self.assertIs(
                        target_config.dflash_config["sample_from_anchor"], False
                    )
                    target = AutoDraftModel.from_config(target_config)
                    report = warm_start_draft_model(
                        target,
                        directory,
                        draft_config=target_config,
                        strategy=strategy,
                        retrain_anchor_alignment=True,
                    )
                    self.assertTrue(
                        report.missing_keys
                        if strategy == "dflash2"
                        else not report.missing_keys
                    )
                    for key, value in source.state_dict().items():
                        if key in target.state_dict():
                            torch.testing.assert_close(target.state_dict()[key], value)
                    with self.assertRaisesRegex(ValueError, "sample_from_anchor=false"):
                        warm_start_draft_model(
                            AutoDraftModel.from_config(target_config),
                            directory,
                            draft_config=target_config,
                            strategy=strategy,
                        )
                    runtime_dir = os.path.join(directory, f"runtime-{strategy}")
                    os.mkdir(runtime_dir)
                    torch.save(
                        {
                            "strategy": "dspark",
                            "draft_state_dict": source.state_dict(),
                            "dspark_block_size": 4,
                            "dspark_target_layer_ids": (1, 5),
                            "dspark_mask_token_id": 127,
                            "dspark_draft_num_hidden_layers": 1,
                            "dspark_sample_from_anchor": True,
                        },
                        os.path.join(runtime_dir, "training_state.pt"),
                    )
                    from_runtime = AutoDraftModel.from_config(target_config)
                    warm_start_draft_model(
                        from_runtime,
                        runtime_dir,
                        draft_config=target_config,
                        strategy=strategy,
                        retrain_anchor_alignment=True,
                    )
                    for key, value in source.state_dict().items():
                        if key in from_runtime.state_dict():
                            torch.testing.assert_close(
                                from_runtime.state_dict()[key], value
                            )

    def test_official_minicpm_dspark_warm_start_preserves_weights(self):
        from pathlib import Path

        from draftfit.modeling.draft.dspark import DSparkDraftModel
        from draftfit.training.model_loading import _draft_config_from_dict

        payload = _draft_payload("DSparkDraftModel", block_size=4)
        payload.update(
            model_type="qwen3",
            draft_vocab_size=128,
            head_dim=8,
            layer_types=["full_attention"],
        )
        payload["dflash_config"] = {
            "target_layer_ids": [1, 5],
            "mask_token_id": 127,
            "markov_rank": 8,
            "projector_type": "dspark",
        }
        config = _draft_config_from_dict(payload)
        original = DSparkDraftModel(config)
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            path = Path(directory) / "config.json"
            official = json.loads(path.read_text())
            official["architectures"] = ["Qwen3DSparkModel"]
            official.update(official.pop("dflash_config"))
            path.write_text(json.dumps(official))
            normalized = load_draft_config_source(directory)
            restored = DSparkDraftModel(normalized)
            report = warm_start_draft_model(
                restored, directory, draft_config=normalized, strategy="dspark"
            )
            self.assertFalse(report.missing_keys)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)

    def test_supplied_target_metadata_never_loads_target(self):
        from transformers import PretrainedConfig

        metadata = PretrainedConfig.from_dict(
            {"text_config": _target_config().to_dict()}
        )
        with mock.patch(
            "draftfit.modeling.target.target_utils.load_target_config",
            side_effect=AssertionError("metadata preparation must not fetch target"),
        ):
            config = resolve_draft_config(
                _run_config("dspark"),
                provider=_draft_config_provider("dspark"),
                target_config=metadata,
            )
        self.assertEqual(config.hidden_size, 32)
        self.assertEqual(config.architectures, ["DSparkDraftModel"])

    def test_config_resolution_does_not_initialize_cuda_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("DominoDraftModel", block_size=16), stream)
            script = """
import sys
import torch
import draftfit.data.preprocessing
import draftfit.modeling.auto
from draftfit.training.model_loading import load_draft_config_source

config = load_draft_config_source(sys.argv[1])
assert config.architectures == ["DominoDraftModel"]
assert config.block_size == 16
assert "yunchang" not in sys.modules
assert not torch.cuda.is_initialized()
"""
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [sys.executable, "-c", script, path],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

    def test_target_derived_defaults_match_legacy_trainers(self):
        cases = (
            ("eagle3", "LlamaForCausalLMEagle3", 1, None),
            ("peagle", "PEagleDraftModel", 4, None),
            ("dflash", "DFlashDraftModel", 1, 16),
        )
        for strategy, architecture, layers, block_size in cases:
            with self.subTest(strategy=strategy):
                cfg = _run_config(strategy)
                with mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=_target_config(),
                ):
                    resolved = resolve_draft_config(
                        cfg,
                        provider=_draft_config_provider(strategy),
                    )
                self.assertEqual(resolved.architectures, [architecture])
                self.assertEqual(resolved.num_hidden_layers, layers)
                if block_size is not None:
                    self.assertEqual(resolved.block_size, block_size)
                    self.assertEqual(resolved.num_target_layers, 12)
                    self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 1)
                    self.assertEqual(resolved.layer_types, ["full_attention"])
                    self.assertIsNone(resolved.sliding_window)
                    self.assertFalse(resolved.use_sliding_window)
                else:
                    self.assertEqual(resolved.draft_vocab_size, 32000)

    def test_family_scratch_configs_need_no_draft_source(self):
        for strategy, architecture in (
            ("dspark", "DSparkDraftModel"),
            ("dflash", "DFlashDraftModel"),
            ("dflash2", "DFlash2DraftModel"),
        ):
            with self.subTest(strategy=strategy):
                cfg = _run_config(strategy)
                self.assertIsNone(cfg.model.draft_checkpoint_path)
                self.assertIsNone(cfg.model.draft_model_config)
                resolved = resolve_draft_config(
                    cfg,
                    provider=_draft_config_provider(strategy),
                    target_config=_target_config(),
                )
                self.assertEqual(resolved.architectures, [architecture])
                self.assertEqual(resolved.vocab_size, 128)

    def test_dflash_typed_overrides_rebuild_capture_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=5, block_size=16)
            payload["dflash_config"] = {"target_layer_ids": [1, 3, 5, 7, 9]}
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
                draft_block_size=8,
            )
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("dflash"),
            )
        self.assertEqual(resolved.num_hidden_layers, 2)
        self.assertEqual(resolved.block_size, 8)
        self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 2)
        self.assertEqual(
            resolved.layer_types,
            ["full_attention", "full_attention"],
        )

    def test_dflash_layer_override_rejects_ambiguous_hybrid_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=3, block_size=16)
            payload.update(
                layer_types=[
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                sliding_window=128,
                use_sliding_window=True,
            )
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
            )
            with self.assertRaisesRegex(ValueError, "mixed DFlash layer_types"):
                resolve_draft_config(
                    cfg,
                    provider=_draft_config_provider("dflash"),
                )

    def test_local_json_and_directory_are_equivalent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("LlamaForCausalLMEagle3"), stream)
            from_file = load_draft_config_source(path)
            from_directory = load_draft_config_source(directory)
        self.assertEqual(from_file.to_dict(), from_directory.to_dict())

    def test_hf_repository_is_a_supported_config_source(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = load_draft_config_source(
                "org/draft-model",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once_with(
            "org/draft-model", cache_dir="/cache", trust_remote_code=True
        )

    def test_hf_warm_checkpoint_supplies_config_when_not_explicit(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        cfg = _run_config("eagle3", draft_checkpoint_path="org/base-draft")
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("eagle3"),
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], "org/base-draft")


class _TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.proj = torch.nn.Linear(3, 2)


class WarmStartTest(unittest.TestCase):
    def _write_runtime_state(self, directory, state, *, strategy="dflash"):
        path = os.path.join(directory, "training_state.pt")
        torch.save(
            {
                "draft_state_dict": state,
                "strategy": strategy,
                # Warm start must ignore every field below.
                "global_step": 91,
                "epoch": 7,
                "backend": {
                    "optimizer": {"state": {1: {"step": torch.tensor(91)}}},
                    "rng": torch.tensor([123]),
                },
            },
            path,
        )
        return path

    def test_specforge_checkpoint_loads_only_draft_weights(self):
        torch.manual_seed(1)
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, source.state_dict())
            with mock.patch("torch.load", wraps=torch.load) as load:
                report = warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )
        self.assertEqual(report.checkpoint_format, "specforge")
        self.assertEqual(report.loaded_keys, len(source.state_dict()))
        self.assertTrue(report.loaded_embedding)
        self.assertTrue(
            all(
                torch.equal(destination.state_dict()[key], value)
                for key, value in source.state_dict().items()
            )
        )
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_eagle_checkpoint_may_omit_target_copied_embedding(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        original_embedding = destination.embed_tokens.weight.detach().clone()
        state = {
            key: value
            for key, value in source.state_dict().items()
            if "embed" not in key
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state, strategy="eagle3")
            report = warm_start_draft_model(
                destination,
                path,
                draft_config=object(),
                strategy="eagle3",
                allow_missing_embedding=True,
            )
        self.assertFalse(report.loaded_embedding)
        self.assertIn("embed_tokens.weight", report.missing_keys)
        self.assertTrue(
            torch.equal(destination.embed_tokens.weight, original_embedding)
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_missing_non_embedding_weights_fail_closed(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        state = {"embed_tokens.weight": source.embed_tokens.weight.detach().clone()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state)
            with self.assertRaisesRegex(ValueError, "missing draft weights"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="dflash",
                )

    def test_embedding_projection_is_not_optional(self):
        model = _TinyDraft()
        model.embed_projection = torch.nn.Linear(3, 3)
        state = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("embed_projection.")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state, strategy="eagle3")
            with self.assertRaisesRegex(ValueError, "embed_projection"):
                warm_start_draft_model(
                    model,
                    path,
                    draft_config=object(),
                    strategy="eagle3",
                    allow_missing_embedding=True,
                )

    def test_runtime_checkpoint_strategy_must_match(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(
                directory, source.state_dict(), strategy="peagle"
            )
            with self.assertRaisesRegex(ValueError, "written by strategy"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=object(),
                    strategy="eagle3",
                )

    def test_pretrained_repo_uses_registered_hf_loader(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        config = mock.Mock(architectures=["DFlashDraftModel"])
        with (
            mock.patch(
                "draftfit.training.model_loading.load_draft_config_source",
                return_value=config,
            ),
            mock.patch(
                "draftfit.modeling.auto.AutoDraftModel.from_pretrained",
                return_value=(source, {"missing_keys": []}),
            ) as load,
        ):
            report = warm_start_draft_model(
                destination,
                "org/base-draft",
                draft_config=config,
                strategy="dflash",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(report.checkpoint_format, "pretrained")
        load.assert_called_once_with(
            "org/base-draft",
            config=mock.ANY,
            cache_dir="/cache",
            trust_remote_code=True,
            output_loading_info=True,
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_pretrained_load_refuses_discarded_tensors(self):
        config = mock.Mock(architectures=["DFlashDraftModel"])
        for field in ("unexpected_keys", "mismatched_keys", "error_msgs"):
            with (
                self.subTest(field=field),
                mock.patch(
                    "draftfit.training.model_loading.load_draft_config_source",
                    return_value=config,
                ),
                mock.patch(
                    "draftfit.modeling.auto.AutoDraftModel.from_pretrained",
                    return_value=(_TinyDraft(), {field: ["extra_head.weight"]}),
                ),
            ):
                with self.assertRaisesRegex(ValueError, field):
                    warm_start_draft_model(
                        _TinyDraft(),
                        "org/draft",
                        draft_config=config,
                        strategy="dflash",
                    )

    def test_local_hf_roundtrip_and_wrong_algorithm(self):
        from draftfit.modeling.auto import AutoDraftModel

        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "config.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("DFlashDraftModel", block_size=4), stream)
            config = load_draft_config_source(directory)
            source = AutoDraftModel.from_config(config)
            source.save_pretrained(directory)
            destination = AutoDraftModel.from_config(config)
            warm_start_draft_model(
                destination, directory, draft_config=config, strategy="dflash"
            )
            for key, value in source.state_dict().items():
                torch.testing.assert_close(destination.state_dict()[key], value)
            wrong_config = mock.Mock(architectures=["DSparkDraftModel"])
            with self.assertRaisesRegex(ValueError, "draft architecture"):
                warm_start_draft_model(
                    destination,
                    directory,
                    draft_config=wrong_config,
                    strategy="dspark",
                )

    def test_missing_embedding_requires_target_embedding_load(self):
        from draftfit.algorithms.model_providers import _warm_start

        model = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(
                directory,
                {"proj.weight": model.proj.weight, "proj.bias": model.proj.bias},
                strategy="eagle3",
            )
            cfg = _run_config(
                "eagle3", draft_checkpoint_path=path, load_target_embedding=False
            )
            with self.assertRaisesRegex(ValueError, "missing draft weights"):
                _warm_start(cfg, model, object(), allow_missing_embedding=True)

    def test_nested_draft_head_checkpoint_keys_are_migrated(self):
        from draftfit.modeling.auto import AutoDraftModel, AutoDraftModelConfig

        cases = (
            (
                "domino",
                {
                    **_draft_payload("DominoDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "domino",
                        "emb_dim": 16,
                        "gru_hidden_dim": 16,
                        "pure_draft_prefix_len": 0,
                        "shift_label": False,
                    },
                },
                ("prefix_gru.", "embed_proj."),
            ),
            (
                "dspark",
                {
                    **_draft_payload("DSparkDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "dspark",
                        "markov_rank": 8,
                        "markov_head_type": "vanilla",
                        "confidence_head_alpha": 1.0,
                        "confidence_head_with_markov": True,
                    },
                },
                ("markov_head.", "confidence_head."),
            ),
        )
        for strategy, payload, head_prefixes in cases:
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
            ):
                config_path = os.path.join(directory, "config.json")
                with open(config_path, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream)
                config = AutoDraftModelConfig.from_file(config_path)
                source = AutoDraftModel.from_config(config)
                destination = AutoDraftModel.from_config(config)
                checkpoint_state = {}
                migrated = []
                for key, value in source.state_dict().items():
                    checkpoint_key = key
                    for head_prefix in head_prefixes:
                        if key.startswith(head_prefix):
                            checkpoint_key = "logit_head." + key
                            migrated.append(checkpoint_key)
                            break
                    checkpoint_state[checkpoint_key] = value.detach().clone()
                self.assertTrue(migrated)
                state_path = self._write_runtime_state(
                    directory, checkpoint_state, strategy=strategy
                )
                report = warm_start_draft_model(
                    destination,
                    state_path,
                    draft_config=config,
                    strategy=strategy,
                )
                self.assertEqual(report.loaded_keys, len(checkpoint_state))
                self.assertTrue(
                    all(
                        torch.equal(destination.state_dict()[key], value)
                        for key, value in source.state_dict().items()
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
