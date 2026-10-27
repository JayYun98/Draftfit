"""Nested speculators warm starts must retain their published contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from draftfit.modeling.auto import AutoDraftModel
from draftfit.training.model_loading import (
    _draft_config_from_dict,
    convert_family_draft_config,
    load_draft_config_source,
    warm_start_draft_model,
)
from draftfit.training.official_dspark import (
    is_official_taichu_source,
    verify_taichu_target_copies,
    verify_target_copies,
)


def _payload(family="dflash", *, sliding=True):
    payload = {
        "architectures": [
            "DFlashDraftModel" if family == "dflash" else "DSparkDraftModel"
        ],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "layer_types": ["sliding_attention" if sliding else "full_attention"],
            "use_sliding_window": sliding,
            "sliding_window": 8 if sliding else None,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        },
        "aux_hidden_state_layer_ids": [1, 13, 25],
        "draft_capture_layer_indexing": "block_output",
        "block_size": 4,
        "draft_vocab_size": None,
        "mask_token_id": 31,
        "sliding_window_non_causal": False,
        "speculators_config": {
            "algorithm": family,
            "verifier": {"name_or_path": "test/target"},
        },
    }
    if family == "dflash":
        payload.update(
            projector_type="dflash", pure_draft_prefix_len=1, domino_auf=False
        )
    else:
        payload.update(
            markov_rank=4, markov_head_type="vanilla", sample_from_anchor=True
        )
    return payload


def _target():
    return SimpleNamespace(num_hidden_layers=28, hidden_size=16, vocab_size=32)


class SpeculatorsWarmStartTest(unittest.TestCase):
    def test_taichu_keeps_legacy_default_and_rejects_indexing_conflicts(self):
        payload = _payload("dspark")
        del payload["draft_capture_layer_indexing"]
        payload["official_dspark_format"] = "taichu_speculators"
        native = _draft_config_from_dict(payload, target_config=_target())
        self.assertEqual(native.dflash_config["target_layer_ids"], [1, 13, 25])
        for source in (payload, native.to_dict()):
            source["draft_capture_layer_indexing"] = "embedding_inclusive"
            with (
                self.subTest(nested="transformer_layer_config" in source),
                self.assertRaisesRegex(ValueError, "Taichu.*indexing"),
            ):
                _draft_config_from_dict(source, target_config=_target())

    def test_generic_nested_requires_explicit_capture_indexing(self):
        for family in ("dspark", "dflash"):
            payload = _payload(family)
            del payload["draft_capture_layer_indexing"]
            with (
                self.subTest(family=family),
                self.assertRaisesRegex(ValueError, "draft_capture_layer_indexing"),
            ):
                _draft_config_from_dict(payload, target_config=_target())

    def test_capture_indexing_maps_original_ids_and_survives_reload(self):
        for indexing, source_ids in (
            ("block_output", [0, 13, 27]),
            ("embedding_inclusive", [1, 14, 28]),
        ):
            with self.subTest(indexing=indexing), tempfile.TemporaryDirectory() as d:
                payload = _payload()
                payload.update(
                    draft_capture_layer_indexing=indexing,
                    aux_hidden_state_layer_ids=source_ids,
                )
                config = _draft_config_from_dict(payload, target_config=_target())
                self.assertEqual(config.draft_capture_source_layer_ids, source_ids)
                self.assertEqual(config.draft_capture_layer_indexing, indexing)
                self.assertEqual(config.dflash_config["target_layer_ids"], [0, 13, 27])
                path = str(Path(d, "config.json"))
                config.to_json_file(path)
                reloaded = load_draft_config_source(path)
                self.assertEqual(reloaded.draft_capture_source_layer_ids, source_ids)
                self.assertEqual(
                    reloaded.dflash_config["target_layer_ids"], [0, 13, 27]
                )
                with self.assertRaisesRegex(ValueError, "indexing"):
                    load_draft_config_source(
                        path,
                        draft_capture_layer_indexing=(
                            "embedding_inclusive"
                            if indexing == "block_output"
                            else "block_output"
                        ),
                    )
                legacy = config.to_dict()
                del legacy["draft_capture_source_layer_ids"]
                with self.assertRaisesRegex(ValueError, "source.*layer|provenance"):
                    _draft_config_from_dict(legacy)

    def test_invalid_capture_conventions_and_indices_fail(self):
        for indexing, taps in (
            ("unknown", [1]),
            ("embedding_inclusive", [0, 1]),
            ("block_output", [1, 1]),
            ("block_output", [True]),
            ("block_output", []),
            ("embedding_inclusive", [29]),
        ):
            payload = _payload()
            payload.update(
                draft_capture_layer_indexing=indexing,
                aux_hidden_state_layer_ids=taps,
            )
            with (
                self.subTest(indexing=indexing, taps=taps),
                self.assertRaises(ValueError),
            ):
                _draft_config_from_dict(payload, target_config=_target())

    def test_run_declaration_binds_source_and_rejects_physical_override(self):
        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.config import Config
        from draftfit.training.model_loading import resolve_draft_config

        with tempfile.TemporaryDirectory() as d:
            payload = _payload()
            del payload["draft_capture_layer_indexing"]
            path = Path(d, "draft.json")
            path.write_text(json.dumps(payload))
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "/unused-target",
                        "draft_model_config": str(path),
                        "draft_capture_layer_indexing": "embedding_inclusive",
                    },
                    "data": {"hidden_states_path": "/features"},
                    "training": {"strategy": "dflash"},
                }
            )
            provider = (
                builtin_algorithm_registry()
                .resolve("dflash")
                .providers.model.draft_config
            )
            config = resolve_draft_config(
                cfg, provider=provider, target_config=_target()
            )
            self.assertEqual(config.dflash_config["target_layer_ids"], [0, 12, 24])
            cfg.model.target_layer_ids = [1, 13, 25]
            with self.assertRaisesRegex(ValueError, "capture|target_layer_ids"):
                resolve_draft_config(cfg, provider=provider, target_config=_target())

    def test_warm_start_preserves_and_checks_capture_provenance(self):
        for source_declares in (False, True):
            with (
                self.subTest(source_declares=source_declares),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                target = root / "target"
                target.mkdir()
                (target / "config.json").write_text(
                    json.dumps(
                        {
                            **vars(_target()),
                            "model_type": "qwen3",
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "head_dim": 8,
                        }
                    )
                )
                payload = _payload()
                if source_declares:
                    payload["draft_capture_layer_indexing"] = "embedding_inclusive"
                else:
                    del payload["draft_capture_layer_indexing"]
                payload["speculators_config"]["verifier"]["name_or_path"] = str(target)
                (root / "config.json").write_text(json.dumps(payload))
                config = _draft_config_from_dict(
                    payload,
                    target_config=_target(),
                    draft_capture_layer_indexing="embedding_inclusive",
                )
                original = AutoDraftModel.from_config(config)
                save_file(original.state_dict(), str(root / "model.safetensors"))
                flat = root / "flat.json"
                config.to_json_file(str(flat))
                config = load_draft_config_source(str(flat))
                destination = AutoDraftModel.from_config(config)
                report = warm_start_draft_model(
                    destination, str(root), draft_config=config, strategy="dflash"
                )
                self.assertFalse(report.missing_keys)
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(
                        destination.state_dict()[key], value, rtol=0, atol=0
                    )
                # Same physical taps, but a different account of the source IDs.
                config.draft_capture_layer_indexing = "block_output"
                config.draft_capture_source_layer_ids = list(
                    config.dflash_config["target_layer_ids"]
                )
                with self.assertRaisesRegex(
                    ValueError, "capture|indexing|target_layer_ids"
                ):
                    warm_start_draft_model(
                        destination, str(root), draft_config=config, strategy="dflash"
                    )

    def test_existing_baseline_raw_source_uses_arm_capture_provenance(self):
        from draftfit.algorithms.common.dflash_family_model import (
            OnlineDFlashModel,
            OnlineDSparkModel,
        )
        from draftfit.training.joint_verification import (
            _load_existing_draft,
            preflight_existing_drafts,
        )

        for family, wrapper in (
            ("dflash", OnlineDFlashModel),
            ("dspark", OnlineDSparkModel),
        ):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                target = root / "target"
                target.mkdir()
                (target / "config.json").write_text(
                    json.dumps({**vars(_target()), "model_type": "qwen3"})
                )
                payload = _payload(family)
                del payload["draft_capture_layer_indexing"]
                # The arm's bound target must take precedence over a stale verifier path.
                payload["speculators_config"]["verifier"]["name_or_path"] = str(
                    root / "absent-target"
                )
                config = _draft_config_from_dict(
                    payload,
                    target_config=_target(),
                    draft_capture_layer_indexing="embedding_inclusive",
                )
                config.speculators_target_model_path = str(target)
                original = AutoDraftModel.from_config(config)
                (root / "config.json").write_text(json.dumps(payload))
                save_file(
                    original.state_dict(),
                    str(root / "model.safetensors"),
                    metadata={"format": "pt"},
                )
                model = wrapper(
                    draft_model=original,
                    target_lm_head=torch.nn.Linear(16, 32, bias=False),
                    target_embed_tokens=torch.nn.Embedding(32, 16),
                    mask_token_id=31,
                    block_size=4,
                    attention_backend="sdpa",
                )
                bundle = SimpleNamespace(model=model, draft_model=original)
                receipt = preflight_existing_drafts(
                    [SimpleNamespace(name=family)],
                    [bundle],
                    {"existing_drafts": {family: d}},
                )
                self.assertIn("config.json", receipt[family]["files"])
                loaded = _load_existing_draft(bundle, d)
                self.assertEqual(loaded.target_layer_ids, [0, 12, 24])
                self.assertEqual(
                    loaded.config.draft_capture_source_layer_ids, [1, 13, 25]
                )
                self.assertEqual(
                    loaded.config.draft_capture_layer_indexing, "embedding_inclusive"
                )
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(
                        loaded.state_dict()[key], value, rtol=0, atol=0
                    )
                # A self-declared different convention may not be reinterpreted by the arm.
                payload["draft_capture_layer_indexing"] = "block_output"
                (root / "config.json").write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "indexing|provenance"):
                    _load_existing_draft(bundle, d)

    def test_dflash2_export_reload_keeps_capture_and_published_masks(self):
        from torch import nn

        from draftfit.algorithms.common.dflash_family_model import OnlineDFlash2Model
        from tests.test_utils.test_speculators_mask import _published_reference

        payload = _payload(sliding=False)
        payload["draft_capture_layer_indexing"] = "embedding_inclusive"
        config = convert_family_draft_config(
            _draft_config_from_dict(payload, target_config=_target()), "dflash2"
        )
        with tempfile.TemporaryDirectory() as d:
            original = AutoDraftModel.from_config(config)
            original.save_pretrained(d)
            reloaded = load_draft_config_source(d)
            draft = AutoDraftModel.from_config(reloaded)
            warm_start_draft_model(draft, d, draft_config=reloaded, strategy="dflash2")
            self.assertEqual(reloaded.draft_capture_source_layer_ids, [1, 13, 25])
            self.assertEqual(reloaded.dflash_config["target_layer_ids"], [0, 12, 24])
            self.assertEqual(reloaded.dflash_config["pure_draft_prefix_len"], 1)
            anchors, keep = torch.tensor([[11]]), torch.tensor([[True]])
            for backend in ("sdpa", "flex_attention"):
                model = OnlineDFlash2Model(
                    draft_model=draft,
                    target_lm_head=nn.Linear(16, 32, bias=False),
                    target_embed_tokens=nn.Embedding(32, 16),
                    mask_token_id=31,
                    block_size=4,
                    attention_backend=backend,
                )
                masks = []

                def capture(module, args, kwargs):
                    masks.append(kwargs["attention_mask"])

                def forward(**kwargs):
                    masks.append(kwargs["attention_mask"])
                    return kwargs["noise_embedding"]

                if backend == "sdpa":
                    handle = draft.register_forward_pre_hook(capture, with_kwargs=True)
                    context = mock.patch.object(
                        model, "_sample_anchor_positions", return_value=(anchors, keep)
                    )
                else:
                    handle = None
                    context = mock.patch.object(draft, "forward", side_effect=forward)
                with (
                    context,
                    mock.patch.object(
                        model, "_sample_anchor_positions", return_value=(anchors, keep)
                    ),
                ):
                    model._forward_draft_blocks(
                        input_ids=torch.arange(16).unsqueeze(0),
                        hidden_states=torch.randn(1, 16, 48),
                        loss_mask=torch.ones(1, 16),
                    )
                if handle is not None:
                    handle.remove()
                expected = _published_reference(anchors, keep, 16, 4, None, False)
                actual = masks[0]
                if backend == "flex_attention":
                    actual = actual.mask_mod(
                        0,
                        0,
                        torch.arange(4).unsqueeze(1),
                        torch.arange(20).unsqueeze(0),
                    )
                    expected = expected[0, 0]
                self.assertTrue(torch.equal(actual, expected), backend)

    def test_dflash2_sliding_conversion_fails_before_export(self):
        config = _draft_config_from_dict(_payload(), target_config=_target())
        with self.assertRaisesRegex(
            ValueError, "DFlash2.*anchor.relative|anchor.relative.*DFlash2"
        ):
            convert_family_draft_config(config, "dflash2")

    def test_runtime_state_warm_start_checks_saved_capture_provenance(self):
        payload = _payload()
        payload["draft_capture_layer_indexing"] = "embedding_inclusive"
        config = _draft_config_from_dict(payload, target_config=_target())
        original = AutoDraftModel.from_config(config)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "config.json")
            config.to_json_file(str(path))
            torch.save(
                {"strategy": "dflash", "draft_state_dict": original.state_dict()},
                Path(d, "training_state.pt"),
            )
            destination = AutoDraftModel.from_config(config)
            warm_start_draft_model(
                destination, d, draft_config=config, strategy="dflash"
            )
            source = config.to_dict()
            source["draft_capture_layer_indexing"] = "block_output"
            source["draft_capture_source_layer_ids"] = source["dflash_config"][
                "target_layer_ids"
            ]
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "capture|indexing"):
                warm_start_draft_model(
                    destination, d, draft_config=config, strategy="dflash"
                )
            path.unlink()
            with self.assertRaisesRegex(ValueError, "source.*config|provenance"):
                warm_start_draft_model(
                    destination, d, draft_config=config, strategy="dflash"
                )

    def test_nested_families_use_actual_depth_and_keep_semantics(self):
        for family in ("dspark", "dflash"):
            with self.subTest(family=family):
                payload = _payload(family)
                original = copy.deepcopy(payload)
                config = _draft_config_from_dict(payload, target_config=_target())
                self.assertEqual(config.num_target_layers, 28)
                self.assertEqual(config.num_hidden_layers, 1)
                self.assertEqual(config.layer_types, ["sliding_attention"])
                self.assertEqual(config.rope_parameters["rope_theta"], 1000000)
                self.assertEqual(config.dflash_config["target_layer_ids"], [1, 13, 25])
                self.assertFalse(config.dflash_config["sliding_window_non_causal"])
                self.assertEqual(payload, original)
                if family == "dflash":
                    self.assertEqual(config.dflash_config["pure_draft_prefix_len"], 1)
                else:
                    self.assertTrue(config.dflash_config["sample_from_anchor"])

    def test_missing_depth_fails_instead_of_using_last_tap(self):
        with self.assertRaisesRegex(ValueError, "target.*metadata"):
            _draft_config_from_dict(_payload("dspark"))

    def test_metadata_conflicts_and_reduced_vocabulary_fail(self):
        for changes, error in [
            ({"num_target_layers": 26}, "num_target_layers"),
            ({"draft_vocab_size": 16}, "reduced.*vocab"),
            ({"hidden_size": 17}, "conflicting nested"),
        ]:
            payload = {**_payload(), **changes}
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, error),
            ):
                _draft_config_from_dict(payload, target_config=_target())

    def test_loader_reads_target_metadata_without_importing_remote_config(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(_payload()))
            with (
                mock.patch(
                    "draftfit.modeling.target.target_utils.load_target_config",
                    return_value=_target(),
                ) as target_loader,
                mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    side_effect=AssertionError("remote config executed"),
                ),
            ):
                config = load_draft_config_source(
                    directory,
                    target_model_path="/actual/target",
                    target_revision="pinned-target",
                )
            self.assertEqual(config.num_target_layers, 28)
            self.assertEqual(
                target_loader.call_args.kwargs["revision"], "pinned-target"
            )
            self.assertFalse(target_loader.call_args.kwargs["trust_remote_code"])

    def test_generic_nested_source_is_not_taichu(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(_payload("dspark")))
            self.assertFalse(is_official_taichu_source(directory))

    def test_native_resave_is_not_the_original_taichu_export(self):
        payload = {**_payload("dspark"), "official_dspark_format": "taichu_speculators"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text(json.dumps(payload))
            self.assertTrue(is_official_taichu_source(directory))
            config = _draft_config_from_dict(payload, target_config=_target())
            config.to_json_file(str(path))
            self.assertFalse(is_official_taichu_source(directory))

    def test_nested_sliding_source_warm_starts_both_native_families(self):
        for family in ("dspark", "dflash"):
            payload = _payload(family)
            config = _draft_config_from_dict(payload, target_config=_target())
            self.assertEqual(config.speculators_attention_semantics, "anchor_relative")
            original = AutoDraftModel.from_config(config)
            with (
                tempfile.TemporaryDirectory() as directory,
                tempfile.TemporaryDirectory() as target_dir,
            ):
                Path(directory, "config.json").write_text(json.dumps(payload))
                Path(target_dir, "config.json").write_text(
                    json.dumps(
                        {
                            **vars(_target()),
                            "model_type": "qwen3",
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "head_dim": 8,
                        }
                    )
                )
                config.speculators_target_model_path = target_dir
                state = {
                    key: value.detach().clone().contiguous()
                    for key, value in original.state_dict().items()
                }
                save_file(state, str(Path(directory, "model.safetensors")))
                destination = AutoDraftModel.from_config(config)
                report = warm_start_draft_model(
                    destination, directory, draft_config=config, strategy=family
                )
                self.assertFalse(report.missing_keys)
                for key, value in state.items():
                    torch.testing.assert_close(
                        destination.state_dict()[key], value, rtol=0, atol=0
                    )
                config.speculators_attention_semantics = "query_relative"
                with self.assertRaisesRegex(ValueError, "anchor_relative"):
                    warm_start_draft_model(
                        destination, directory, draft_config=config, strategy=family
                    )

    def test_plain_dflash_prefix_one_keeps_backbone_and_loss_semantics(self):
        # Upstream core.py uses pure_draft_prefix_len only in the Domino metrics
        # branch; plain DFlash uses same-position labels and excludes slot zero.
        from draftfit.algorithms.common.dflash_family_model import OnlineDFlashModel

        config = _draft_config_from_dict(_payload(), target_config=_target())
        config._attn_implementation = "sdpa"
        draft = AutoDraftModel.from_config(config)
        wrapper = OnlineDFlashModel(
            draft,
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.Embedding(32, 16),
            mask_token_id=31,
            block_size=4,
            attention_backend="sdpa",
        )
        inputs = dict(
            input_ids=torch.arange(16).unsqueeze(0),
            hidden_states=torch.randn(1, 16, 48),
            loss_mask=torch.ones(1, 16),
        )
        with mock.patch.object(
            wrapper,
            "_sample_anchor_positions",
            return_value=(torch.tensor([[6]]), torch.tensor([[True]])),
        ):
            loss_one, _, _ = wrapper(**inputs)
            draft.pure_draft_prefix_len = 0
            loss_zero, _, _ = wrapper(**inputs)
        torch.testing.assert_close(loss_one, loss_zero, rtol=0, atol=0)
        loss_one.backward()
        self.assertTrue(torch.isfinite(draft.fc.weight.grad).all())

    def test_exact_target_copies_are_checked_and_arbitrary_extras_fail(self):
        payload = _payload(sliding=False)
        config = _draft_config_from_dict(payload, target_config=_target())
        original = AutoDraftModel.from_config(config)
        state = {
            key: value.detach().clone().contiguous()
            for key, value in original.state_dict().items()
        }
        embed = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            config.speculators_target_model_path = target_dir
            Path(target_dir, "config.json").write_text(
                json.dumps(
                    {
                        **vars(_target()),
                        "model_type": "qwen3",
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 8,
                        "tie_word_embeddings": False,
                    }
                )
            )
            save_file(
                {
                    "model.embed_tokens.weight": embed.clone(),
                    "lm_head.weight": embed.clone(),
                },
                str(Path(target_dir, "model.safetensors")),
            )
            config.to_json_file(str(Path(directory, "config.json")))
            state["embed_tokens.weight"] = embed
            state["lm_head.weight"] = embed.clone()
            path = str(Path(directory, "model.safetensors"))
            save_file(state, path)
            destination = AutoDraftModel.from_config(config)
            report = warm_start_draft_model(
                destination, directory, draft_config=config, strategy="dflash"
            )
            self.assertEqual(report.loaded_keys, len(original.state_dict()))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(
                    destination.state_dict()[key], value, rtol=0, atol=0
                )
            state["lm_head.weight"][0, 0] += 1
            save_file(state, path)
            with self.assertRaisesRegex(ValueError, "lm_head.weight differs"):
                warm_start_draft_model(
                    destination, directory, draft_config=config, strategy="dflash"
                )
            state["lm_head.weight"] = embed.clone()
            state["arbitrary.weight"] = torch.zeros(1)
            save_file(state, path)
            with self.assertRaisesRegex(ValueError, "unexpected_keys"):
                warm_start_draft_model(
                    destination, directory, draft_config=config, strategy="dflash"
                )

    def test_target_copy_comparison_does_not_round_away_mismatch(self):
        target = torch.ones(2, 2, dtype=torch.bfloat16)
        parts = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=target),
            lm_head=SimpleNamespace(weight=target),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "model.safetensors"))
            save_file({"embed_tokens.weight": target.float() + 0.001}, path)
            with self.assertRaisesRegex(ValueError, "embed_tokens.weight differs"):
                verify_target_copies(directory, parts)
            save_file({"embed_tokens.weight": target}, path)
            verify_target_copies(directory, parts)

    def test_taichu_requires_both_copies(self):
        parts = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(32, 16)),
            lm_head=SimpleNamespace(weight=torch.zeros(32, 16)),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"embed_tokens.weight": parts.embed_tokens.weight},
                str(Path(directory, "model.safetensors")),
            )
            with self.assertRaisesRegex(ValueError, "missing lm_head.weight"):
                verify_taichu_target_copies(directory, parts)


if __name__ == "__main__":
    unittest.main()
