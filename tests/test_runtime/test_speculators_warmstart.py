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
    _load_pretrained_draft_state,
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
