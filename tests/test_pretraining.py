"""Tiny target weights and real corpus exercise all three family pretrain paths."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from draftfit.algorithms.common.dflash_family_model import OnlineDSparkModel
from draftfit.cli import main
from draftfit.modeling.auto import AutoDraftModel
from draftfit.pretraining import expand_aux_layers, pretrain
from draftfit.training.model_loading import _draft_config_from_dict, warm_start_draft_model


class RawPretrainingTest(unittest.TestCase):
    def test_all_family_updates_and_exact_widening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            Qwen3ForCausalLM(Qwen3Config(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8,
                max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
            )).save_pretrained(target)
            tokenizer = Tokenizer(models.WordLevel({
                "[PAD]": 0, "[UNK]": 1, "[EOS]": 2,
                "hello": 3, "world": 4, "again": 5,
            }, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, pad_token="[PAD]",
                unk_token="[UNK]", eos_token="[EOS]",
            ).save_pretrained(target)
            corpus = root / "train.jsonl"
            corpus.write_text(''.join(json.dumps({"text": "hello world again " * 8}) + "\n" for _ in range(3)))
            validation = root / "validation.jsonl"
            validation.write_text(json.dumps({"text": "world again hello " * 8}) + "\n")
            for strategy, architecture in (
                ("dflash", "DFlashDraftModel"),
                ("dspark", "DSparkDraftModel"),
                ("dflash2", "DFlash2DraftModel"),
            ):
                with self.subTest(strategy=strategy):
                    config = _draft_config_from_dict({
                        "architectures": [architecture],
                        "model_type": "qwen3_dflash2" if strategy == "dflash2" else "qwen3",
                        "vocab_size": 32, "hidden_size": 16,
                        "intermediate_size": 32, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "num_key_value_heads": 1,
                        "head_dim": 8, "max_position_embeddings": 128,
                        "layer_types": ["full_attention"],
                        "block_size": 4, "num_target_layers": 2,
                        "target_num_hidden_layers": 2,
                        "selector_rank": 4, "selector_top_k": 4,
                        "conv_group_size": 4,
                        "dflash_config": {
                            "target_layer_ids": [0, 1], "mask_token_id": 31,
                            "projector_type": "dspark" if strategy == "dspark" else None,
                            "markov_rank": 4, "sample_from_anchor": strategy == "dspark",
                        },
                    })
                    config_path = root / f"{strategy}.json"
                    config_path.write_text(config.to_json_string())
                    output = root / f"{strategy}-pretrain"
                    options = dict(
                        target=str(target), draft_config=str(config_path),
                        strategy=strategy, corpus=str(corpus), output=str(output),
                        steps=1, sequence_length=8, validation=str(validation),
                    )
                    if strategy == "dflash":
                        self.assertEqual(main([
                            "pretrain", "--target", str(target),
                            "--draft-config", str(config_path),
                            "--strategy", strategy, "--corpus", str(corpus),
                            "--validation", str(validation), "--output", str(output),
                            "--steps", "1", "--sequence-length", "8",
                        ]), 0)
                        result = json.loads((output / "summary.json").read_text())
                    else:
                        result = pretrain(**options)
                    self.assertGreater(result["train"]["grad_norm"], 0)
                    self.assertGreater(result["projection_update_norm"], 0)
                    self.assertEqual(len(result["train"]["position_accuracy"]), 4)
                    if strategy != "dspark":
                        self.assertEqual(result["train"]["position_ce"][0], 0.0)
                    self.assertTrue(torch.isfinite(torch.tensor(result["validation"]["loss"])))
                    self.assertGreater(result["validation"]["sequences"], 1)
                    self.assertGreater(result["validation"]["supervised_tokens"], 0)
                    source = AutoDraftModel.from_pretrained(output)
                    self.assertEqual(source.fc.in_features, 16)
                    widened = root / f"{strategy}-widened"
                    if strategy == "dflash":
                        self.assertEqual(main([
                            "expand-aux-layers", "--checkpoint", str(output),
                            "--target-layer-ids", "1", "0", "--output", str(widened),
                        ]), 0)
                    else:
                        expand_aux_layers(str(output), str(widened), [1, 0])
                    destination = AutoDraftModel.from_pretrained(widened)
                    self.assertEqual(destination.fc.in_features, 32)
                    self.assertTrue(torch.equal(destination.fc.weight[:, :16], torch.zeros_like(destination.fc.weight[:, :16])))
                    self.assertTrue(torch.equal(destination.fc.weight[:, 16:], source.fc.weight))
                    for name, tensor in source.state_dict().items():
                        if name != "fc.weight":
                            self.assertTrue(torch.equal(tensor, destination.state_dict()[name]), name)
                    feature = torch.randn(2, 3, 16)
                    torch.testing.assert_close(
                        source.fc(feature),
                        destination.fc(torch.cat([torch.randn_like(feature), feature], -1)),
                        rtol=1e-6, atol=1e-7,
                    )
                    context = torch.randn(1, 8, 16)
                    noise = torch.randn(1, 4, 16)
                    positions = torch.arange(12).unsqueeze(0)
                    with torch.no_grad():
                        original_hidden = source(
                            position_ids=positions, noise_embedding=noise,
                            target_hidden=context,
                        )
                        expanded_hidden = destination(
                            position_ids=positions, noise_embedding=noise,
                            target_hidden=torch.cat([torch.randn_like(context), context], -1),
                        )
                    torch.testing.assert_close(original_hidden, expanded_hidden, rtol=1e-6, atol=1e-7)
                    report = warm_start_draft_model(
                        destination, str(widened),
                        draft_config=destination.config, strategy=strategy,
                    )
                    self.assertGreater(report.loaded_keys, 0)
            with self.assertRaisesRegex(ValueError, "including 0"):
                expand_aux_layers(str(output), str(root / "bad-taps"), [1])
            with self.assertRaisesRegex(ValueError, "separate held-out"):
                pretrain(**{**options, "validation": str(corpus), "output": str(root / "bad-validation")})
            short = root / "short.jsonl"
            short.write_text('{"text":"hello"}\n')
            failed_output = root / "too-short"
            with self.assertRaisesRegex(ValueError, "corpus exhausted"):
                pretrain(**{**options, "corpus": str(short), "validation": None,
                            "output": str(failed_output)})
            self.assertFalse(failed_output.exists())

    def test_dspark_point_mass_tv_is_exact(self):
        config = _draft_config_from_dict({
            "architectures": ["DSparkDraftModel"], "model_type": "qwen3",
            "vocab_size": 16, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 8,
            "layer_types": ["full_attention"], "block_size": 4,
            "num_target_layers": 2,
            "dflash_config": {"target_layer_ids": [0], "mask_token_id": 15,
                              "projector_type": "dspark", "markov_rank": 4,
                              "enable_confidence_head": True},
        })
        draft = AutoDraftModel.from_config(config)
        wrapper = OnlineDSparkModel(
            draft_model=draft, target_lm_head=torch.nn.Linear(16, 16, bias=False),
            target_embed_tokens=torch.nn.Embedding(16, 16), mask_token_id=15,
            block_size=4, attention_backend="sdpa", hard_targets=True,
        )
        hidden = torch.randn(1, 1, 4, 16)
        previous = torch.tensor([[[1, 2, 3, 4]]])
        targets = torch.tensor([[[2, 3, 4, 5]]])
        weights = torch.ones(1, 1, 4)
        terms = wrapper._dspark_objective_chunk_terms(
            hidden, previous, targets, weights, weights.bool(), None,
        )
        logits = draft.apply_logits_head(wrapper.lm_head(hidden),
                                         prev_token_ids=previous, hidden_states=hidden)
        p_true = logits.softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(terms[1], (2 * (1 - p_true)).sum())
        self.assertGreater(float(terms[2].detach()), 0)


if __name__ == "__main__":
    unittest.main()
