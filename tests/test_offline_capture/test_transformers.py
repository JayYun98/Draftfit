import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import (
    Lfm2Config,
    Lfm2Model,
    Lfm2MoeConfig,
    Lfm2MoeModel,
    LlamaConfig,
    LlamaModel,
    Qwen2Config,
    Qwen2Model,
    Qwen3Config,
    Qwen3Model,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from draftfit.offline_capture.transformers import OfflineTransformersCapture


class TransformersCaptureTest(unittest.TestCase):
    def test_cached_hub_snapshot_uses_pinned_repo_for_remote_code(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory) / "models--TaichuAI--ZDTaichu5.0-9B"
                / "snapshots" / revision
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            with patch("transformers.AutoModel.from_pretrained", return_value=self.model()) as load:
                OfflineTransformersCapture.from_pretrained(
                    str(snapshot), revision=revision, trust_remote_code=True,
                    torch_dtype=torch.float32, device="cpu",
                )
            load.assert_called_once_with(
                "TaichuAI/ZDTaichu5.0-9B", revision=revision,
                trust_remote_code=True, torch_dtype=torch.float32,
                cache_dir=directory, attn_implementation="sdpa",
                local_files_only=True,
            )

    def test_cached_hub_snapshot_rejects_mismatched_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory) / "models--TaichuAI--ZDTaichu5.0-9B"
                / "snapshots" / ("a" * 40)
            )
            snapshot.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "does not match"):
                OfflineTransformersCapture.from_pretrained(
                    str(snapshot), revision="b" * 40, trust_remote_code=True,
                )

    def test_taichu_text_model_capture(self):
        from types import SimpleNamespace

        config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
        )
        text = Qwen3_5TextModel(config)
        wrapper = SimpleNamespace(base_model=SimpleNamespace(
            config=SimpleNamespace(model_type="zdtaichu5_0"),
            language_model=SimpleNamespace(model=text),
        ))
        teacher = OfflineTransformersCapture(wrapper)
        teacher.set_capture_layers([0, 1], capture_method="dspark")
        ids = torch.tensor([[1, 2, 3, 4]])
        result = teacher.capture(
            input_ids=ids, attention_mask=torch.ones_like(ids), loss_mask=torch.ones_like(ids)
        )
        torch.testing.assert_close(
            text.norm(result.hidden_states[..., 16:]), result.last_hidden_states
        )

    def test_lfm_conv_and_moe_capture_unnormalized_taps(self):
        for config_cls, model_cls in ((Lfm2Config, Lfm2Model), (Lfm2MoeConfig, Lfm2MoeModel)):
            with self.subTest(model=model_cls.__name__):
                options = dict(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, layer_types=["conv", "full_attention"],
                )
                if config_cls is Lfm2MoeConfig:
                    options.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16)
                model = model_cls(config_cls(**options))
                teacher = OfflineTransformersCapture(model)
                teacher.set_capture_layers([0, 1], capture_method="dspark")
                ids = torch.tensor([[1, 2, 3, 4]])
                result = teacher.capture(
                    input_ids=ids, attention_mask=torch.ones_like(ids), loss_mask=torch.ones_like(ids)
                )
                torch.testing.assert_close(
                    model.embedding_norm(result.hidden_states[..., 16:]),
                    result.last_hidden_states,
                )
                self.assertEqual(result.hidden_states.shape, (1, 4, 32))

    def test_vlm_text_decoder_selection_and_xing_reduction(self):
        from types import SimpleNamespace

        decoder = self.model()
        for family, language_model in (
            ("agnes", decoder),
            ("qwen3_5", decoder),
            ("zdtaichu5_0", SimpleNamespace(base_model=decoder)),
        ):
            wrapper = SimpleNamespace(
                base_model=SimpleNamespace(
                    config=SimpleNamespace(model_type=family),
                    language_model=language_model,
                )
            )
            teacher = OfflineTransformersCapture(wrapper)
            teacher.set_capture_layers([0, 2], capture_method="dspark")
            result = teacher.capture(**self.batch())
            self.assertEqual(result.hidden_states.shape, (2, 4, 32))
        decoder.config.model_type = "xing4_0"
        decoder.config.hc_mult = 4
        teacher = OfflineTransformersCapture(decoder)
        streams = torch.randn(2, 3, 4, 16)
        torch.testing.assert_close(teacher._capture_output(streams), streams.mean(2))
        with self.assertRaisesRegex(ValueError, "hyperconnection"):
            teacher._capture_output(streams[:, :, :2])

    def model(self, config_cls=LlamaConfig, model_cls=LlamaModel):
        torch.manual_seed(3)
        return model_cls(
            config_cls(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=32,
            )
        )

    def batch(self):
        return dict(
            input_ids=torch.tensor([[1, 2, 3, 0], [0, 4, 5, 6]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]]),
            loss_mask=torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]]),
        )

    def test_real_decoders_capture_raw_blocks_and_normalized_final(self):
        for config_cls, model_cls in (
            (LlamaConfig, LlamaModel),
            (Qwen2Config, Qwen2Model),
            (Qwen3Config, Qwen3Model),
        ):
            with self.subTest(model=model_cls.__name__):
                model = self.model(config_cls, model_cls)
                teacher = OfflineTransformersCapture(model)
                teacher.set_capture_layers([0, 2], capture_method="dspark")
                batch = self.batch()
                hooks_before = [dict(layer._forward_hooks) for layer in model.layers]
                result = teacher.capture(**batch)
                self.assertEqual(
                    [dict(layer._forward_hooks) for layer in model.layers], hooks_before
                )
                self.assertEqual(result.hidden_states.shape, (2, 4, 32))
                with torch.no_grad():
                    reference = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(
                            0
                        ),
                        output_hidden_states=True,
                        use_cache=False,
                    )
                torch.testing.assert_close(
                    result.hidden_states[..., :16], reference.hidden_states[1]
                )
                torch.testing.assert_close(
                    model.norm(result.hidden_states[..., 16:]),
                    result.last_hidden_states,
                )
                torch.testing.assert_close(
                    result.last_hidden_states, reference.last_hidden_state
                )
                self.assertFalse(
                    torch.allclose(
                        result.hidden_states[..., 16:], result.last_hidden_states
                    )
                )
                self.assertFalse(result.hidden_states.requires_grad)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                # Both left and right padding preserve the actual sequence's features.
                for row in range(2):
                    ids = batch["input_ids"][row][
                        batch["attention_mask"][row].bool()
                    ].unsqueeze(0)
                    single = teacher.capture(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        loss_mask=torch.ones_like(ids),
                    )
                    torch.testing.assert_close(
                        result.hidden_states[row][batch["attention_mask"][row].bool()],
                        single.hidden_states[0],
                    )

    def test_local_pretrained_loading_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model().save_pretrained(directory)
            teacher = OfflineTransformersCapture.from_pretrained(
                directory,
                revision="main",
                trust_remote_code=False,
                torch_dtype=torch.float32,
                device="cpu",
                cache_dir=directory,
                attn_implementation="eager",
            )
        with self.assertRaisesRegex(ValueError, "set_capture_layers"):
            teacher.capture(**self.batch())
        for layers in ([True], [3], [1, 1], [2, 0], []):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                teacher.set_capture_layers(layers, capture_method="dflash")
        teacher.set_capture_layers([0, 1, 2], capture_method="eagle3")
        for replacement in (
            {"loss_mask": torch.ones((2, 4))},
            {"attention_mask": torch.zeros((2, 4))},
            {"loss_mask": torch.full((2, 4), float("nan"))},
            {"input_ids": torch.ones((2, 4), dtype=torch.float32)},
            {"input_ids": torch.full((2, 4), 32)},
            {"attention_mask": torch.ones((2, 3))},
            {"attention_mask": torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]])},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                teacher.capture(**(self.batch() | replacement))

    def test_failed_forward_removes_capture_hooks(self):
        model = self.model()
        teacher = OfflineTransformersCapture(model)
        teacher.set_capture_layers([0], capture_method="dflash")
        before = dict(model.layers[0]._forward_hooks)
        with patch.object(model, "forward", side_effect=RuntimeError("failed forward")):
            with self.assertRaisesRegex(RuntimeError, "failed forward"):
                teacher.capture(**self.batch())
        self.assertEqual(dict(model.layers[0]._forward_hooks), before)

    def test_nonfinite_teacher_output_is_rejected(self):
        model = self.model()
        teacher = OfflineTransformersCapture(model)
        teacher.set_capture_layers([0], capture_method="dflash")
        with torch.no_grad():
            model.norm.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            teacher.capture(**self.batch())


if __name__ == "__main__":
    unittest.main()
