import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import (
    LlamaConfig,
    LlamaModel,
    Qwen2Config,
    Qwen2Model,
    Qwen3Config,
    Qwen3Model,
)

from dspark.offline_capture.transformers import OfflineTransformersCapture


class TransformersCaptureTest(unittest.TestCase):
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
