"""CPU contracts for optional EAGLE4 arithmetic and large-vocabulary loss."""

import unittest

import torch

from draftfit.algorithms.eagle3.model import _teacher_cross_entropy
from draftfit.modeling.draft.eagle4_precision import (
    Eagle4NativeAttention,
    Eagle4NativeMLP,
)
from draftfit.modeling.draft.llama3_eagle import LlamaAttention, LlamaMLP
from tests.test_modeling.test_eagle4 import tiny_config


class Eagle4PrecisionTest(unittest.TestCase):
    def test_grouped_qkv_preserves_outputs_and_all_projection_gradients(self):
        torch.manual_seed(31)
        reference = LlamaAttention(tiny_config())
        grouped = Eagle4NativeAttention(tiny_config())
        grouped.load_state_dict(reference.state_dict())
        value = torch.randn(2, 3, 32, requires_grad=True)
        other = value.detach().clone().requires_grad_()
        actual = grouped.project_qkv(value)
        expected = reference.project_qkv(other)
        for result, target in zip(actual, expected):
            torch.testing.assert_close(result, target)
        sum(result.square().sum() for result in actual).backward()
        sum(result.square().sum() for result in expected).backward()
        torch.testing.assert_close(value.grad, other.grad)
        for name in ("q_proj", "k_proj", "v_proj"):
            torch.testing.assert_close(
                getattr(grouped, name).weight.grad,
                getattr(reference, name).weight.grad,
            )

    def test_grouped_mlp_preserves_outputs_and_gradients(self):
        torch.manual_seed(37)
        reference = LlamaMLP(tiny_config())
        grouped = Eagle4NativeMLP(tiny_config())
        grouped.load_state_dict(reference.state_dict())
        value = torch.randn(2, 3, 16, requires_grad=True)
        other = value.detach().clone().requires_grad_()
        actual, expected = grouped(value), reference(other)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(value.grad, other.grad)
        for name, parameter in grouped.named_parameters():
            torch.testing.assert_close(
                parameter.grad, dict(reference.named_parameters())[name].grad
            )

    def test_large_bfloat16_vocabularies_match_float32_loss_and_gradients(self):
        torch.manual_seed(41)
        for size in (131072, 131073, 248320):
            for supervised in (True, False):
                with self.subTest(vocabulary=size, supervised=supervised):
                    logits = (torch.randn(1, 2, size) * 20).bfloat16().requires_grad_()
                    teacher = torch.randn(1, 2, size).softmax(-1)
                    mask = torch.tensor([[[float(supervised)], [0.0]]])
                    reference = logits.detach().float().requires_grad_()
                    expected = (
                        -(teacher * reference.log_softmax(-1) * mask).sum(-1).mean()
                    )
                    actual = _teacher_cross_entropy(logits, teacher, mask)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual.backward()
                    expected.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())
                    torch.testing.assert_close(
                        logits.grad, reference.grad.bfloat16(), rtol=0, atol=0
                    )
                    self.assertFalse(logits.grad[:, 1].any())


if __name__ == "__main__":
    unittest.main()
