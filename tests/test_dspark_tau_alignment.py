"""Closed-form tau diagnostics through the real DSpark token-pooled objective."""

import unittest

import torch

from draftfit.algorithms.common.dflash_family_model import OnlineDSparkModel
from draftfit.modeling.auto import AutoDraftModel
from draftfit.training.model_loading import _draft_config_from_dict


class DSparkTauAlignmentTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def objective(self, sample_from_anchor, chunk_size, *, padded=False):
        config = _draft_config_from_dict(
            {
                "architectures": ["DSparkDraftModel"],
                "model_type": "qwen3",
                "vocab_size": 2,
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "layer_types": ["full_attention"],
                "block_size": 4,
                "num_target_layers": 2,
                "dflash_config": {
                    "target_layer_ids": [0],
                    "mask_token_id": 1,
                    "projector_type": "dspark",
                    "attention_mode": "mha",
                    "markov_rank": 0,
                    "enable_confidence_head": False,
                    "confidence_head_with_markov": False,
                    "sample_from_anchor": sample_from_anchor,
                },
            }
        )
        model = OnlineDSparkModel(
            draft_model=AutoDraftModel.from_config(config),
            target_lm_head=torch.nn.Identity(),
            target_embed_tokens=torch.nn.Embedding(2, 2),
            mask_token_id=1,
            block_size=4,
            attention_backend="sdpa",
            sample_from_anchor=sample_from_anchor,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=0.0,
            objective_chunk_blocks=chunk_size,
        )
        input_ids = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
        anchors = torch.tensor([[1, 6, 1]] if padded else [[1]])
        keep = torch.tensor([[True, True, False]] if padded else [[True]])
        targets, mask, indices = model._build_dspark_labels_and_mask(
            input_ids, torch.ones_like(input_ids), anchors, keep
        )
        # Teacher p=(3/4,1/4); these q rows have TV overlap probabilities
        # (3/8, 1/2, 3/4, 1/2). No probability or objective is mocked.
        hidden = (
            torch.tensor([[0.125, 0.875], [0.25, 0.75], [0.5, 0.5], [0.25, 0.75]])
            .log()
            .repeat(anchors.size(1), 1)
            .unsqueeze(0)
            .requires_grad_()
        )
        teacher_hidden = torch.tensor([0.75, 0.25]).log().view(1, 1, 2).repeat(1, 8, 1)
        loss, metrics = model._compute_dspark_loss(
            output_hidden=hidden,
            target_ids=targets,
            eval_mask=mask,
            prev_token_ids=torch.zeros_like(targets),
            safe_label_indices=indices,
            target_last_hidden_states=teacher_hidden,
        )
        return loss, metrics, hidden, targets, mask

    def test_committed_anchor_is_excluded_from_tau_prefix(self):
        for chunk_size in (0, 1):
            with self.subTest(chunk_size=chunk_size):
                _, metrics, _, _, mask = self.objective(False, chunk_size)
                self.assertEqual(mask.tolist(), [[[False, True, True, True]]])
                numerator, denominator = metrics["ratio_metrics"]["tau_probabilistic"]
                # 1 + 1/2 + 3/8 + 3/16 = 33/16.
                self.assertAlmostEqual(numerator.item(), 2.0625, places=6)
                self.assertEqual(denominator.item(), 1.0)
                self.assertFalse(numerator.requires_grad)

    def test_anchor_relative_tau_keeps_existing_all_slot_definition(self):
        for chunk_size in (0, 1):
            with self.subTest(chunk_size=chunk_size):
                _, metrics, _, _, mask = self.objective(True, chunk_size)
                self.assertTrue(mask.all())
                numerator, denominator = metrics["ratio_metrics"]["tau_probabilistic"]
                # 1 + 3/8 + 3/16 + 9/64 + 9/128 = 227/128.
                self.assertAlmostEqual(numerator.item(), 1.7734375, places=6)
                self.assertEqual(denominator.item(), 1.0)

    def test_invalid_block_is_excluded_and_truncated_suffix_counts_once(self):
        for alignment, expected in ((False, 3.5625), (True, 3.1484375)):
            for chunk_size in (0, 1):
                with self.subTest(alignment=alignment, chunk_size=chunk_size):
                    _, metrics, _, _, mask = self.objective(
                        alignment, chunk_size, padded=True
                    )
                    self.assertFalse(mask[:, 2].any())
                    self.assertEqual(mask[:, 1].sum().item(), 1)
                    numerator, denominator = metrics["ratio_metrics"][
                        "tau_probabilistic"
                    ]
                    # The short block adds 1+1/2 (aligned) or 1+3/8 (relative).
                    # The invalid block adds neither numerator nor denominator.
                    self.assertAlmostEqual(numerator.item(), expected, places=6)
                    self.assertEqual(denominator.item(), 2.0)

    def test_loss_and_gradient_still_match_token_pooled_ce_and_tv(self):
        for alignment in (False, True):
            with self.subTest(alignment=alignment):
                loss, _, hidden, targets, mask = self.objective(
                    alignment, 1, padded=True
                )
                logits = hidden.reshape(1, 3, 4, 2)
                ce = (
                    -logits.log_softmax(-1)
                    .gather(-1, targets.unsqueeze(-1))
                    .squeeze(-1)
                )
                tv = (logits.softmax(-1) - torch.tensor([0.75, 0.25])).abs().sum(-1)
                expected = ((0.1 * ce + 0.9 * tv) * mask).sum() / mask.sum()
                torch.testing.assert_close(loss, expected)
                actual_gradient = torch.autograd.grad(loss, hidden, retain_graph=True)[
                    0
                ]
                expected_gradient = torch.autograd.grad(expected, hidden)[0]
                torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
