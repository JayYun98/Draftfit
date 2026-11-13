"""Independent values, gradients, and public boundaries for REAL."""

import math
import unittest

import torch

from draftfit.research.real import REALResult, real_loss


def two_step_inputs(phi, theta):
    q1 = torch.stack((phi, 1 - phi))
    q2 = torch.stack((theta, 1 - theta))
    draft = torch.stack([torch.stack((q1, q2)) for _ in range(4)]).log()
    target = torch.tensor(
        [[[0.4, 0.6], [0.9, 0.1]]] * 2 + [[[0.4, 0.6], [0.1, 0.9]]] * 2,
        dtype=torch.float64,
    ).log()
    ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    return draft, target, ids


def exact_and_wtv(phi, theta):
    """Enumerate draft paths and verifier p/q acceptance, independently of REAL."""
    p1, p2 = (0.4, 0.6), ((0.9, 0.1), (0.1, 0.9))
    q1, q2 = (phi, 1 - phi), (theta, 1 - theta)
    exact = phi * 0
    for first in range(2):
        accept_first = (p1[first] / q1[first]).clamp(max=1)
        exact = exact + q1[first] * accept_first
        for second in range(2):
            accept_second = (p2[first][second] / q2[second]).clamp(max=1)
            exact = exact + q1[first] * q2[second] * accept_first * accept_second
    overlap_first = sum(torch.minimum(q1[i], phi.new_tensor(p1[i])) for i in range(2))
    wtv = sum(
        p1[first]
        * (
            overlap_first
            + overlap_first
            * sum(
                torch.minimum(q2[i], theta.new_tensor(p2[first][i])) for i in range(2)
            )
        )
        for first in range(2)
    )
    return exact, wtv


class RealLossTests(unittest.TestCase):
    def test_two_step_exact_real_wtv_values_and_gradients(self):
        for theta_value in (0.3, 0.5, 0.65):
            with self.subTest(theta=theta_value):
                phi = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
                theta = torch.tensor(
                    theta_value, dtype=torch.float64, requires_grad=True
                )
                draft, target, ids = two_step_inputs(phi, theta)
                result = real_loss(draft, target, ids, real_weight=1)
                probabilities = torch.tensor(
                    [0.36, 0.04, 0.06, 0.54], dtype=torch.float64
                )
                branch = result.acceptance_estimate @ probabilities
                exact, wtv = exact_and_wtv(phi, theta)
                branch_grad = torch.stack(
                    torch.autograd.grad(branch, (phi, theta), retain_graph=True)
                )
                exact_grad = torch.stack(
                    torch.autograd.grad(exact, (phi, theta), retain_graph=True)
                )
                wtv_grad = torch.stack(torch.autograd.grad(wtv, (phi, theta)))
                self.assertAlmostEqual(
                    exact.item(), 0.86 + 0.2 * theta_value, places=12
                )
                torch.testing.assert_close(branch, exact, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(
                    branch_grad, exact_grad, rtol=1e-12, atol=1e-12
                )
                torch.testing.assert_close(
                    exact_grad,
                    torch.tensor([-(2.1 - theta_value), 0.2], dtype=torch.float64),
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertAlmostEqual(wtv.item(), 1.02 - 0.12 * theta_value, places=12)
                self.assertAlmostEqual(wtv_grad[1].item(), -0.12, places=12)
                self.assertGreater(branch_grad[1].item(), 0)
                if theta_value == 0.5:
                    self.assertAlmostEqual(branch.item(), 0.96, places=12)
                    self.assertAlmostEqual(wtv.item(), branch.item(), places=12)

    def test_first_step_gradient_includes_prefix_survival(self):
        phi = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        theta = phi.new_tensor(0.5)
        draft, target, ids = two_step_inputs(phi, theta)
        result = real_loss(draft[2:3], target[2:3], ids[2:3], real_weight=1)
        result.loss.backward()
        self.assertAlmostEqual(result.loss.item(), 0.6, places=12)
        self.assertAlmostEqual(phi.grad.item(), 1.0, places=12)
        self.assertNotAlmostEqual(phi.grad.item(), 0.5, places=8)

    def test_50_block_replication_has_the_expected_logit_gradient(self):
        logit = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        draft, target, ids = two_step_inputs(logit.new_tensor(0.8), logit.sigmoid())
        paths = torch.arange(4).repeat_interleave(torch.tensor([18, 2, 3, 27]))
        result = real_loss(draft[paths], target[paths], ids[paths], real_weight=1)
        result.loss.backward()
        self.assertAlmostEqual(result.real_loss.item(), 0.52, places=12)
        self.assertAlmostEqual(logit.grad.item(), -0.025, places=12)

    def test_target_sampled_ratio_is_q_over_p(self):
        draft, target, ids = two_step_inputs(torch.tensor(0.8), torch.tensor(0.5))
        result = real_loss(draft.double(), target, ids)
        self.assertAlmostEqual(result.path_weights[0, 0].item(), 1, places=7)
        self.assertAlmostEqual(result.path_weights[2, 0].item(), 1 / 3, places=7)
        self.assertAlmostEqual(result.prefix_survival[2, 1].item(), 1 / 3, places=7)
        self.assertAlmostEqual(result.acceptance_estimate[2].item(), 0.8, places=7)

    def test_structured_result_ce_argmax_and_mixture_endpoints(self):
        draft = torch.tensor(
            [[[0.7, 0.2, 0.1], [0.4, 0.1, 0.5]]], dtype=torch.float64
        ).log()
        target = torch.tensor(
            [[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]], dtype=torch.float64
        ).log()
        ids = torch.tensor([[0, 2]])
        expected_ce = -(math.log(0.2) + math.log(0.4)) / 2
        soft_ce = (
            -(
                0.1 * math.log(0.7)
                + 0.7 * math.log(0.2)
                + 0.2 * math.log(0.1)
                + 0.6 * math.log(0.4)
                + 0.3 * math.log(0.1)
                + 0.1 * math.log(0.5)
            )
            / 2
        )
        sampled_ce = -(math.log(0.7) + math.log(0.5)) / 2
        for weight in (0, 0.9, 1):
            with self.subTest(weight=weight):
                result = real_loss(draft, target, ids, real_weight=weight)
                self.assertIsInstance(result, REALResult)
                self.assertAlmostEqual(result.ce_loss.item(), expected_ce, places=12)
                self.assertNotAlmostEqual(result.ce_loss.item(), soft_ce, places=8)
                self.assertNotAlmostEqual(result.ce_loss.item(), sampled_ce, places=8)
                self.assertAlmostEqual(
                    result.loss.item(),
                    weight * result.real_loss.item() + (1 - weight) * expected_ce,
                    places=12,
                )
                torch.testing.assert_close(
                    result.branch_terms.sum(-1), result.acceptance_estimate
                )
                self.assertAlmostEqual(result.real_loss.item(), 0.5, places=12)
        other_ids = real_loss(draft, target, torch.tensor([[1, 0]]))
        self.assertAlmostEqual(other_ids.ce_loss.item(), expected_ce, places=12)
        self.assertNotAlmostEqual(
            other_ids.real_loss.item(), result.real_loss.item(), places=8
        )

    def test_target_detach_and_public_call_does_not_backward(self):
        draft, target, ids = two_step_inputs(torch.tensor(0.8), torch.tensor(0.5))
        draft, target = draft.double().requires_grad_(), target.requires_grad_()
        before_draft, before_target, before_ids = (
            draft.detach().clone(),
            target.detach().clone(),
            ids.clone(),
        )
        result = real_loss(draft, target, ids)
        self.assertIsNone(draft.grad)
        self.assertTrue(result.overlap.requires_grad)
        self.assertTrue(result.path_weights.requires_grad)
        self.assertTrue(result.prefix_survival.requires_grad)
        result.loss.backward()
        self.assertTrue(torch.isfinite(draft.grad).all())
        self.assertGreater(draft.grad.norm().item(), 0)
        self.assertIsNone(target.grad)
        torch.testing.assert_close(draft, before_draft)
        torch.testing.assert_close(target, before_target)
        self.assertTrue(torch.equal(ids, before_ids))

    def test_mask_gaps_fixed_gamma_empty_blocks_and_zero_suffix_gradient(self):
        draft = (
            torch.tensor([0.8, 0.2], dtype=torch.float64)
            .log()
            .repeat(3, 4, 1)
            .requires_grad_()
        )
        target = (
            torch.tensor([0.4, 0.6], dtype=torch.float64)
            .log()
            .repeat(3, 4, 1)
            .requires_grad_()
        )
        ids = torch.tensor([[0, -100, -100, 1], [0, 0, -100, 0], [-100, -100, 0, 1]])
        mask = torch.tensor(
            [
                [True, False, True, True],
                [True, True, False, True],
                [False, True, True, True],
            ]
        )
        original_mask = mask.clone()
        result = real_loss(draft, target, ids, prediction_mask=mask)
        active = torch.tensor(
            [[True, False, False, False], [True, True, False, False], [False] * 4]
        )
        self.assertTrue(torch.equal(result.active_mask, active))
        torch.testing.assert_close(
            result.acceptance_estimate, draft.new_tensor([0.6, 1.2, 0])
        )
        self.assertAlmostEqual(result.real_loss.item(), 0.775, places=12)
        self.assertAlmostEqual(result.ce_loss.item(), -math.log(0.2), places=12)
        for terms in (result.overlap, result.prefix_survival, result.branch_terms):
            self.assertTrue(
                torch.equal(terms[~active], torch.zeros_like(terms[~active]))
            )
        self.assertTrue(
            torch.equal(
                result.path_weights[~active],
                torch.ones_like(result.path_weights[~active]),
            )
        )
        result.loss.backward()
        self.assertTrue(
            torch.equal(draft.grad[~active], torch.zeros_like(draft.grad[~active]))
        )
        self.assertGreater(draft.grad[active].norm().item(), 0)
        self.assertIsNone(target.grad)
        altered = draft.detach().clone()
        altered[~active] = torch.tensor([12.0, -8.0], dtype=torch.float64)
        changed = real_loss(altered, target, ids, prediction_mask=mask)
        torch.testing.assert_close(changed.loss, result.loss)
        self.assertTrue(torch.equal(mask, original_mask))
        order = torch.tensor([1, 2, 0])
        permuted = real_loss(
            draft.detach()[order],
            target[order],
            ids[order],
            prediction_mask=mask[order],
        )
        torch.testing.assert_close(permuted.loss, result.loss)
        torch.testing.assert_close(
            permuted.acceptance_estimate, result.acceptance_estimate[order]
        )

    def test_fp64_and_mixed_input_precision(self):
        source, target, ids = two_step_inputs(
            torch.tensor(0.8, dtype=torch.float64),
            torch.tensor(0.5, dtype=torch.float64),
        )
        for draft_dtype, target_dtype, expected_dtype in (
            (torch.float64, torch.float64, torch.float64),
            (torch.float32, torch.float64, torch.float64),
            (torch.float64, torch.float16, torch.float64),
            (torch.float32, torch.float32, torch.float32),
            (torch.float16, torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16, torch.float32),
            (torch.float16, torch.bfloat16, torch.float32),
        ):
            with self.subTest(draft=draft_dtype, target=target_dtype):
                draft = source.to(draft_dtype).detach().requires_grad_()
                frozen = target.to(target_dtype).detach().requires_grad_()
                reference = real_loss(
                    draft.detach().double(), frozen.detach().double(), ids
                )
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    result = real_loss(draft, frozen, ids)
                for name in (
                    "loss",
                    "real_loss",
                    "ce_loss",
                    "acceptance_estimate",
                    "overlap",
                    "path_weights",
                    "prefix_survival",
                    "branch_terms",
                ):
                    self.assertEqual(getattr(result, name).dtype, expected_dtype)
                    torch.testing.assert_close(
                        getattr(result, name).double(),
                        getattr(reference, name),
                        rtol=2e-6,
                        atol=2e-7,
                    )
                result.loss.backward()
                self.assertEqual(draft.grad.dtype, draft_dtype)
                self.assertTrue(torch.isfinite(draft.grad).all())
                self.assertIsNone(frozen.grad)

    def test_noncontiguous_inputs_and_metadata(self):
        generator = torch.Generator().manual_seed(19)
        draft = (
            torch.randn(2, 3, 8, generator=generator, dtype=torch.float64)[:, :, ::2]
            .detach()
            .requires_grad_()
        )
        target = torch.randn(2, 3, 8, generator=generator, dtype=torch.float64)[
            :, :, ::2
        ]
        ids = torch.tensor([[0, 9, 1, 9, 2, 9], [1, 9, 2, 9, 3, 9]])[:, ::2]
        mask = torch.tensor([[True, False, True, False, False, False]] * 2)[:, ::2]
        self.assertFalse(draft.is_contiguous())
        result = real_loss(draft, target, ids, prediction_mask=mask)
        copied = real_loss(
            draft.detach().clone(),
            target.clone(),
            ids.clone(),
            prediction_mask=mask.clone(),
        )
        torch.testing.assert_close(result.loss, copied.loss)
        result.loss.backward()
        self.assertEqual(draft.grad.shape, draft.shape)
        self.assertTrue(
            torch.equal(draft.grad[:, 2], torch.zeros_like(draft.grad[:, 2]))
        )

    def test_log_ratio_handles_probability_underflow(self):
        draft = torch.tensor(
            [[[-1000.0, 0.0], [-1000.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([[[-900.0, 0.0], [-900.0, 0.0]]], dtype=torch.float64)
        result = real_loss(draft, target, torch.tensor([[0, 0]]))
        self.assertAlmostEqual(
            result.path_weights[0, 0].item() / math.exp(-100), 1, places=12
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(draft.grad).all())

    def test_finite_extreme_normalization_fails_explicitly(self):
        for dtype in (torch.float32, torch.float64):
            extreme = torch.tensor(
                [[[-torch.finfo(dtype).max, torch.finfo(dtype).max]]], dtype=dtype
            )
            ordinary = torch.zeros_like(extreme)
            self.assertTrue(torch.isfinite(extreme).all())
            for draft, target in ((extreme, ordinary), (ordinary, extreme)):
                with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                    real_loss(draft, target, torch.tensor([[0]]))

    def test_invalid_public_boundaries(self):
        logits = torch.zeros(1, 2, 3)
        ids = torch.zeros(1, 2, dtype=torch.long)
        for invalid in (
            None,
            [],
            torch.zeros(2, 3),
            torch.zeros(0, 2, 3),
            torch.zeros(1, 0, 3),
            torch.zeros(1, 2, 0),
            logits.long(),
            logits.to(torch.complex64),
            logits.to_sparse(),
            torch.empty(1, 2, 3, device="meta"),
        ):
            with self.subTest(logits=type(invalid)), self.assertRaises(ValueError):
                real_loss(invalid, logits, ids)
        with self.assertRaises(ValueError):
            real_loss(logits, torch.zeros(2, 2, 3), ids)
        for invalid in (
            ids.float(),
            ids.int(),
            ids.bool(),
            ids[:, :1],
            ids.tolist(),
            ids.to_sparse(),
            torch.empty_like(ids, device="meta"),
            torch.tensor([[-100, 0]]),
            torch.tensor([[-1, 0]]),
            torch.tensor([[3, 0]]),
        ):
            with self.subTest(ids=type(invalid)), self.assertRaises(ValueError):
                real_loss(logits, logits, invalid)
        for invalid in (
            torch.ones_like(ids),
            torch.ones(2, 2, dtype=torch.bool),
            [[True, True]],
            torch.ones(1, 2, dtype=torch.bool, device="meta"),
            torch.zeros_like(ids, dtype=torch.bool),
        ):
            with self.subTest(mask=type(invalid)), self.assertRaises(ValueError):
                real_loss(logits, logits, ids, prediction_mask=invalid)
        mask = torch.tensor([[True, False]])
        for invalid_id in (-1, 3, -101):
            with self.subTest(inactive=invalid_id), self.assertRaises(ValueError):
                real_loss(
                    logits,
                    logits,
                    torch.tensor([[0, invalid_id]]),
                    prediction_mask=mask,
                )
        for weight in (
            True,
            False,
            "0.9",
            None,
            [0.9],
            torch.tensor(0.9),
            1j,
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            -float("inf"),
            10**400,
        ):
            with self.subTest(weight=type(weight)), self.assertRaises(ValueError):
                real_loss(logits, logits, ids, real_weight=weight)
        for value in (float("nan"), float("inf"), -float("inf")):
            nonfinite = logits.clone()
            nonfinite[0, 1, 0] = value
            for draft, target in ((nonfinite, logits), (logits, nonfinite)):
                with self.subTest(nonfinite=value), self.assertRaises(ValueError):
                    real_loss(draft, target, ids, prediction_mask=mask)

    def test_gamma_one_and_single_vocabulary_are_valid(self):
        draft = torch.tensor([[[2.0]]], dtype=torch.float64, requires_grad=True)
        result = real_loss(draft, torch.tensor([[[9.0]]]), torch.tensor([[0]]))
        self.assertEqual(result.loss.item(), 0)
        self.assertEqual(result.acceptance_estimate.item(), 1)
        result.loss.backward()
        self.assertEqual(draft.grad.item(), 0)

    def test_one_small_cpu_sgd_update_lowers_fixed_objective(self):
        draft, target, ids = two_step_inputs(
            torch.tensor(0.8, dtype=torch.float64),
            torch.tensor(0.5, dtype=torch.float64),
        )
        parameter = torch.nn.Parameter(draft.clone())
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        before = parameter.detach().clone()
        loss = real_loss(parameter, target, ids).loss
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.norm().item(), 0)
        optimizer.step()
        after = real_loss(parameter, target, ids).loss
        self.assertFalse(torch.equal(before, parameter))
        self.assertLess(after.item(), loss.item())


if __name__ == "__main__":
    unittest.main()
