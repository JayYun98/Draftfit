"""Synthetic numerical checks, not a calibrated deployment claim."""

import math
import unittest

import numpy as np

from draftfit.benchmarks.redline import _holm, select


class RedlineTest(unittest.TestCase):
    def test_exact_binomial_cdf_matches_direct_finite_sum(self):
        n, alpha = 10, 0.25
        correct = np.ones((n + 1, n), dtype=int)
        for k in range(n + 1):
            correct[k, :k] = 0
        # Put an all-correct reference in the last row to test a nonzero index.
        correct = np.vstack([correct, np.ones(n, dtype=int)])
        result = select(correct, np.ones(n + 2), reference=n + 1, alpha=alpha)
        expected = [
            math.fsum(
                math.comb(n, i) * alpha**i * (1 - alpha) ** (n - i)
                for i in range(k + 1)
            )
            for k in range(n + 1)
        ]
        np.testing.assert_allclose(result.pvalues[:-1], expected, rtol=1e-13, atol=0)
        self.assertAlmostEqual(result.pvalues[-1], (1 - alpha) ** n)
        self.assertEqual(result.pvalues[n], 1)

    def test_holm_stepdown_stops_after_first_failed_hypothesis(self):
        p = np.asarray([0.061, 0.01, 0.06])
        np.testing.assert_array_equal(_holm(np.log(p), 0.1), [False, True, False])
        np.testing.assert_array_equal(
            _holm(np.log([0.02, 0.03, 0.04]), 0.1), [True, True, True]
        )

    def test_selects_best_valid_cost_and_rejects_faster_risky_config(self):
        correct = np.ones((3, 100), dtype=int)
        correct[1, :1] = 0
        correct[2, :30] = 0
        result = select(correct, [1, 2, 5], ["reference", "safe", "risky"])
        self.assertEqual(result.deployed_name, "safe")
        self.assertEqual(result.reference, 0)
        self.assertEqual(result.n, 100)
        self.assertEqual(result.alpha, 0.1)
        self.assertEqual(result.delta, 0.1)
        np.testing.assert_array_equal(result.valid, [True, True, False])
        np.testing.assert_allclose(result.risk, [0, 0.01, 0.3])
        np.testing.assert_array_equal(result.cost, [1, 2, 5])
        lower = select(correct, [8, 4, 2], names=result.names, lower_is_better=True)
        self.assertEqual(lower.deployed_name, "safe")

    def test_fallback_does_not_certify_the_reference(self):
        result = select([[1], [1]], [1, 5], reference=0)
        self.assertEqual(result.names, ["0", "1"])
        self.assertEqual(result.deployed, 0)
        self.assertFalse(result.valid.any())
        self.assertEqual(result.deployed_name, "0")

    def test_risk_counts_only_harm_and_divides_by_all_paired_prompts(self):
        result = select([[1, 0, 1, 0], [0, 1, 1, 1]], [1, 2])
        np.testing.assert_array_equal(result.violations, [0, 1])
        np.testing.assert_allclose(result.risk, [0, 0.25])
        np.testing.assert_allclose(result.accuracy, [0.5, 0.75])

    def test_cost_ties_use_input_order_and_nonzero_reference(self):
        for lower in (False, True):
            result = select(
                np.ones((3, 100)),
                [2, 2, 2],
                reference=np.int64(2),
                lower_is_better=lower,
            )
            self.assertEqual(result.deployed, 0)
            self.assertEqual(result.reference, 2)

    def test_underflow_cannot_turn_insufficient_evidence_into_validity(self):
        # Exact p=2^-1077 is above delta/16=2^-1078, although both round to 0.
        result = select(
            np.ones((16, 1077)),
            np.arange(1, 17),
            alpha=0.5,
            delta=np.nextafter(0.0, 1.0),
        )
        self.assertFalse(result.valid.any())
        self.assertEqual(result.deployed, 0)
        self.assertTrue((result.pvalues == 0).all())

    def test_rejects_malformed_outcomes_costs_and_options(self):
        for correct in (
            [],
            [[1, 0], [1]],
            [1, 0],
            [[]],
            [[0.5]],
            [[float("nan")]],
            [[2]],
        ):
            with self.subTest(correct=correct), self.assertRaises(ValueError):
                select(correct, [1])
        for cost in (
            [],
            [1],
            [1, 0],
            [1, -1],
            [1, float("nan")],
            [1, float("inf")],
            [[1, 2]],
        ):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                select([[1], [1]], cost)
        for option, values in (
            ("alpha", (0, 1, float("nan"), float("inf"), "0.1", True)),
            ("delta", (0, 1, float("nan"), float("inf"), None, True)),
            ("reference", (-1, 2, 0.5, True, "0")),
            ("lower_is_better", ("false", 0, None)),
            ("names", (["same", "same"], ["one"], ["", "two"], "ab")),
        ):
            for value in values:
                with (
                    self.subTest(option=option, value=value),
                    self.assertRaises(ValueError),
                ):
                    select([[1], [1]], [1, 2], **{option: value})


if __name__ == "__main__":
    unittest.main()
