import math
import unittest

from draftfit.research.papers.dice.method import DiceRankingConfig


def recipe(**updates):
    return DiceRankingConfig.model_validate(
        {
            "method": "dice_ranking",
            "output_dir": "unused",
            "draft_logits": [2, 1, 0],
            "position_gumbels": [-1, 2, 0],
            "alpha": 1,
            "sigma": 1,
            "temperature": 1,
            "top_k": 2,
            **updates,
        }
    )


class DiceRankingTest(unittest.TestCase):
    def test_draw_changes_the_candidate(self):
        self.assertEqual(recipe().execute()["candidate_ids"], [1, 0])
        self.assertEqual(recipe(temperature=0).execute()["candidate_ids"], [0, 1])

    def test_full_vocabulary_normalization(self):
        result = recipe(alpha=2, temperature=0.5, sigma=2).execute()
        # Direct unshifted softmax of the three hand-computed utilities.
        weights = [math.exp(1.75), math.exp(1.5), 1]
        for probability, weight in zip(result["token_probabilities"], weights):
            self.assertAlmostEqual(probability, weight / sum(weights))

    def test_low_probability_ranks_and_exact_ties(self):
        result = recipe(
            draft_logits=[0, -900, -800, -800],
            position_gumbels=[0] * 4,
            top_k=4,
        ).execute()
        self.assertEqual(result["token_probabilities"], [1, 0, 0, 0])
        self.assertEqual(result["candidate_ids"], [0, 2, 3, 1])

    def test_wide_range_with_large_scale(self):
        result = recipe(
            draft_logits=[1e308, -1e308], position_gumbels=[0, 0], sigma=1e308
        ).execute()
        self.assertAlmostEqual(result["token_probabilities"][1], 1 / (1 + math.exp(2)))

    def test_common_offsets_preserve_noise_and_logit_differences(self):
        for logits, draws in (([1e16] * 2, [0, 1]), ([0, 1], [1e16] * 2)):
            with self.subTest(logits=logits):
                result = recipe(draft_logits=logits, position_gumbels=draws).execute()
                self.assertEqual(result["candidate_ids"], [1, 0])
                self.assertAlmostEqual(
                    result["token_probabilities"][0], 1 / (1 + math.e)
                )

    def test_invalid_inputs(self):
        for updates in (
            {"position_gumbels": [0]},
            {"sigma": 0},
            {"temperature": -1},
            {"draft_logits": [0, math.nan, 1]},
            {"top_k": 4},
            {"alpha": math.inf},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                recipe(**updates).execute()
