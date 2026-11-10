"""Prefix-dependent numerical drift must not change Spark greedy verification."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from draftfit.training.joint_verification import _generate, _settings


class PrefixStableVerificationTest(unittest.TestCase):
    def compare(self, *, reject=False, stops=()):
        embedding = torch.nn.Embedding(17, 1)
        teacher = SimpleNamespace(
            _model=SimpleNamespace(
                config=SimpleNamespace(model_type="spark2_5"),
                get_input_embeddings=lambda: embedding,
            )
        )
        bundle = SimpleNamespace(model=torch.nn.Linear(1, 1))
        config = _settings(
            dict(max_new_tokens=9, draft_tokens=3, stop_token_ids=list(stops))
        )

        def capture(teacher, ids, config):
            return SimpleNamespace(input_ids=ids.clone())

        def logits(teacher, bundle, captured, positions):
            ids = captured.input_ids
            indices = torch.arange(ids.shape[1])[positions]
            # Model the observed boundary: the same prefix position can round
            # differently when a longer block is evaluated. Its last-position
            # prediction stays deterministic, matching standalone generation.
            drift = (indices < ids.shape[1] - 1).long()[None]
            predicted = (ids[:, positions] + 1 + drift) % 17
            return torch.nn.functional.one_hot(predicted, 17).float()

        def propose(bundle, features, anchor, count):
            offsets = torch.arange(1, count + 1)[None]
            if reject:
                offsets = torch.full_like(offsets, 7)
            return (anchor[:, None] + offsets) % 17

        with (
            mock.patch(
                "draftfit.training.joint_verification._capture", side_effect=capture
            ),
            mock.patch(
                "draftfit.training.joint_verification._target_logits",
                side_effect=logits,
            ),
        ):
            baseline = _generate(
                teacher, bundle, torch.tensor([4]), config, speculative=False
            )
            run = _generate(
                teacher,
                bundle,
                torch.tensor([4]),
                config,
                speculative=True,
                proposal_fn=propose,
            )
        self.assertEqual(run["token_ids"], baseline["token_ids"])
        self.assertEqual(run["output_token_ids"], baseline["output_token_ids"])
        self.assertEqual(baseline["target_verification"], "standalone_prefix")
        self.assertEqual(
            run["target_verification"], "sequential_prefix_until_rejection"
        )
        return run

    def test_all_accepted_and_bonus_preserve_standalone_tokens(self):
        run = self.compare()
        self.assertGreater(run["proposed_tokens"], 0)
        self.assertEqual(run["accepted_tokens"], run["proposed_tokens"])
        self.assertEqual(run["generated_tokens"], 9)

    def test_rejection_fallback_preserves_standalone_tokens(self):
        run = self.compare(reject=True)
        self.assertEqual(run["accepted_tokens"], 0)
        self.assertGreater(run["reject_events"], 0)

    def test_stop_inside_accepted_suffix_preserves_standalone_tokens(self):
        run = self.compare(stops=[7])
        self.assertEqual(run["token_ids"], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
