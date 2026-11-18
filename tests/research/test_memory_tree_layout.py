"""Compatibility and observation boundaries for memory and tree methods."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from draftfit.research import memory
from draftfit.research.diagnostics import position_bins
from draftfit.research.papers.cast.decode import walk_tree
from draftfit.research.papers.cast.tree import TreeNode
from draftfit.research.papers.ngramrl.method import NgramMemory
from draftfit.research.papers.specmem.method import SpecMemMemory
from draftfit.research.papers.valuspec.method import ConditionalBranchFilter
from draftfit.research.proposals import Proposal
from draftfit.research.reference_decode import generate
from tests.test_research_methods import target_model


class MemoryTreeLayoutTest(unittest.TestCase):
    def test_legacy_memory_proposal_works_in_reference_verifier(self):
        self.assertIs(memory.NgramMemory, NgramMemory)
        self.assertIs(memory.SpecMemMemory, SpecMemMemory)
        model, ids = target_model(), torch.tensor([[1, 2, 3]])
        expected = int(model(ids).logits[0, -1].argmax())
        wrong = (expected + 1) % model.config.vocab_size
        proposal = memory.Proposal(
            (wrong, wrong), ({wrong: 0.75, expected: 0.25}, {wrong: 1.0})
        )
        _, stats = generate(
            model,
            ids,
            lambda _, remaining, rng: proposal if remaining == 2 else Proposal(),
            max_new_tokens=2,
            record_acceptance=True,
        )
        self.assertEqual(
            position_bins(stats["token_acceptance"], width=8),
            [
                {
                    "start_position": 0,
                    "attempted_proposals": 1,
                    "observed_acceptance": 0.0,
                    "mean_distribution_overlap": 0.25,
                    "argmax_agreement": 0.0,
                }
            ],
        )

    def test_legacy_patches_reach_memory_policy_globals_and_restore(self):
        normalize = memory.normalized_embedding
        with patch.object(memory, "normalized_embedding", wraps=normalize) as called:
            SpecMemMemory().add_shared_seed([1.0], [1])
            called.assert_called_once()
        self.assertIs(memory.normalized_embedding, normalize)

        snapshot = memory.NgramSnapshot({}, 2, 3, 0)
        with patch.object(memory, "Proposal", wraps=Proposal) as called:
            snapshot.propose([1], 1)
            called.assert_called_once()
        self.assertIs(memory.Proposal, Proposal)

    def test_patching_legacy_decoding_and_scoring_affects_execution(self):
        with patch("draftfit.research.cast.decode.pick", return_value=3):
            self.assertEqual(walk_tree([], torch.zeros(1, 4)), ([], 3))

        scorer = SimpleNamespace(device=torch.device("cpu"))
        nodes = [TreeNode(1, 1, -1, 0.0)]
        with patch("draftfit.research.valuspec.conditional_score", return_value=-2):
            kept, stats = ConditionalBranchFilter(scorer, -1)([0], nodes)
        self.assertEqual(kept, [])
        self.assertEqual(stats["scorer_forwards"], 1)
