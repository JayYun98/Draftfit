"""Token-position regression for both supported DSpark label alignments."""

import copy
import unittest
from types import SimpleNamespace

import torch

from draftfit.modeling.auto import AutoDraftModel
from draftfit.training.model_loading import _draft_config_from_dict
from tests.test_modeling.test_draft_registry import TINY_DSPARK


class DSparkSamplingAlignmentTest(unittest.TestCase):
    def test_generated_suffix_matches_supervised_token_positions(self):
        # Each hidden slot identifies one distinct next token. A shifted slice
        # therefore emits the wrong token even with a perfectly trained head.
        for alignment in (True, False, None):
            for use_markov in (True, False):
                for prefix_length in (0, 3):
                    with self.subTest(
                        sample_from_anchor=alignment,
                        markov=use_markov,
                        prefix_length=prefix_length,
                    ):
                        raw = copy.deepcopy(TINY_DSPARK)
                        if alignment is not None:
                            raw["dflash_config"]["sample_from_anchor"] = alignment
                        if not use_markov:
                            raw["dflash_config"].update(
                                markov_rank=0,
                                confidence_head_with_markov=False,
                            )
                        model = AutoDraftModel.from_config(_draft_config_from_dict(raw))
                        if model.markov_head is not None:
                            for parameter in model.markov_head.parameters():
                                parameter.data.zero_()
                        block_size = model.block_size
                        hidden = torch.zeros(
                            1, prefix_length + block_size, model.config.hidden_size
                        )
                        head = torch.nn.Linear(
                            model.config.hidden_size,
                            model.config.vocab_size,
                            bias=False,
                        )
                        head.weight.data.zero_()
                        for position in range(block_size):
                            hidden[0, prefix_length + position, position] = 10
                            head.weight.data[11 + position, position] = 1
                        anchor = torch.tensor([[10, 0, 0, 0]])
                        actual = model._sample_draft_tokens(
                            SimpleNamespace(lm_head=head), hidden, anchor
                        )
                        # Anchor-relative training: slots predict 11,12,13,14.
                        # DFlash training: slots predict 10,11,12,13.
                        expected = [12, 13, 14] if alignment is False else [11, 12, 13]
                        self.assertEqual(actual.tolist(), [expected])


if __name__ == "__main__":
    unittest.main()
