"""Published speculators attention parity with an independent scalar reference.

Reference: vllm-project/speculators commit
47eee66a19cb5f967b215575d866934250522ad1,
src/speculators/models/dflash/attention.py:create_anchor_block_mask_mod.
Speculators v0.8.0 uses the same rule for DSpark. Each batch item is one document.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from draftfit.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDSparkModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)


def _published_reference(anchors, keep, sequence_length, block_size, window, causal):
    batches, blocks = anchors.shape
    result = torch.zeros(
        batches,
        1,
        blocks * block_size,
        sequence_length + blocks * block_size,
        dtype=torch.bool,
    )
    for batch in range(batches):
        for block in range(blocks):
            if not keep[batch, block]:
                continue
            anchor = int(anchors[batch, block])
            for offset in range(block_size):
                for key in range(result.shape[-1]):
                    prefix = key < sequence_length and key < anchor
                    if window is not None:
                        prefix = prefix and key >= anchor - window
                    own_block = key >= sequence_length and (
                        (key - sequence_length) // block_size == block
                    )
                    if window is not None and causal:
                        own_block = (
                            own_block and (key - sequence_length) % block_size <= offset
                        )
                    result[batch, 0, block * block_size + offset, key] = (
                        prefix or own_block
                    )
    return result


class _RecordingDraft(nn.Module):
    def __init__(self, non_causal):
        super().__init__()
        self.config = SimpleNamespace(
            speculators_attention_semantics="anchor_relative",
            dflash_config={"sliding_window_non_causal": non_causal},
        )
        self.sliding_window = 4

    def forward(self, noise_embedding, attention_mask, **kwargs):
        self.masks = attention_mask
        return noise_embedding


class PublishedSpeculatorsMaskTest(unittest.TestCase):
    def test_sdpa_and_flex_match_published_element_rule(self):
        anchors = torch.tensor([[0, 6, 11], [3, 7, 10]])
        keep = torch.tensor([[True, True, False], [True, False, True]])
        for window, causal in ((None, False), (1, True), (4, True), (4, False)):
            with self.subTest(window=window, causal=causal):
                args = dict(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=16,
                    block_size=8,
                    device=torch.device("cpu"),
                    sliding_window=window,
                    is_causal=causal,
                    context_window_origin="anchor",
                )
                expected = _published_reference(anchors, keep, 16, 8, window, causal)
                dense = create_dflash_sdpa_mask(**args)
                block = create_dflash_block_mask(**args)
                self.assertTrue(torch.equal(dense, expected))
                query = torch.arange(24).unsqueeze(1)
                key = torch.arange(40).unsqueeze(0)
                for batch in range(2):
                    self.assertTrue(
                        torch.equal(
                            block.mask_mod(batch, 0, query, key), expected[batch, 0]
                        )
                    )
        # The inclusive anchor-window boundary differs from the old default.
        args.update(sliding_window=4, is_causal=True)
        published = create_dflash_sdpa_mask(**args)
        legacy = create_dflash_sdpa_mask(**{**args, "context_window_origin": "query"})
        self.assertTrue(published[0, 0, 8, 2])
        self.assertFalse(legacy[0, 0, 8, 2])

    def test_wrapper_passes_published_options_for_both_families(self):
        anchors = torch.tensor([[6]])
        keep = torch.tensor([[True]])
        for wrapper in (OnlineDFlashModel, OnlineDSparkModel):
            for non_causal in (False, True):
                with self.subTest(wrapper=wrapper.__name__, non_causal=non_causal):
                    draft = _RecordingDraft(non_causal)
                    model = wrapper(
                        draft_model=draft,
                        target_lm_head=nn.Identity(),
                        target_embed_tokens=nn.Embedding(32, 8),
                        mask_token_id=31,
                        block_size=4,
                        attention_backend="sdpa",
                    )
                    with mock.patch.object(
                        model, "_sample_anchor_positions", return_value=(anchors, keep)
                    ):
                        model._forward_draft_blocks(
                            input_ids=torch.arange(16).unsqueeze(0),
                            hidden_states=torch.randn(1, 16, 8),
                            loss_mask=torch.ones(1, 16),
                        )
                    for kind, window in (
                        ("full_attention", None),
                        ("sliding_attention", 4),
                    ):
                        expected = _published_reference(
                            anchors, keep, 16, 4, window, not non_causal
                        )
                        self.assertTrue(torch.equal(draft.masks[kind], expected))

    def test_unknown_window_origin_fails(self):
        for builder in (create_dflash_sdpa_mask, create_dflash_block_mask):
            with self.assertRaisesRegex(ValueError, "context_window_origin"):
                builder(
                    torch.tensor([[6]]),
                    torch.tensor([[True]]),
                    16,
                    4,
                    torch.device("cpu"),
                    context_window_origin="unknown",
                )
