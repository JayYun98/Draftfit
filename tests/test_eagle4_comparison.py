"""Real tiny target checks for the comparison's capture and recurrent layout."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.evaluate_eagle4_comparison import eagle4_propose
from benchmarks.train_eagle4_comparison import (
    capture_example,
    parameter_hashes,
    warm_start,
    write_json,
)
from draftfit.algorithms.eagle4.model import Eagle4TrainingModel
from draftfit.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.utils import padding
from tests.test_modeling.test_eagle4 import tiny_config


class ComparisonCaptureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.target = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=3,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=64,
                    eos_token_id=None,
                    pad_token_id=0,
                )
            )
            .eval()
            .requires_grad_(False)
        )
        self.capture = OfflineTransformersCapture(self.target)
        self.capture.set_capture_layers([0, 1, 2], capture_method="dspark")

    def test_training_continuation_matches_target_generation(self):
        prompt = [1, 2, 3, 4]
        expected = self.target.generate(
            torch.tensor([prompt]), do_sample=False, max_new_tokens=4
        )
        # Raw assistant text can have a different reasoning mode or vocabulary.
        reference = prompt + [(int(t) + 1) % 32 for t in expected[0, 4:]]
        record = capture_example(self.target, self.capture, prompt, reference, 8)
        self.assertTrue(torch.equal(record["input_ids"], expected))
        self.assertTrue(torch.equal(record["prompt_ids"], torch.tensor([prompt])))
        self.assertEqual(record["loss_mask"].tolist(), [[0, 0, 0, 0, 1, 1, 1, 0]])
        captured = self.capture.capture(
            input_ids=expected,
            attention_mask=torch.ones_like(expected),
            loss_mask=record["loss_mask"],
        )
        torch.testing.assert_close(record["hidden_states"], captured.hidden_states)

    def test_reference_control_keeps_raw_assistant_tokens(self):
        prompt, ids = [1, 2], [1, 2, 8, 9, 10]
        record = capture_example(
            self.target, self.capture, prompt, ids, 5, source="reference"
        )
        self.assertEqual(record["input_ids"].tolist(), [ids])
        with self.assertRaisesRegex(ValueError, "preserve"):
            capture_example(
                self.target, self.capture, prompt, [3, 2, 8], 5, source="reference"
            )

    def test_extension_starts_from_dense_export_with_trainable_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            arm = source / "eagle4"
            draft = LlamaForCausalLMEagle4(tiny_config())
            draft.save_pretrained(arm / "dense-export")
            expected = {k: v.clone() for k, v in draft.state_dict().items()}
            write_json(arm / "dense-parameter-hashes.json", parameter_hashes(draft))
            write_json(arm / "result.json", {"dense_steps": 5000})
            with torch.no_grad():
                draft.efficient_lm_head.weight.add_(1)
            draft.save_pretrained(arm / "export")
            fresh = LlamaForCausalLMEagle4(tiny_config())
            receipt = warm_start(fresh, source, "eagle4")
            self.assertEqual(receipt["previous_dense_steps"], 5000)
            for key, value in fresh.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            self.assertTrue(fresh.midlayer.self_attn.q_proj.weight.requires_grad)
            self.assertFalse(fresh.efficient_lm_head.weight.requires_grad)

    @torch.inference_mode()
    def test_recurrent_training_logits_match_generation_at_each_depth(self):
        draft = LlamaForCausalLMEagle4(
            tiny_config(
                efficient_lm_head_config={
                    "type": "low_rank",
                    "rank": 8,
                    "num_candidates": 32,
                }
            )
        ).eval()
        prefix = torch.tensor([[1, 2, 3, 4, 5]])
        features = self.capture.capture(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            loss_mask=torch.ones_like(prefix),
        )
        anchor = self.target.lm_head(features.last_hidden_states[:, -1]).argmax(-1)
        # The comparison stores five taps and selects three for EAGLE4.
        h0, h1, h2 = features.hidden_states.chunk(3, dim=-1)
        features = replace(features, hidden_states=torch.cat((h0, h1, h1, h1, h2), -1))
        proposal_logits = []
        original = draft.candidate_logits

        def collect(hidden):
            proposal_logits.append(draft.compute_logits(hidden).clone())
            return original(hidden)

        with patch.object(draft, "candidate_logits", side_effect=collect):
            proposals = eagle4_propose(
                SimpleNamespace(draft_model=draft), features, anchor, 3
            )
        ids = torch.cat((prefix, anchor[:, None], proposals), dim=1)
        full = self.capture.capture(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            loss_mask=torch.ones_like(ids),
        )
        logits = []
        hook = draft.lm_head.register_forward_hook(
            lambda module, inputs, output: logits.append(output.clone())
        )
        try:
            Eagle4TrainingModel(draft, length=3)(
                input_ids=padding(ids, left=False),
                attention_mask=torch.ones_like(ids),
                target=padding(
                    self.target.lm_head(full.last_hidden_states), left=False
                ),
                loss_mask=torch.ones_like(ids)[..., None],
                hidden_states=full.hidden_states,
            )
        finally:
            hook.remove()
        self.assertEqual(len(logits), 3)
        for depth in range(3):
            torch.testing.assert_close(
                logits[depth][:, prefix.shape[1] - 1],
                proposal_logits[depth][:, 0],
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
