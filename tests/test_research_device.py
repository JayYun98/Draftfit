"""Tiny device gates; CPU by default, CUDA only when explicitly selected.

DRAFTFIT_RESEARCH_TEST_DEVICE=cuda enables the same three gates on the caller's
GPU. DRAFTFIT_RESEARCH_TEST_DTYPE=bfloat16 optionally exercises reduced precision.
No pretrained download, distributed setup, or serving engine is required.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from tests.test_research_methods import draft_model, target_model
from tests.test_research_providers import feature_batch


class ResearchDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("DRAFTFIT_RESEARCH_TEST_DEVICE", "cpu")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("DRAFTFIT_RESEARCH_TEST_DEVICE must be cpu or cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        cls.device = torch.device(requested)
        name = os.environ.get("DRAFTFIT_RESEARCH_TEST_DTYPE", "float32")
        if name not in {"float32", "bfloat16"}:
            raise ValueError("DRAFTFIT_RESEARCH_TEST_DTYPE must be float32 or bfloat16")
        cls.dtype = getattr(torch, name)
        if requested == "cuda":
            torch.cuda.reset_peak_memory_stats()

    @classmethod
    def tearDownClass(cls):
        if cls.device.type == "cuda":
            torch.cuda.synchronize()
            print(
                f"research peak CUDA allocated bytes: {torch.cuda.max_memory_allocated()}"
            )

    def test_alr_rollout_loader_and_update(self):
        from draftfit.algorithms.alr_ira.model import ALRTrainingModel
        from draftfit.algorithms.alr_ira.prepare import prepare
        from draftfit.research.config import ALRConfig

        target = target_model().to(device=self.device, dtype=self.dtype)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1]})
                + "\n"
            )
            config = ALRConfig.model_validate(
                {
                    "method": "alr_ira",
                    "model": {"target_model_path": "tiny"},
                    "data": {"input_path": str(corpus)},
                    "training": {
                        "block_size": 6,
                        "target_layer_ids": [0, 2],
                        "anchors_per_sample": 1,
                    },
                    "output_dir": str(root / "features"),
                }
            )
            receipt = prepare(config, target=target)
            self.assertEqual(receipt["forward_calls"], 6)
            self.assertEqual(receipt["secondary_blocks"], 1)
            draft = draft_model(6).to(device=self.device, dtype=self.dtype)
            wrapper = ALRTrainingModel(
                draft_model=draft,
                target_lm_head=target.lm_head,
                target_embed_tokens=target.get_input_embeddings(),
                mask_token_id=31,
                block_size=6,
                attention_backend="sdpa",
            )
            registration = builtin_algorithm_registry().resolve("alr_ira")
            strategy = registration.providers.step.build(wrapper)
            optimizer = torch.optim.SGD(draft.parameters(), lr=0.1)
            before = draft.fc.weight.detach().clone()
            output = strategy.forward_loss(
                feature_batch(registration, root / "features")
            )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertTrue(torch.isfinite(draft.fc.weight.grad).all())
            self.assertGreater(draft.fc.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, draft.fc.weight))

    def test_osprey_preservation_and_ttt_update(self):
        from draftfit.algorithms.eagle3.model import OnlineEagle3Model
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.research.osprey import convert

        source = target_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.save_pretrained(root / "source")
            convert(str(root / "source"), str(root / "draft"), 3, 16, [0, 1, 2])
            draft = AutoDraftModel.from_pretrained(root / "draft").to(
                device=self.device, dtype=self.dtype
            )
            source = source.to(device=self.device, dtype=self.dtype)
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=self.device)
            features = torch.randn(1, 6, 48, device=self.device, dtype=self.dtype)
            mask = draft.prepare_decoder_attention_mask(
                torch.ones_like(ids), features, 1, 6, 0
            )
            hidden = draft.backbone(
                input_embeds=draft.embed_input_ids(ids),
                hidden_states=draft.project_hidden_states(features),
                cache_hidden=None,
                attention_mask=mask,
                position_ids=torch.arange(6, device=self.device).unsqueeze(0),
                use_cache=False,
            )
            tolerance = 0.02 if self.dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(
                draft.compute_logits(hidden),
                source(ids).logits,
                atol=tolerance,
                rtol=tolerance,
            )
            draft.freeze_embedding()
            wrapper = OnlineEagle3Model(draft, length=2, attention_backend="sdpa")
            optimizer = torch.optim.SGD(
                [p for p in draft.parameters() if p.requires_grad], lr=0.1
            )
            before = draft.layers[0].self_attn.q_proj.weight.detach().clone()
            embedding = draft.embed_tokens.weight.detach().clone()
            losses, *_ = wrapper(
                ids,
                torch.ones_like(ids),
                source(ids).logits.detach(),
                torch.ones_like(ids, dtype=torch.float32).unsqueeze(-1),
                features,
            )
            loss = sum(losses)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradient = draft.layers[0].self_attn.q_proj.weight.grad[:, :16]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(
                torch.equal(before, draft.layers[0].self_attn.q_proj.weight)
            )
            torch.testing.assert_close(
                draft.embed_tokens.weight, embedding, atol=0, rtol=0
            )

    def test_cast_target_greedy_parity_and_eos(self):
        from draftfit.research.cast.decode import cast_generate

        target = target_model().to(device=self.device, dtype=self.dtype)
        draft = draft_model(4).to(device=self.device, dtype=self.dtype).eval()
        prompt = torch.tensor([[1, 2, 3]], device=self.device)
        expected = target.generate(
            prompt, max_new_tokens=7, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        actual, stats = cast_generate(
            draft, target, prompt, 7, [], budget=7, branch_k=2, return_stats=True
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertGreater(stats["rounds"], 0)
        first = int(target(prompt).logits[0, -1].argmax())
        stopped = cast_generate(draft, target, prompt, 7, [first], budget=7, branch_k=2)
        self.assertEqual(stopped.shape[1], prompt.shape[1] + 1)


if __name__ == "__main__":
    unittest.main()
