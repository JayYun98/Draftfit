"""Real CUDA draft updates/reload on synthetic features, not target certification."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn


@unittest.skipUnless(torch.cuda.is_available(), "draft update gate requires CUDA")
class DraftFamilyGpuSmoke(unittest.TestCase):
    def test_updates_and_reload(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDFlashModel, OnlineDSparkModel, OnlineDominoModel,
        )
        from specforge.modeling.auto import AutoDraftModel
        from specforge.training.model_loading import load_draft_config_source

        root = Path(__file__).resolve().parents[2]
        for name, wrapper in (("dflash", OnlineDFlashModel), ("dspark", OnlineDSparkModel),
                              ("domino", OnlineDominoModel)):
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as temporary:
                torch.manual_seed(7)
                payload = json.loads((root / f"configs/qwen3-8b-{name}.json").read_text())
                payload.pop("auto_map", None)
                payload.update(hidden_size=64, intermediate_size=128, vocab_size=64,
                               head_dim=16, num_attention_heads=4, num_key_value_heads=2,
                               num_hidden_layers=2, num_target_layers=8, block_size=4,
                               layer_types=["full_attention"] * 2, max_position_embeddings=64,
                               bos_token_id=0, eos_token_id=1)
                payload.setdefault("dflash_config", {}).update(
                    target_layer_ids=[1, 3, 5], mask_token_id=63,
                    markov_rank=8, emb_dim=16, gru_hidden_dim=16, pure_draft_prefix_len=0)
                source = Path(temporary) / "draft.json"
                source.write_text(json.dumps(payload))
                draft = AutoDraftModel.from_config(load_draft_config_source(str(source))).cuda().float()
                head = nn.Linear(64, 64, bias=False).cuda().requires_grad_(False)
                embedding = nn.Embedding(64, 64).cuda().requires_grad_(False)
                model = wrapper(draft_model=draft, target_lm_head=head,
                                target_embed_tokens=embedding, mask_token_id=63,
                                block_size=4, num_anchors=2, attention_backend="sdpa",
                                anchor_sampling="uniform").cuda()
                batch = dict(input_ids=torch.randint(0, 62, (1, 16), device="cuda"),
                             hidden_states=torch.randn(1, 16, 192, device="cuda"),
                             loss_mask=torch.ones(1, 16, device="cuda"))
                if name == "dspark":
                    batch["target_last_hidden_states"] = torch.randn(1, 16, 64, device="cuda")
                before = {key: value.detach().clone() for key, value in draft.named_parameters()}
                optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch)[0]
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all().item()
                                        for p in draft.parameters() if p.grad is not None))
                    optimizer.step()
                self.assertTrue(any(not torch.equal(before[key], value)
                                    for key, value in draft.named_parameters()))
                destination = Path(temporary) / "export"
                draft.save_pretrained(destination)
                reloaded = type(draft).from_pretrained(destination).cuda().float()
                for key, value in draft.state_dict().items():
                    torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
