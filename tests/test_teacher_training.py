"""CPU teacher → persisted features → production loader → real draft updates.

The teacher has random tiny weights; this proves wiring, not target quality or
GPU/backend numerical parity. No model download or synthetic feature replacement.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculative_train_platform.algorithms.builtin import builtin_algorithm_registry
from speculative_train_platform.algorithms.common.dflash_family_model import (
    OnlineDFlash2Model,
    OnlineDSparkModel,
)
from speculative_train_platform.modeling.draft.dflash2 import (
    DFlash2Config,
    DFlash2DraftModel,
)
from speculative_train_platform.modeling.draft.dspark import DSparkDraftModel
from speculative_train_platform.offline_capture.transformers import (
    OfflineTransformersCapture,
)
from speculative_train_platform.runtime.data_plane.feature_dataloader import (
    FeatureDataLoader,
)
from speculative_train_platform.runtime.data_plane.feature_store import (
    LocalFeatureStore,
)


class TeacherTrainingTest(unittest.TestCase):
    def test_qwen3_capture_disk_loader_and_draft_updates(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            config = Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
                tie_word_embeddings=False,
            )
            config._attn_implementation = "eager"
            teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
            capture = OfflineTransformersCapture(teacher)
            registry = builtin_algorithm_registry()
            for name in ("dspark", "dflash2"):
                with (
                    self.subTest(strategy=name),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    provider = registry.resolve(name).providers.offline[0]
                    layout = provider.capture_layout
                    capture.set_capture_layers(
                        [0, 2], capture_method=layout.capture_method
                    )
                    captured_rows = []
                    # Unequal sequences exercise padding in the real collator.
                    for index, length in enumerate((12, 16)):
                        ids = (torch.arange(length).view(1, -1) + 1) % 63
                        mask = torch.ones_like(ids)
                        mask[:, :2] = 0
                        result = capture.capture(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            loss_mask=mask,
                        )
                        self.assertFalse(result.hidden_states.requires_grad)
                        captured_rows.append(result)
                        record = layout.materialize(
                            {
                                "input_ids": result.input_ids[0],
                                "loss_mask": result.loss_mask[0],
                                "aux_hidden_states": result.hidden_states,
                                "last_hidden_states": result.last_hidden_states,
                            }
                        )
                        torch.save(record, Path(directory) / f"{index}.ckpt")
                    reader = provider.build_reader(
                        directory,
                        run_id=name,
                        ttt_length=4,
                        max_len=16,
                    )
                    loader = FeatureDataLoader(
                        LocalFeatureStore(),
                        refs=reader.read(),
                        batch_size=2,
                        strategy=name,
                        device="cpu",
                        gc_interval_s=None,
                        per_sample_transform=provider.build_normalizer(max_len=16),
                        collate_fn=provider.build_collator(),
                    )
                    try:
                        batches = list(loader)
                    finally:
                        loader.close()
                    self.assertEqual(len(batches), 1)
                    batch = batches[0].tensors
                    self.assertEqual(tuple(batch["hidden_states"].shape), (2, 16, 64))
                    self.assertTrue(
                        torch.equal(
                            batch["hidden_states"][0, :12],
                            captured_rows[0].hidden_states[0],
                        )
                    )
                    self.assertEqual(batch["loss_mask"][0, 12:].sum().item(), 0)
                    if name == "dflash2":
                        draft = DFlash2DraftModel(
                            DFlash2Config(
                                hidden_size=32,
                                intermediate_size=64,
                                num_hidden_layers=1,
                                num_attention_heads=4,
                                num_key_value_heads=2,
                                head_dim=8,
                                vocab_size=64,
                                target_num_hidden_layers=3,
                                target_layer_ids=[0, 2],
                                mask_token_id=63,
                                block_size=4,
                                conv_group_size=8,
                                selector_rank=8,
                                selector_top_k=4,
                            )
                        )
                        wrapper_class = OnlineDFlash2Model
                        watched = draft.candidate_selector.hidden_projection.weight
                    else:
                        draft_config = Qwen3Config.from_dict(config.to_dict())
                        draft_config.num_hidden_layers = 1
                        draft_config.layer_types = ["full_attention"]
                        draft_config.num_target_layers = 3
                        draft_config.block_size = 4
                        draft_config.dflash_config = {
                            "target_layer_ids": [0, 2],
                            "projector_type": "dspark",
                            "mask_token_id": 63,
                            "markov_rank": 8,
                        }
                        draft = DSparkDraftModel(draft_config)
                        wrapper_class = OnlineDSparkModel
                        watched = draft.fc.weight
                        self.assertTrue(
                            torch.equal(
                                batch["target_last_hidden_states"][0, :12],
                                captured_rows[0].last_hidden_states[0],
                            )
                        )
                    wrapper = wrapper_class(
                        draft_model=draft,
                        target_lm_head=teacher.lm_head,
                        target_embed_tokens=teacher.get_input_embeddings(),
                        mask_token_id=63,
                        block_size=4,
                        attention_backend="sdpa",
                        num_anchors=2,
                        anchor_sampling="uniform",
                    )
                    before = watched.detach().clone()
                    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
                    loss, _, _ = wrapper(**batch)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    self.assertIsNotNone(watched.grad)
                    self.assertTrue(torch.isfinite(watched.grad).all().item())
                    self.assertGreater(watched.grad.norm().item(), 0)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, watched))
                    self.assertTrue(
                        all(
                            parameter.grad is None for parameter in teacher.parameters()
                        )
                    )


if __name__ == "__main__":
    unittest.main()
