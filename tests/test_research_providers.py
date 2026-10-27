"""Exercise registered model factories, real feature readers, and HF reload."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from draftfit.algorithms.builtin import builtin_algorithm_registry
from draftfit.config import Config
from draftfit.modeling.auto import AutoDraftModel
from draftfit.research.config import ALRConfig
from draftfit.runtime.data_plane.feature_dataloader import FeatureDataLoader
from draftfit.runtime.data_plane.feature_store import LocalFeatureStore
from draftfit.training.model_loading import load_draft_config_source
from tests.test_research_methods import draft_model, target_model


def feature_batch(registration, path, *, max_len=32):
    provider = registration.providers.offline[0]
    reader = provider.build_reader(
        str(path), run_id="research", ttt_length=1, max_len=max_len
    )
    loader = FeatureDataLoader(
        LocalFeatureStore(),
        refs=reader.read(),
        batch_size=1,
        strategy=registration.name,
        device="cpu",
        gc_interval_s=None,
        per_sample_transform=provider.build_normalizer(max_len=max_len),
        collate_fn=provider.build_collator(),
    )
    try:
        return next(iter(loader))
    finally:
        loader.close()


class ResearchProviderTest(unittest.TestCase):
    def test_alr_registered_factories_update_and_reject_wrong_cache(self):
        from draftfit.algorithms.alr_ira.prepare import prepare
        from draftfit.algorithms.alr_ira.providers import validate_prepared_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = target_model()
            target.save_pretrained(root / "target")
            draft_model(4).config.to_json_file(root / "draft.json")
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]})
                + "\n"
            )
            prepare(
                ALRConfig.model_validate(
                    {
                        "method": "alr_ira",
                        "model": {"target_model_path": str(root / "target")},
                        "data": {"input_path": str(corpus)},
                        "training": {
                            "target_layer_ids": [0, 2],
                            "block_size": 4,
                            "max_length": 32,
                        },
                        "output_dir": str(root / "features"),
                    }
                ),
                target=target,
            )
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": str(root / "target"),
                        "draft_model_config": str(root / "draft.json"),
                        "torch_dtype": "float32",
                        "mask_token_id": 31,
                    },
                    "data": {
                        "hidden_states_path": str(root / "features"),
                        "max_length": 32,
                    },
                    "training": {
                        "strategy": "alr_ira",
                        "attention_backend": "sdpa",
                        "ttt_length": 1,
                        "max_steps": 1,
                    },
                }
            )
            registration = builtin_algorithm_registry().resolve("alr_ira")
            draft_config = load_draft_config_source(str(root / "draft.json"))
            draft = registration.providers.model.build_draft(cfg, draft_config)
            parts = registration.providers.model.build_training_model(
                cfg, draft, draft_config, target.config, None
            )
            self.assertTrue(
                all(not p.requires_grad for p in parts.model.lm_head.parameters())
            )
            strategy = registration.providers.step.build(parts.model)
            optimizer = torch.optim.AdamW(draft.parameters(), lr=0.01)
            before = draft.fc.weight.detach().clone()
            output = strategy.forward_loss(
                feature_batch(registration, root / "features")
            )
            output.loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, draft.fc.weight))
            contract = registration.providers.step.resume_contract(
                cfg, draft, parts.model
            )
            self.assertEqual(contract["alr_ira_topk"], 8)
            with self.assertRaisesRegex(ValueError, "block size"):
                validate_prepared_cache(cfg, draft_model(6))
            wrong = cfg.model_copy(
                update={
                    "model": cfg.model.model_copy(update={"target_model_path": "other"})
                }
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_prepared_cache(wrong, draft)
            with self.assertRaises(FileNotFoundError):
                registration.providers.offline[0].build_reader(
                    str(root / "target"), run_id="bad", ttt_length=1, max_len=32
                )

    def test_osprey_registered_factories_preserve_embedding_through_step_and_reload(
        self,
    ):
        from draftfit.offline_capture.transformers import OfflineTransformersCapture
        from draftfit.research.osprey import convert

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = target_model()
            # Existing TargetHead requires indexed BF16 target features.
            target = target.to(dtype=torch.bfloat16)
            target.save_pretrained(root / "target", max_shard_size="2KB")
            convert(str(root / "target"), str(root / "draft"), 3, 16, [0, 1, 2])
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
            capture = OfflineTransformersCapture(target)
            capture.set_capture_layers([0, 1, 2], capture_method="eagle3")
            features = capture.capture(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                loss_mask=torch.ones_like(ids),
            )
            (root / "features").mkdir()
            layout = (
                builtin_algorithm_registry()
                .resolve("osprey")
                .providers.offline[0]
                .capture_layout
            )
            record = layout.materialize(
                {
                    "input_ids": ids[0],
                    "loss_mask": torch.ones_like(ids[0]),
                    "aux_hidden_states": features.hidden_states,
                    "last_hidden_states": features.last_hidden_states,
                }
            )
            torch.save(record, root / "features/row.ckpt")
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": str(root / "target"),
                        "draft_model_config": str(root / "draft"),
                        "draft_checkpoint_path": str(root / "draft"),
                        "torch_dtype": "bfloat16",
                        "load_target_embedding": False,
                    },
                    "data": {
                        "hidden_states_path": str(root / "features"),
                        "max_length": 32,
                    },
                    "training": {
                        "strategy": "osprey",
                        "attention_backend": "sdpa",
                        "ttt_length": 2,
                        "max_steps": 1,
                    },
                }
            )
            registration = builtin_algorithm_registry().resolve("osprey")
            draft_config = load_draft_config_source(str(root / "draft"))
            draft = registration.providers.model.build_draft(cfg, draft_config)
            torch.testing.assert_close(
                draft.embed_tokens.weight, target.get_input_embeddings().weight
            )
            self.assertFalse(draft.embed_tokens.weight.requires_grad)
            parts = registration.providers.model.build_training_model(
                cfg, draft, draft_config, target.config, None
            )
            runtime = registration.providers.step.bind_runtime(cfg, draft, parts.model)
            strategy = registration.providers.step.build(
                parts.model, target_head=parts.target_head, **runtime.options
            )
            optimizer = torch.optim.AdamW(
                [p for p in draft.parameters() if p.requires_grad], lr=0.01
            )
            before = draft.layers[0].self_attn.q_proj.weight.detach().clone()
            result = strategy.forward_loss(
                feature_batch(registration, root / "features")
            )
            result.loss.backward()
            optimizer.step()
            self.assertFalse(
                torch.equal(before, draft.layers[0].self_attn.q_proj.weight)
            )
            state = strategy.checkpoint_state_filter(parts.model.state_dict())
            self.assertIn("embed_tokens.weight", state)
            self.assertEqual(set(state), set(draft.state_dict()))
            draft.save_pretrained(root / "trained", state_dict=state)
            reloaded = AutoDraftModel.from_pretrained(root / "trained")
            for key, tensor in state.items():
                torch.testing.assert_close(
                    reloaded.state_dict()[key], tensor, atol=0, rtol=0
                )
            self.assertTrue(
                any(key.startswith("osprey_") for key in runtime.resume_contract)
            )


if __name__ == "__main__":
    unittest.main()
