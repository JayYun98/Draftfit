"""Quantized teacher loading and feature identity boundaries; no fake GPU claims."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3Model

from draftfit.config import Config
from draftfit.inference.capture_manifest import (
    build_capture_manifest,
    write_capture_manifest,
)
from draftfit.modeling.target.quantization import (
    target_quantization_identity,
    validate_quantized_feature_source,
)
from draftfit.offline_capture.transformers import OfflineTransformersCapture


class QuantizedTargetTest(unittest.TestCase):
    def config(self, root, recipe="bitsandbytes-4bit"):
        return Config.model_validate(
            {
                "model": {
                    "target_model_path": str(root / "target"),
                    "target_backend": "transformers",
                    "target_quantization": recipe,
                },
                "data": {"hidden_states_path": str(root / "features")},
                "training": {"strategy": "dspark"},
            }
        )

    def manifest(self, cfg, target):
        return build_capture_manifest(
            strategy="dspark",
            capture_method="dspark",
            capture_layers=[0, 2],
            feature_names=[
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            ],
            target_model=cfg.model.target_model_path,
            target_revision="revision",
            capture_backend="transformers",
            capture_settings={
                "dtype": cfg.model.torch_dtype,
                "attention_implementation": cfg.model.hf_attn_implementation,
            },
            target_quantization=target_quantization_identity(
                target, cfg.model.target_quantization
            ),
        )

    def test_recipe_constraints_and_prequantized_identity(self):
        target = SimpleNamespace(quantization_config={"quant_method": "awq", "bits": 4})
        self.assertEqual(
            target_quantization_identity(target)["checkpoint_config"]["bits"], 4
        )
        self.assertIsNone(target_quantization_identity(SimpleNamespace()))
        self.assertEqual(
            target_quantization_identity(
                SimpleNamespace(
                    text_config={
                        "quantization_config": {"quant_method": "awq", "bits": 4}
                    }
                )
            ),
            target_quantization_identity(target),
        )
        with self.assertRaisesRegex(ValueError, "already quantized"):
            target_quantization_identity(target, "bitsandbytes-4bit")
        with self.assertRaisesRegex(ValueError, "require target_backend=transformers"):
            Config.model_validate(
                {
                    "model": {
                        "target_model_path": "target",
                        "target_quantization": "bitsandbytes-8bit",
                    }
                }
            )

    def test_packed_teacher_placement_belongs_to_loader(self):
        model = Qwen3Model(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        )
        model.config.quantization_config = {"quant_method": "gptq", "bits": 4}
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=model.config),
            patch("transformers.AutoModel.from_pretrained", return_value=model) as load,
            patch.object(model, "to", side_effect=AssertionError("packed model moved")),
        ):
            capture = OfflineTransformersCapture.from_pretrained(
                "packed", device="cuda:3"
            )
        self.assertEqual(load.call_args.kwargs["device_map"], {"": "cuda:3"})
        self.assertEqual(capture.quantization_identity["checkpoint_config"]["bits"], 4)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_explicit_recipes_reach_transformers_without_moving_packed_modules(self):
        model = Qwen3Model(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        )
        for bits in (4, 8):
            with (
                self.subTest(bits=bits),
                patch(
                    "transformers.AutoConfig.from_pretrained", return_value=model.config
                ),
                patch(
                    "transformers.AutoModel.from_pretrained", return_value=model
                ) as load,
                patch(
                    "transformers.BitsAndBytesConfig", return_value=object()
                ) as recipe,
                patch.object(
                    model, "to", side_effect=AssertionError("packed model moved")
                ),
            ):
                capture = OfflineTransformersCapture.from_pretrained(
                    "floating",
                    device="cuda:0",
                    torch_dtype=torch.bfloat16,
                    quantization=f"bitsandbytes-{bits}bit",
                )
            self.assertEqual(recipe.call_args.kwargs["load_in_4bit"], bits == 4)
            self.assertEqual(recipe.call_args.kwargs["load_in_8bit"], bits == 8)
            self.assertEqual(
                recipe.call_args.kwargs["llm_int8_skip_modules"], ["lm_head"]
            )
            self.assertEqual(load.call_args.kwargs["device_map"], {"": "cuda:0"})
            self.assertEqual(
                capture.quantization_identity["recipe"], f"bitsandbytes-{bits}bit"
            )

    def test_quantized_cache_requires_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            target = SimpleNamespace()
            with self.assertRaisesRegex(ValueError, "requires capture_manifest"):
                validate_quantized_feature_source(cfg, target)
            path = root / "features/capture_manifest.json"
            manifest = self.manifest(cfg, target)
            write_capture_manifest(str(path), manifest)
            validate_quantized_feature_source(cfg, target)
            cfg.model.target_quantization = "bitsandbytes-8bit"
            with self.assertRaisesRegex(ValueError, "quantization differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.target_quantization = "auto"
            with self.assertRaisesRegex(ValueError, "quantization differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.target_quantization = "bitsandbytes-4bit"
            cfg.model.target_model_path = "other-target"
            with self.assertRaisesRegex(ValueError, "model/revision differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.target_model_path = str(root / "target")
            cfg.model.target_revision = "another-revision"
            with self.assertRaisesRegex(ValueError, "model/revision differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.target_revision = None
            cfg.model.target_backend = "vllm"
            with self.assertRaisesRegex(ValueError, "backend differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.target_backend = "transformers"
            cfg.model.torch_dtype = "float32"
            with self.assertRaisesRegex(ValueError, "compute dtype differs"):
                validate_quantized_feature_source(cfg, target)
            cfg.model.torch_dtype = "bfloat16"
            cfg.model.hf_attn_implementation = "eager"
            with self.assertRaisesRegex(ValueError, "attention implementation differs"):
                validate_quantized_feature_source(cfg, target)

    def test_eval_cache_checked_and_dense_legacy_cache_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root, "auto")
            validate_quantized_feature_source(cfg, SimpleNamespace())
            cfg.model.target_quantization = "bitsandbytes-4bit"
            write_capture_manifest(
                str(root / "features/capture_manifest.json"),
                self.manifest(cfg, SimpleNamespace()),
            )
            cfg.data.eval_hidden_states_path = str(root / "eval")
            with self.assertRaisesRegex(ValueError, "requires capture_manifest"):
                validate_quantized_feature_source(cfg, SimpleNamespace())

    def test_training_rejects_stale_cache_before_model_assembly(self):
        from draftfit.algorithms.builtin import builtin_algorithm_registry
        from draftfit.training.assembly import build_training_run

        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(Path(directory))
            with (
                patch(
                    "draftfit.modeling.target.target_utils.load_target_config",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "draftfit.training.assembly.build_model_bundle",
                    side_effect=AssertionError("model allocated"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "requires capture_manifest"):
                    build_training_run(
                        cfg, algorithm=builtin_algorithm_registry().resolve("dspark")
                    )

    def test_export_restores_checkpoint_quantization_metadata(self):
        from draftfit.algorithms.common.providers import MODEL_PROVENANCE_CONTRACT_KEY
        from draftfit.export.to_hf import export_to_hf
        from draftfit.modeling.auto import AutoDraftModel
        from draftfit.modeling.draft.dflash2 import DFlash2Config, DFlash2DraftModel

        config = DFlash2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            target_num_hidden_layers=3,
            target_layer_ids=[0, 2],
            mask_token_id=31,
            block_size=4,
            conv_group_size=8,
            selector_rank=8,
            selector_top_k=4,
        )
        config.architectures = ["DFlash2DraftModel"]
        draft = DFlash2DraftModel(config)
        identity = target_quantization_identity(SimpleNamespace(), "bitsandbytes-4bit")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.to_json_file(root / "draft.json")
            state = {
                "draft_state_dict": draft.state_dict(),
                MODEL_PROVENANCE_CONTRACT_KEY: (("target_quantization", identity),),
            }
            torch.save(state, root / "training_state.pt")
            export_to_hf(
                str(root / "training_state.pt"),
                str(root / "draft.json"),
                str(root / "export"),
            )
            reloaded = AutoDraftModel.from_pretrained(
                root / "export", config=DFlash2Config.from_pretrained(root / "export")
            )
            self.assertEqual(reloaded.config.draftfit_target_quantization, identity)
            self.assertFalse(getattr(reloaded.config, "quantization_config", None))
            for key, value in draft.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[key],
                    value.to(reloaded.state_dict()[key].dtype),
                    rtol=0,
                    atol=0,
                )

    def test_quantized_resume_contract_includes_recipe_and_backend(self):
        from draftfit.training.provenance import _compute_model_resume_provenance

        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(Path(directory))
            target = SimpleNamespace()
            first = _compute_model_resume_provenance(
                cfg, SimpleNamespace(), target, capture_layers=[0, 2]
            )
            cfg.model.target_quantization = "bitsandbytes-8bit"
            second = _compute_model_resume_provenance(
                cfg, SimpleNamespace(), target, capture_layers=[0, 2]
            )
            self.assertNotEqual(first, second)
            self.assertEqual(first["quantized_target_backend"], "transformers")
            cfg.model.target_quantization = "auto"
            dense = _compute_model_resume_provenance(
                cfg, SimpleNamespace(), target, capture_layers=[0, 2]
            )
            self.assertNotIn("target_quantization", dense)

    def test_integer_embedding_and_head_are_rejected_without_copy(self):
        from safetensors.torch import save_file

        from draftfit.modeling.target.target_utils import TargetEmbeddingsAndHead

        config = Qwen3Config(vocab_size=32, hidden_size=16)
        parts = TargetEmbeddingsAndHead(config)
        before = parts.embed_tokens.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            save_file(
                {"model.embed_tokens.weight": torch.ones(32, 16, dtype=torch.uint8)},
                path,
            )
            with self.assertRaisesRegex(
                ValueError, "packed embeddings or output heads"
            ):
                parts._load_file_content(
                    path,
                    ["model.embed_tokens.weight"],
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                )
        self.assertTrue(torch.equal(before, parts.embed_tokens.weight))
        from draftfit.modeling.target.quantization import require_floating_target_weight

        with self.assertRaisesRegex(ValueError, "packed embeddings or output heads"):
            require_floating_target_weight(
                torch.ones(2, 2).to(torch.float8_e4m3fn), "lm_head.weight"
            )


if __name__ == "__main__":
    unittest.main()
