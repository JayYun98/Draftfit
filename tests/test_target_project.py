import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speculative_train_platform.cli import main
from speculative_train_platform.config import Config
from speculative_train_platform.target_project import algorithm_catalog, prepare_project


class TargetProjectTest(unittest.TestCase):
    def test_owned_teacher_preparation_does_not_inherit_sglang_tuning(self):
        for backend in ("transformers", "vllm"):
            result = self.prepare(
                backend, train_data=str(self.data), target_backend=backend
            )
            config = Config.from_file(result["config"])
            self.assertEqual(config.model.target_backend, backend)
            self.assertEqual(config.mode, "online")
            self.assertEqual(config.model.sglang_attention_backend, "flashinfer")
            self.assertIsNone(config.model.sglang_linear_attn_backend)
            self.assertEqual(
                config.deployment.disaggregated.managed_local.capture_servers[
                    0
                ].cuda_visible_devices,
                ["0"],
            )

    def test_metadata_validation_without_model_construction(self):
        from speculative_train_platform.application.project_validation import (
            validate_draft_metadata,
        )

        arguments = dict(
            strategy="dspark",
            target_depth=8,
            target_metadata={
                "hidden_size": 32,
                "vocab_size": 60,
                "padded_vocab_size": 64,
            },
            draft=SimpleNamespace(
                hidden_size=32,
                vocab_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
            ),
            layers=[1, 3, 5],
            mask_token_id=63,
        )
        validate_draft_metadata(**arguments)
        for change, message in (
            ({"layers": [1, 1]}, "capture layers"),
            ({"layers": [True]}, "capture layers"),
            ({"layers": [8]}, "capture layers"),
            ({"mask_token_id": 64}, "mask_token_id"),
            (
                {
                    "draft": SimpleNamespace(
                        hidden_size=16,
                        vocab_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                    )
                },
                "hidden_size must match",
            ),
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_draft_metadata(**(arguments | change))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.target.mkdir()
        self.metadata = {
            "model_type": "custom_moe_hybrid",
            "architectures": ["UserTargetModel"],
            "hidden_size": 32,
            "vocab_size": 64,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "max_position_embeddings": 2048,
            "num_local_experts": 4,
            "layer_types": ["linear_attention", "full_attention"] * 4,
            "auto_map": {"AutoModelForCausalLM": "never_execute.CustomModel"},
        }
        (self.target / "config.json").write_text(json.dumps(self.metadata))
        (self.target / "tokenizer_config.json").write_text(
            json.dumps(
                {"chat_template": "{% generation %}{{ messages }}{% endgeneration %}"}
            )
        )
        (self.target / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.language_model.embed_tokens.weight": "not-downloaded.safetensors",
                        "lm_head.weight": "not-downloaded.safetensors",
                    }
                }
            )
        )
        self.data = self.root / "data.jsonl"
        self.data.write_text(
            '{"conversations": [{"role": "assistant", "content": "hello"}]}\n'
        )

    def prepare(self, name="project", **kwargs):
        return prepare_project(
            str(self.target), str(self.root / name), local_only=True, **kwargs
        )

    def test_catalog_is_derived_from_registry(self):
        catalog = algorithm_catalog()
        self.assertEqual(
            {x["algorithm"] for x in catalog},
            {"dspark", "dflash", "dflash2", "eagle3", "peagle", "domino"},
        )
        dspark = next(x for x in catalog if x["algorithm"] == "dspark")
        self.assertIn(
            "target_last_hidden_states", dspark["features"][0]["required_tensors"]
        )

    def test_metadata_only_dspark_and_dflash_build_real_small_draft_models(self):
        from speculative_train_platform.modeling.auto import AutoDraftModel
        from speculative_train_platform.training.model_loading import (
            load_draft_config_source,
        )

        with patch(
            "speculative_train_platform.modeling.target.target_utils.load_target_config",
            side_effect=AssertionError("must not load target"),
        ):
            for strategy in ("dspark", "dflash", "dflash2"):
                with self.subTest(strategy=strategy):
                    result = self.prepare(
                        strategy,
                        strategy=strategy,
                        hidden_states="./features",
                        overrides=[
                            "model.target_layer_ids=[1,3,5]",
                            "model.draft_num_hidden_layers=2",
                        ],
                    )
                    cfg = Config.from_file(result["config"])
                    manifest = json.loads(Path(result["manifest"]).read_text())
                    self.assertEqual(
                        cfg.model.embedding_key,
                        "model.language_model.embed_tokens.weight",
                    )
                    self.assertEqual(manifest["capture_layers"], [1, 3, 5])
                    self.assertEqual(manifest["validation"]["training"], "not_run")
                    draft = load_draft_config_source(cfg.model.draft_model_config)
                    self.assertEqual(draft.num_hidden_layers, 2)
                    model = AutoDraftModel.from_config(draft)
                    self.assertEqual(list(model.target_layer_ids), [1, 3, 5])
                    self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_online_topology_and_eagle_specific_overrides(self):
        for strategy in ("eagle3", "peagle", "dspark"):
            result = self.prepare(
                strategy,
                strategy=strategy,
                train_data=str(self.data),
                overrides=["model.vocab_mapping_path=/explicit/shared-mapping.pt"],
            )
            cfg = Config.from_file(result["config"])
            self.assertEqual(
                cfg.deployment.disaggregated.managed_local.trainer_cuda_visible_devices,
                ["1"],
            )
            self.assertEqual(
                cfg.deployment.disaggregated.managed_local.capture_servers[
                    0
                ].cuda_visible_devices,
                ["0"],
            )
            self.assertTrue(cfg.model.sglang_disable_cuda_graph)
            if strategy == "eagle3":
                self.assertEqual(cfg.model.draft_num_hidden_layers, 1)
                self.assertIsNone(cfg.model.draft_block_size)

    def test_custom_draft_keeps_depth_unless_explicitly_overridden(self):
        generated = self.prepare(
            "first",
            hidden_states="features",
            overrides=["model.draft_num_hidden_layers=2"],
        )
        source = Path(generated["project"]) / "draft.json"
        result = self.prepare(
            "second", hidden_states="features", draft_config=str(source)
        )
        draft = json.loads((Path(result["project"]) / "draft.json").read_text())
        self.assertEqual(draft["num_hidden_layers"], 2)
        draft["hidden_size"] = 64
        source = self.root / "wrong-width.json"
        source.write_text(json.dumps(draft))
        with self.assertRaisesRegex(ValueError, "hidden_size must match target"):
            self.prepare(
                "wrong-width", hidden_states="features", draft_config=str(source)
            )
        self.assertFalse((self.root / "wrong-width").exists())

    def test_invalid_inputs_do_not_create_or_overwrite_projects(self):
        for kwargs in (
            {"strategy": "not-implemented"},
            {"overrides": ["model.target_layer_ids=[99]"]},
            {"overrides": ["model.mask_token_id=64"]},
            {"overrides": ["model.typo=1"]},
            {"strategy": "domino"},
            {"strategy": "eagle3", "overrides": ["model.draft_num_hidden_layers=5"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, KeyError)):
                self.prepare(hidden_states="features", **kwargs)
            self.assertFalse((self.root / "project").exists())
        result = self.prepare(hidden_states="features")
        before = Path(result["config"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(hidden_states="features")
        self.assertEqual(Path(result["config"]).read_bytes(), before)

    def test_public_input_checks_fail_before_output(self):
        for content in (
            "",
            "not-json",
            '{"messages": []}',
            '{"conversations": [{"role": "user", "content": "secret"}]}',
            '{"conversations": [false]}',
        ):
            self.data.write_text(content)
            with self.assertRaises(ValueError):
                self.prepare(train_data=str(self.data))
            self.assertFalse((self.root / "project").exists())
        self.data.write_text(
            '{"conversations": [{"from": "gpt", "value": "answer"}]}\n'
        )
        (self.target / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"})
        )
        with self.assertRaisesRegex(ValueError, "generation"):
            self.prepare(train_data=str(self.data))
        with self.assertRaisesRegex(ValueError, "command arguments"):
            self.prepare(
                hidden_states="features",
                overrides=["model.draft_checkpoint_path=another/model"],
            )

    def test_cli_prepare_outputs_parseable_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "target",
                        "prepare",
                        str(self.target),
                        "--local-only",
                        "--output-dir",
                        str(self.root / "cli"),
                        "--hidden-states",
                        "features",
                        "--set",
                        "model.target_layer_ids=[1,3,5]",
                    ]
                ),
                0,
            )
        self.assertTrue(Path(json.loads(output.getvalue())["config"]).is_file())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(
                [
                    "target",
                    "prepare",
                    str(self.target),
                    "--output-dir",
                    str(self.root / "bad"),
                    "--hidden-states",
                    "features",
                    "--strategy",
                    "unknown",
                ]
            )
        self.assertEqual(error.exception.code, 2)

    def test_source_parent_overrides_and_remote_code_fail_before_resolution(self):
        for override in (
            'model={"target_model_path": "other/target"}',
            'data={"hidden_states_path": "other-features"}',
            "model.trust_remote_code=true",
        ):
            with (
                self.subTest(override=override),
                patch(
                    "speculative_train_platform.training.model_loading.resolve_draft_config",
                    side_effect=AssertionError(
                        "must reject before resolving remote draft config"
                    ),
                ) as resolve,
            ):
                with self.assertRaisesRegex(
                    ValueError, "command arguments|metadata-only"
                ):
                    self.prepare(hidden_states="features", overrides=[override])
                resolve.assert_not_called()
                self.assertFalse((self.root / "project").exists())

    def test_inspection_pins_metadata_fetches_before_reading_remote_files(self):
        from speculative_train_platform.target_inspector import inspect_target

        sha = "a" * 40
        with (
            patch(
                "speculative_train_platform.target_inspector._remote_api",
                return_value={"sha": sha},
            ),
            patch(
                "speculative_train_platform.target_inspector._remote_json",
                return_value=self.metadata,
            ) as get_json,
            patch(
                "speculative_train_platform.target_inspector._remote_text",
                return_value=None,
            ) as get_text,
        ):
            result = inspect_target("user/target", revision="branch")
        self.assertEqual(result["revision"], sha)
        self.assertTrue(
            all(
                call.args[2] == sha
                for call in get_json.call_args_list + get_text.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
