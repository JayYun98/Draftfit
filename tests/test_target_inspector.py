import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from speculative_train_platform.cli import main as cli_main
from speculative_train_platform.inference.target_adapter import TargetAdapter
from speculative_train_platform.target_inspector import inspect_target


class TargetInspectorTest(unittest.TestCase):
    def _inspect(self, config, tokenizer=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            if tokenizer is not None:
                (root / "tokenizer_config.json").write_text(
                    json.dumps(tokenizer), encoding="utf-8"
                )
            return inspect_target(directory, local_only=True)

    def test_dense_config_has_kv_state_and_chat_template(self):
        result = self._inspect(
            {
                "model_type": "granite",
                "architectures": ["GraniteForCausalLM"],
                "num_hidden_layers": 12,
            },
            {"chat_template": "{{ messages }}"},
        )
        facts = result["facts"]
        self.assertEqual(facts["architecture_lane"], "dense")
        self.assertEqual(facts["state_kind"], "kv")
        self.assertTrue(facts["chat_template_present"])
        self.assertTrue(
            any(item["field"] == "architecture" for item in result["evidence"])
        )
        self.assertIn("target_tap_candidates", result["recommendations"]["provenance"])

    def test_conv_attention_config_requires_state_replay(self):
        result = self._inspect(
            {
                "model_type": "lfm2",
                "architectures": ["Lfm2ForCausalLM"],
                "num_hidden_layers": 6,
                "layer_types": [
                    "conv",
                    "conv",
                    "full_attention",
                    "conv",
                    "conv",
                    "full_attention",
                ],
            }
        )
        self.assertEqual(result["facts"]["architecture_lane"], "hybrid_stateful")
        self.assertIn(
            "state_snapshot_rollback_replay",
            result["recommendations"]["required_gates"],
        )
        self.assertIn(2, result["recommendations"]["target_tap_candidates"])

    def test_moe_config_is_not_misclassified_as_dense(self):
        result = self._inspect(
            {
                "model_type": "granitemoe_swa",
                "num_hidden_layers": 8,
                "num_local_experts": 16,
            }
        )
        self.assertEqual(result["facts"]["architecture_lane"], "moe")

    def test_linear_attention_hybrid_has_linear_state_and_full_attention_taps(self):
        result = self._inspect(
            {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 8,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
            }
        )
        facts = result["facts"]
        self.assertEqual(facts["architecture_lane"], "hybrid_stateful")
        self.assertEqual(facts["state_kind"], "linear_plus_kv")
        self.assertEqual(result["recommendations"]["target_tap_candidates"], [3, 7])

    def test_target_inspect_is_available_from_single_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "rwkv7", "num_hidden_layers": 4}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    cli_main(["target", "inspect", directory, "--local-only"]), 0
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["facts"]["state_kind"], "recurrent")

    def test_inspector_metadata_builds_dense_and_hybrid_adapters(self):
        dense = self._inspect(
            {
                "model_type": "granite",
                "architectures": ["GraniteForCausalLM"],
                "num_hidden_layers": 8,
            },
            {"chat_template": "{{ messages }}"},
        )
        hybrid = self._inspect(
            {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 8,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                ],
            },
            {"chat_template": "{{ messages }}"},
        )
        # A local inspection defaults to ``main``; production callers replace
        # it with the resolved immutable commit before creating an adapter.
        dense["revision"] = "dense-0123456789abcdef"
        hybrid["revision"] = "hybrid-0123456789abcdef"
        dense_adapter = TargetAdapter.from_inspection(dense)
        hybrid_adapter = TargetAdapter.from_inspection(hybrid)
        self.assertEqual(dense_adapter.architecture_lane, "dense")
        self.assertEqual(dense_adapter.state_kind, "kv")
        self.assertEqual(hybrid_adapter.architecture_lane, "hybrid_stateful")
        self.assertEqual(hybrid_adapter.state_kind, "linear_plus_kv")
        self.assertEqual(hybrid_adapter.capture_layers, (2, 6))


if __name__ == "__main__":
    unittest.main()
