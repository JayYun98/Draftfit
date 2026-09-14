import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from specforge.inference.acceptance import validate_acceptance_summary
from specforge.inference.parity import (
    compare_feature_manifests,
    compare_state_replay,
    compare_state_snapshots,
    compare_token_ids,
    feature_manifest,
)
from specforge.inference.state import TargetStateAdapter, TargetStateSnapshot
from specforge.inference.target_adapter import (
    ChatTemplateRenderer,
    LingRenderer,
    LingTargetAdapter,
    TargetAdapter,
)
from specforge.sweep import build_sweep
from specforge.torchspec_bridge import TorchSpecLaunch


class _Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False
    ):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class _FallbackTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [[ord(char) for char in text]]}


class _FakeTensor:
    dtype = "float32"
    shape = (2,)

    def __init__(self, data):
        self.data = bytes(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.data


class InferenceContractTest(unittest.TestCase):
    @staticmethod
    def _inspection(lane, state, taps=(0, 3), depth=4):
        return {
            "schema_version": "target_inspect_v1",
            "source": "org/model",
            "revision": "0123456789abcdef",
            "facts": {
                "architecture_lane": lane,
                "state_kind": state,
                "num_hidden_layers": depth,
                "chat_template_present": True,
            },
            "recommendations": {"target_tap_candidates": list(taps)},
        }

    def test_generic_factory_supports_dense_and_hybrid_without_trainer_changes(self):
        dense = TargetAdapter.from_inspection(self._inspection("dense", "kv"))
        hybrid = TargetAdapter.from_inspection(
            self._inspection("hybrid_stateful", "linear_plus_kv", taps=(1, 3))
        )
        self.assertEqual(dense.capture_layers, (0, 3))
        self.assertEqual(hybrid.capture_layers, (1, 3))
        self.assertEqual(dense.renderer.name, "hf-chat-template")
        self.assertEqual(
            dense.render(_Tokenizer(), [{"role": "user", "content": "u"}]).renderer,
            "hf-chat-template",
        )
        self.assertNotEqual(dense.contract_hash(), hybrid.contract_hash())

    def test_generic_factory_accepts_moe_and_rejects_unknown_state(self):
        adapter = TargetAdapter.from_inspection(self._inspection("moe", "kv"))
        self.assertEqual(adapter.architecture_lane, "moe")
        with self.assertRaisesRegex(ValueError, "unsupported target state kind"):
            TargetAdapter.from_inspection(self._inspection("dense", "mystery_cache"))
        with self.assertRaisesRegex(ValueError, "incompatible with architecture lane"):
            TargetAdapter.from_inspection(self._inspection("dense", "recurrent"))

    def test_generic_factory_requires_pinned_revision_and_explicit_renderer(self):
        with self.assertRaisesRegex(ValueError, "pinned model revision"):
            TargetAdapter.from_inspection(
                {**self._inspection("dense", "kv"), "revision": "main"}
            )
        with self.assertRaisesRegex(
            ValueError, "requires tokenizer.apply_chat_template"
        ):
            ChatTemplateRenderer().render(
                _FallbackTokenizer(), [{"role": "user", "content": "u"}]
            )

    def test_ling_renderer_marks_assistant_content(self):
        result = LingRenderer().render(
            _Tokenizer(),
            [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "ok"}],
        )
        self.assertEqual(result.renderer, "ling-3.0")
        self.assertEqual(sum(result.loss_mask), 2)
        self.assertEqual(result.loss_mask[-2:], (1, 1))
        self.assertEqual(
            TargetAdapter.ling(capture_layers=[3, 15]).contract_hash().__len__(), 16
        )
        self.assertEqual(
            LingTargetAdapter(capture_layers=[3]).model_id, "inclusionAI/Ling-3.0-tiny"
        )

    def test_ling_renderer_fallback_is_deterministic_and_accepts_batch_wrappers(self):
        result = LingRenderer().render(
            _FallbackTokenizer(),
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "ok"}],
        )
        self.assertIn("<role>HUMAN</role>u", result.text)
        self.assertIn("<role>ASSISTANT</role>ok<|role_end|>", result.text)
        self.assertEqual(len(result.input_ids), len(result.loss_mask))
        self.assertEqual(sum(result.loss_mask), 2)

    def test_ling_renderer_normalizes_legacy_role_aliases_before_template(self):
        result = LingRenderer().render(
            _Tokenizer(),
            [{"role": "human", "content": "u"}, {"role": "gpt", "content": "a"}],
        )
        self.assertIn("<user>u", result.text)
        self.assertIn("<assistant>a", result.text)

    def test_ling_adapter_rejects_invalid_capture_layers(self):
        with self.assertRaises(ValueError):
            TargetAdapter.ling(capture_layers=[2, 2])
        with self.assertRaises(ValueError):
            TargetAdapter.ling(capture_layers=[-1])
        self.assertEqual(
            TargetAdapter.ling().validate_capture_layers([2, 7], num_hidden_layers=24),
            (2, 7),
        )
        with self.assertRaises(ValueError):
            TargetAdapter.ling().validate_capture_layers([2, 2])

    def test_parity_reports_mismatch(self):
        left = {
            "schema_version": 1,
            "aux_layer_ids": [3],
            "features": {"x": {"shape": [1, 2], "dtype": "bf16"}},
        }
        right = {
            "schema_version": 1,
            "aux_layer_ids": [5],
            "features": {"x": {"shape": [1, 3], "dtype": "bf16"}},
        }
        self.assertFalse(compare_feature_manifests(left, right)["passed"])
        self.assertEqual(
            feature_manifest({"x": {"shape": [1, 2], "dtype": "bf16"}})["features"][
                "x"
            ]["shape"],
            [1, 2],
        )
        self.assertTrue(compare_token_ids([1, 2], [1, 2])["passed"])
        self.assertTrue(
            compare_state_snapshots({"state": [1]}, {"state": [1]})["passed"]
        )

    def test_state_snapshot_is_immutable_and_replays_with_exact_prefix(self):
        source = {"conv": _FakeTensor(b"ab"), "step": 2}
        adapter = TargetStateAdapter(
            target_model_version="ling@rev-a", state_kind="moe_plus_conv"
        )
        snapshot = adapter.capture(source, [10, 11], metadata={"backend": "sglang"})
        source["step"] = 99
        encoded = snapshot.to_dict()
        restored = TargetStateSnapshot.from_dict(encoded)
        self.assertTrue(compare_state_replay(restored, encoded)["passed"])
        self.assertEqual(restored.sequence_position, 2)
        with self.assertRaisesRegex(ValueError, "token prefix mismatch"):
            adapter.restore(restored, token_ids=[10, 12])
        with self.assertRaisesRegex(ValueError, "token_ids are required"):
            adapter.restore(restored)
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            TargetStateAdapter(
                target_model_version="ling@rev-b", state_kind="moe_plus_conv"
            ).restore(restored)

    def test_state_adapter_callbacks_are_the_engine_boundary(self):
        adapter = TargetStateAdapter(
            target_model_version="target@1",
            state_kind="linear_plus_kv",
            capture_state=lambda state: state["cache"],
            restore_state=lambda snapshot, current: {
                "snapshot": snapshot,
                "current": current,
            },
        )
        snapshot = adapter.capture({"cache": {"x": [1, 2]}}, [7])
        result = adapter.restore(snapshot, token_ids=[7], current_state="engine")
        self.assertEqual(result["current"], "engine")
        self.assertEqual(result["snapshot"], {"x": [1, 2]})

    def test_exact_parity_fails_closed_for_malformed_inputs(self):
        self.assertFalse(compare_token_ids([1.5], [1])["passed"])
        self.assertFalse(compare_token_ids([-1], [1])["passed"])
        self.assertFalse(compare_state_snapshots(None, None)["passed"])
        snapshot = {
            "schema_version": "target_state_v1",
            "target_model_version": "target@1",
            "state_kind": "kv",
            "sequence_position": 1,
            "token_ids": [1],
            "state": {"k": [1]},
            "metadata": {},
            "state_hash": "tampered",
        }
        self.assertFalse(compare_state_snapshots(snapshot, snapshot)["passed"])

    def test_parity_includes_target_representation_metadata(self):
        left = feature_manifest(
            {
                "target": {
                    "shape": [1, 2, 8],
                    "dtype": "bf16",
                    "target_repr": "hidden_state",
                    "target_meta": {"vocab_map_version": "v1"},
                }
            },
            schema_version=1,
            target_model_version="target@rev-a",
        )
        right = dict(left)
        right["features"] = dict(left["features"])
        right["features"]["target"] = dict(left["features"]["target"])
        right["features"]["target"]["target_meta"] = {"vocab_map_version": "v2"}
        result = compare_feature_manifests(left, right)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(item.get("field") == "target_meta" for item in result["differences"])
        )

    def test_acceptance_summary_rechecks_thresholds(self):
        summary = {
            "passed": False,
            "train": {
                "requests": 4,
                "weighted_acc_len": 2.8,
                "true_position_acceptance": [0.8, 0.4],
                "invalid_requests": [],
            },
            "holdout": {
                "requests": 20,
                "weighted_acc_len": 2.1,
                "true_position_acceptance": [0.7, 0.25],
                "invalid_requests": [],
            },
        }
        result = validate_acceptance_summary(
            summary,
            min_train_acc_len=2.5,
            min_holdout_acc_len=2.0,
            min_holdout_pos2=0.2,
            expected_train_requests=4,
            expected_holdout_requests=20,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][-1]["name"], "holdout_requests")

        failed = validate_acceptance_summary(
            summary, min_holdout_pos2=0.3, expected_holdout_requests=21
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(any("holdout pos2" in error for error in failed["errors"]))
        self.assertFalse(
            validate_acceptance_summary(summary, min_train_acc_len=float("nan"))[
                "passed"
            ]
        )
        self.assertFalse(
            validate_acceptance_summary(summary, expected_holdout_requests=1.5)[
                "passed"
            ]
        )

    def test_cli_acceptance_returns_gate_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "train": {
                            "requests": 1,
                            "weighted_acc_len": 2.5,
                            "true_position_acceptance": [0.5, 0.3],
                            "invalid_requests": [],
                        },
                        "holdout": {
                            "requests": 1,
                            "weighted_acc_len": 2.0,
                            "true_position_acceptance": [0.5, 0.2],
                            "invalid_requests": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "specforge.cli",
                    "validate",
                    "acceptance",
                    "--summary",
                    str(path),
                    "--min-train-acc-len",
                    "2.5",
                    "--min-holdout-acc-len",
                    "2.0",
                    "--min-holdout-pos2",
                    "0.2",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(json.loads(result.stdout)["passed"])

    def test_sweep_and_torchspec_plan_are_deterministic(self):
        rows = build_sweep(
            {
                "recommendations": {
                    "draft_depth_candidates": [3, 5],
                    "target_tap_candidates": [1, 2, 3, 4, 5],
                    "train_block_candidates": [7],
                    "anchor_sampling_candidates": ["uniform", "random"],
                }
            },
            max_runs=3,
        )
        self.assertEqual([row["run_index"] for row in rows], [1, 2, 3])
        self.assertTrue(
            all(row["decode_block_size"] <= row["train_block_size"] for row in rows)
        )
        self.assertEqual(TorchSpecLaunch(nodes=2, gpus_per_node=4).world_size, 8)
        manifest = TorchSpecLaunch(
            nodes=2,
            gpus_per_node=4,
            checkpoint_root="/shared/checkpoints",
            max_checkpoints=3,
        ).manifest()
        self.assertTrue(manifest["checkpoint_contract"]["shared_filesystem_required"])
        self.assertEqual(manifest["checkpoint_contract"]["rotation"], 3)

    def test_cli_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text("[1,2]\n", encoding="utf-8")
            (root / "b.json").write_text("[1,3]\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "specforge.cli",
                    "validate",
                    "tokens",
                    "--expected",
                    str(root / "a.json"),
                    "--actual",
                    str(root / "b.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["passed"])


if __name__ == "__main__":
    unittest.main()
