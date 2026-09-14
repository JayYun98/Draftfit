import json
import os
import tempfile
import unittest

from specforge.inference.capture_manifest import (
    build_capture_manifest,
    load_capture_manifest,
    validate_capture_manifest,
    write_capture_manifest,
)


class TestCaptureManifest(unittest.TestCase):
    def test_backend_provenance_does_not_change_tensor_contract(self):
        options = dict(
            strategy="dspark", capture_method="dspark", capture_layers=[0, 2],
            feature_names=["hidden_states"], target_model="target", target_revision="sha",
        )
        hf = build_capture_manifest(**options, capture_backend="transformers")
        vllm = build_capture_manifest(**options, capture_backend="vllm")
        self.assertEqual(hf["contract_hash"], vllm["contract_hash"])
        self.assertNotEqual(hf["manifest_hash"], vllm["manifest_hash"])
        self.assertEqual(validate_capture_manifest(hf)["producer"]["backend"], "transformers")

    def _manifest(self):
        return build_capture_manifest(
            strategy="dspark",
            capture_method="ling",
            capture_layers=(1, 5),
            feature_names=("input_ids", "target_hidden"),
            target_model="example/ling",
            target_revision="0123456789abcdef0123456789abcdef01234567",
            target_hidden_size=128,
            target_vocab_size=32000,
            draft_vocab_size=32000,
            target_feature="target_hidden",
            target_repr="hidden_state",
            tokenizer_version="tok-1",
            max_length=128,
            draft_config={"hidden_size": 128},
        )

    def test_round_trip_is_deterministic(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture_manifest.json")
            write_capture_manifest(path, manifest)
            loaded = load_capture_manifest(path)
            self.assertEqual(loaded, manifest)
            self.assertEqual(
                json.dumps(loaded, sort_keys=True),
                json.dumps(manifest, sort_keys=True),
            )

    def test_tampering_fails_closed(self):
        manifest = self._manifest()
        manifest["contract"]["capture_layers"].append(7)
        with self.assertRaisesRegex(ValueError, "contract_hash mismatch"):
            validate_capture_manifest(manifest)

    def test_reader_constraints_are_checked(self):
        manifest = self._manifest()
        with self.assertRaisesRegex(ValueError, "strategy"):
            validate_capture_manifest(manifest, strategy="eagle3")
        with self.assertRaisesRegex(ValueError, "features"):
            validate_capture_manifest(manifest, feature_names=("input_ids",))


if __name__ == "__main__":
    unittest.main()
