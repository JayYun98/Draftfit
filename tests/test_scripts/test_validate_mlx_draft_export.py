import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "validate_mlx_draft_export.py"
SPEC = importlib.util.spec_from_file_location("validate_mlx_draft_export", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_tensor_file(path, names):
    header = {}
    data = bytearray()
    for name in names:
        start = len(data)
        data.extend(struct.pack("<f", 1.0))
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [start, len(data)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class TestValidateMlxDraftExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(json.dumps({
            "model_type": "qwen3", "hidden_size": 2048, "block_size": 9,
            "dflash_config": {"mask_token_id": 125017,
                              "target_layer_ids": [2, 9, 17, 21, 27]},
        }))

    def test_format_pass_is_not_inference_or_mlx_pass(self):
        write_tensor_file(self.root / "model.safetensors", validator.REQUIRED)
        result = validator.check_export(self.root, "lfm-2.6b", "dspark")
        self.assertEqual(result["format_config"], "pass")
        self.assertEqual(result["mlx_tensor_load"], "not_run")
        self.assertEqual(result["token_inference"], "not_run")

    def test_official_taichu_nested_config(self):
        (self.root / "config.json").write_text(json.dumps({
            "transformer_layer_config": {"hidden_size": 4096},
            "block_size": 8, "mask_token_id": 248077,
            "aux_hidden_state_layer_ids": [1, 8, 15, 22, 29],
        }))
        write_tensor_file(self.root / "model.safetensors", validator.REQUIRED)
        self.assertEqual(
            validator.check_export(self.root, "taichu-9b", "dspark")["format_config"],
            "pass",
        )

    def test_cli_reports_format_pass_only(self):
        write_tensor_file(self.root / "model.safetensors", validator.REQUIRED)
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root), "--target", "lfm-2.6b",
             "--strategy", "dspark"], capture_output=True, text=True, check=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["format_config"], "pass")
        self.assertEqual(result["mlx_tensor_load"], "not_run")

    @unittest.skipUnless(importlib.util.find_spec("mlx"), "MLX is not installed")
    def test_cli_mlx_load_is_distinct_from_inference(self):
        write_tensor_file(self.root / "model.safetensors", validator.REQUIRED)
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root), "--target", "lfm-2.6b",
             "--strategy", "dspark", "--mlx-load"],
            capture_output=True, text=True, check=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["mlx_tensor_load"], "pass")
        self.assertEqual(result["token_inference"], "not_run")

    def test_dflash2_requires_selector(self):
        write_tensor_file(self.root / "model.safetensors", validator.REQUIRED)
        with self.assertRaisesRegex(ValueError, "candidate_selector"):
            validator.check_export(self.root, "lfm-2.6b", "dflash2")

    def test_rejects_truncated_tensor_payload(self):
        path = self.root / "model.safetensors"
        write_tensor_file(path, validator.REQUIRED)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "byte span"):
            validator.check_export(self.root, "lfm-2.6b", "dspark")

    def test_shard_index_must_match_contents(self):
        write_tensor_file(self.root / "model-00001-of-00001.safetensors", validator.REQUIRED)
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "wrong.safetensors" for key in validator.REQUIRED}
        }))
        with self.assertRaises(FileNotFoundError):
            validator.check_export(self.root, "lfm-2.6b", "dspark")


if __name__ == "__main__":
    unittest.main()
