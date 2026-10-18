"""Small metadata-only checks for the Taichu MLX target view."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import prepare_taichu_mlx_target as helper


class TaichuTargetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.official = (
            root
            / "models--TaichuAI--ZDTaichu5.0-9B"
            / "snapshots"
            / helper.OFFICIAL_REV
        )
        self.quant = (
            root
            / "models--abenzerps--ZDTaichu5.0-9B-MLX-4bit"
            / "snapshots"
            / helper.QUANT_REV
        )
        self.official.mkdir(parents=True)
        self.quant.mkdir(parents=True)
        names = list(helper.SHARDS)
        text = {f"language_model.model.layers.{i}.weight": names[0] for i in range(426)}
        text["language_model.lm_head.weight"] = names[4]
        self.official_index = {"weight_map": text}
        quant_text = dict(text)
        for i in range(249):
            quant_text[f"language_model.model.layers.{i}.scales"] = names[0]
            quant_text[f"language_model.model.layers.{i}.biases"] = names[0]
        quant_text["language_model.lm_head.scales"] = names[4]
        quant_text["language_model.lm_head.biases"] = names[4]
        for i in range(1, 5):
            quant_text[f"vision_model.radio_model.{i}.weight"] = names[i]
        self.quant_index = {"weight_map": quant_text}
        self.base = {
            "model_type": "zdtaichu5_0",
            "architectures": ["ZDTaichu5_0_ForConditionalGeneration"],
            "llm_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 32,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            },
            "eos_token_id": 248046,
            "pad_token_id": 248044,
        }
        self.base["llm_config"]["layer_types"] *= 8
        self.quant_cfg = dict(
            self.base,
            quantization={"group_size": 64, "bits": 4, "mode": "affine"},
            quantization_config={"group_size": 64, "bits": 4, "mode": "affine"},
        )

    def test_rejects_config_drift_and_missing_text_key(self):
        self.assertEqual(
            helper.validate_metadata(
                self.base, self.quant_cfg, self.official_index, self.quant_index
            )["model_type"],
            "qwen3_5",
        )
        bad = json.loads(json.dumps(self.quant_cfg))
        bad["llm_config"]["num_hidden_layers"] = 31
        with self.assertRaisesRegex(ValueError, "differs"):
            helper.validate_metadata(
                self.base, bad, self.official_index, self.quant_index
            )
        missing = json.loads(json.dumps(self.quant_index))
        del missing["weight_map"]["language_model.lm_head.weight"]
        with self.assertRaisesRegex(ValueError, "text weight names"):
            helper.validate_metadata(
                self.base, self.quant_cfg, self.official_index, missing
            )

    def test_symlink_view_requires_all_shards_and_original_tokenizer(self):
        official_hashes = {}
        quant_hashes = {}
        for root, cfg, index, hashes in (
            (self.official, self.base, self.official_index, official_hashes),
            (self.quant, self.quant_cfg, self.quant_index, quant_hashes),
        ):
            for name, content in (
                ("config.json", cfg),
                ("model.safetensors.index.json", index),
            ):
                data = json.dumps(content).encode()
                (root / name).write_bytes(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            data = name.encode()
            (self.official / name).write_bytes(data)
            official_hashes[name] = hashlib.sha256(data).hexdigest()
        shards = {}
        for i in range(1, 6):
            name = f"model-{i:05d}-of-00005.safetensors"
            data = f"fake shard {i}".encode()
            (self.quant / name).write_bytes(data)
            shards[name] = (len(data), hashlib.sha256(data).hexdigest())
        output = Path(self.tmp.name) / "view"
        with (
            patch.object(helper, "OFFICIAL_HASHES", official_hashes),
            patch.object(helper, "QUANT_HASHES", quant_hashes),
            patch.object(helper, "SHARDS", shards),
        ):
            helper.prepare(self.official, self.quant, output)
        self.assertTrue((output / "model-00005-of-00005.safetensors").is_symlink())
        self.assertEqual(
            (output / "tokenizer.json").resolve(),
            (self.official / "tokenizer.json").resolve(),
        )
        self.assertEqual(
            json.loads((output / "config.json").read_text())["model_file"], "model.py"
        )


if __name__ == "__main__":
    unittest.main()
