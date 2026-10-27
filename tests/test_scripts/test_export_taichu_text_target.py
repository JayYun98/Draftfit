import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from scripts.export_taichu_text_target import export_text_target

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/export_taichu_text_target.py"


class TestExportTaichuTextTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.output = self.root / "source", self.root / "output"
        self.source.mkdir()
        self.config = {
            "model_type": "zdtaichu5_0",
            "architectures": ["ZDTaichu5_0_ForConditionalGeneration"],
            "eos_token_id": 17,
            "llm_config": {
                "model_type": "qwen3_5_text",
                "architectures": ["Qwen3_5ForCausalLM"],
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "tie_word_embeddings": False,
                "eos_token_id": None,
                "rope_parameters": {
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [1, 1, 0],
                },
            },
        }
        self.write("config.json", self.config)
        for name, value in [
            ("tokenizer.json", {"version": "1.0"}),
            ("tokenizer_config.json", {"chat_template": "original template"}),
            ("generation_config.json", {"temperature": 0.7, "eos_token_id": None}),
        ]:
            self.write(name, value)
        (self.source / "LICENSE").write_text("Original license notice\n")
        (self.source / "THIRD_PARTY_LICENSES.md").write_text(
            "Original third-party notices\n"
        )
        (self.source / "chat_template.jinja").write_text("{{ messages }}")
        names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        names += [
            "model.layers.0." + name
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
        ]
        self.tensors = {
            name: (torch.arange(6).reshape(2, 3) + i).to(torch.bfloat16)
            for i, name in enumerate(names)
        }
        shards = [{}, {}]
        for i, (name, value) in enumerate(self.tensors.items()):
            shards[i % 2]["language_model." + name] = value
        shards[0]["vision_model.weight"] = torch.ones(2)
        shards[1]["mlp1.weight"] = torch.zeros(2)
        self.mapping = {}
        for i, tensors in enumerate(shards):
            name = f"source-{i}.safetensors"
            save_file(tensors, self.source / name)
            self.mapping.update({key: name for key in tensors})
        self.write("model.safetensors.index.json", {"weight_map": self.mapping})

    def write(self, name, data):
        (self.source / name).write_text(json.dumps(data))

    def test_real_cli_export_reload_preserves_values_and_metadata(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source",
                str(self.source),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unverified", result.stdout)
        index = json.loads((self.output / "model.safetensors.index.json").read_text())
        loaded = {}
        for name in set(index["weight_map"].values()):
            loaded.update(load_file(self.output / name))
        self.assertEqual(set(loaded), set(self.tensors))
        for key, expected in self.tensors.items():
            self.assertEqual(loaded[key].dtype, expected.dtype)
            self.assertTrue(torch.equal(loaded[key], expected))
        self.assertEqual(
            index["metadata"]["total_size"],
            sum(t.numel() * t.element_size() for t in self.tensors.values()),
        )
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(config, dict(self.config["llm_config"], eos_token_id=17))
        self.assertEqual(
            json.loads((self.output / "generation_config.json").read_text()),
            {"temperature": 0.7, "eos_token_id": 17},
        )
        for name in (
            "LICENSE",
            "THIRD_PARTY_LICENSES.md",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ):
            self.assertEqual(
                (self.output / name).read_bytes(), (self.source / name).read_bytes()
            )
        self.assertIn(
            "preserving tensor values",
            (self.output / "TEXT_EXPORT_NOTICE.md").read_text(),
        )

    def test_invalid_type_and_architecture(self):
        for field, value in [("model_type", "other"), ("architectures", ["Other"])]:
            with self.subTest(field=field):
                self.write("config.json", dict(self.config, **{field: value}))
                with self.assertRaises(ValueError):
                    export_text_target(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_invalid_inner_config(self):
        self.config["llm_config"]["model_type"] = "qwen3"
        self.write("config.json", self.config)
        with self.assertRaisesRegex(ValueError, "llm_config"):
            export_text_target(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_write_failure_removes_only_new_output(self):
        with patch(
            "scripts.export_taichu_text_target.save_file",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                export_text_target(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue((self.source / "source-0.safetensors").is_file())

    def test_missing_shard(self):
        (self.source / "source-1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Missing shard"):
            export_text_target(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_modified(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            export_text_target(self.source, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_wrong_index_assignment(self):
        key = next(iter(self.mapping))
        self.mapping[key] = "source-1.safetensors"
        self.write("model.safetensors.index.json", {"weight_map": self.mapping})
        with self.assertRaisesRegex(ValueError, "Shard/index"):
            export_text_target(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_unsafe_path_and_unknown_text_key(self):
        for mapping in [
            dict(self.mapping, **{"language_model.extra": "source-0.safetensors"}),
            {key: "../bad.safetensors" for key in self.mapping},
        ]:
            self.write("model.safetensors.index.json", {"weight_map": mapping})
            with self.assertRaises(ValueError):
                export_text_target(self.source, self.output)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
