"""CPU contracts and real packed readout; no native/MLX compatibility assertion."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from draftfit.modeling.target.quantized_parts import load_target_parts
from draftfit.offline_capture.prepare_mlx import main, resolve, sha256, verify_inputs
from draftfit.offline_capture.prism_config import validate_prism_config
from tests import test_quantized_target_parts as packed_fixtures


class PackedReadoutBoundaries(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.config, self.weights, self.signs = packed_fixtures.QuantizedTargetPartsTest().fixture(self.path)

    def test_embedding_rejects_negative_out_of_range_and_float_ids(self):
        parts = load_target_parts(self.path, device="cpu")
        for ids in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([1.0])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                parts.embed_tokens(ids)
        self.assertEqual(parts.embed_tokens(torch.empty(2, 0, dtype=torch.long)).shape,
                         (2, 0, 512))

    def test_head_autocast_preserves_fp16_contract_and_gradient(self):
        parts = load_target_parts(self.path, device="cpu")
        x = torch.linspace(-0.3, 0.7, 512).reshape(1, -1).requires_grad_()
        expected = parts.lm_head(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = parts.lm_head(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum(), 0)

    def test_fp16_affine_overflow_rejected_for_head_and_embedding_only(self):
        for module, embedding_only in (("lm_head", False), ("model.embed_tokens", True)):
            self.weights[module + ".scales"].fill_(65504.0)
            save_file(self.weights, str(self.path / "model.safetensors"))
            with self.subTest(module=module), self.assertRaisesRegex(ValueError, "overflow|non-finite"):
                load_target_parts(self.path, embedding_only=embedding_only)

    def test_invalid_pack_schema_geometry_and_records_raise_value_error(self):
        bad = [dict(self.config, schema_version=True),
               dict(self.config, text_config=[]),
               dict(self.config, text_config=dict(self.config["text_config"], hidden_size=True)),
               dict(self.config, text_config=dict(self.config["text_config"], vocab_size=0)),
               dict(self.config, modules=[self.config["modules"][0]] * 2)]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_prism_config(payload)

    def test_normal_hf_checkpoint_keeps_original_loader_behavior(self):
        normal = self.path / "normal"
        normal.mkdir()
        (normal / "config.json").write_text(json.dumps({"model_type": "llama",
            "hidden_size": 16, "vocab_size": 4, "tie_word_embeddings": False,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "intermediate_size": 32}))
        embedding, head = torch.randn(4, 16), torch.randn(4, 16)
        save_file({"model.embed_tokens.weight": embedding, "lm_head.weight": head},
                  str(normal / "model.safetensors"))
        parts = load_target_parts(str(normal), device="cpu", dtype=torch.float32)
        torch.testing.assert_close(parts.embed_tokens.weight, embedding, atol=0, rtol=0)
        torch.testing.assert_close(parts.lm_head.weight, head, atol=0, rtol=0)
        self.assertTrue(all(not value.requires_grad for value in parts.parameters()))

    def test_real_cpu_optimizer_step_through_frozen_packed_head(self):
        torch.manual_seed(43)
        parts = load_target_parts(self.path, device="cpu", dtype=torch.float32)
        draft = torch.nn.Linear(16, 512)
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
        before = draft.weight.detach().clone()
        frozen = {k: v.clone() for k, v in parts.named_buffers()}
        hidden = draft(torch.randn(2, 3, 16))
        loss = parts.lm_head(hidden).float().square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in draft.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, draft.weight))
        self.assertEqual(list(parts.parameters()), [])
        for key, value in parts.named_buffers():
            torch.testing.assert_close(value, frozen[key], atol=0, rtol=0)
        checkpoint = self.path / "tiny-draft.pt"
        torch.save(draft.state_dict(), checkpoint)
        reloaded = torch.nn.Linear(16, 512)
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        torch.testing.assert_close(reloaded.weight, draft.weight, atol=0, rtol=0)


class MLXPreparationCPU(unittest.TestCase):
    """Actual dry-run CLI; no MLX import or forward is needed."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.target = self.root / "target"
        self.target.mkdir()
        self.config = {"model_type": "qwen3_5_text", "hidden_size": 128,
                       "vocab_size": 4, "num_hidden_layers": 4}
        (self.target / "config.json").write_text(json.dumps(self.config))
        (self.target / "tokenizer.json").write_text("{}")
        # A file identity fixture only. This is not an executable MLX model.
        save_file({"identity_fixture": torch.ones(1)}, str(self.target / "model.safetensors"))
        self.draft = {"hidden_size": 128, "vocab_size": 4,
                      "dflash_config": {"target_layer_ids": [1, 3]}}
        self.draft_path = self.root / "draft.json"
        self.draft_path.write_text(json.dumps(self.draft))
        self.header = {"kind": "provenance", "target_revision": "a" * 40,
                       "tokenizer_sha256": sha256(self.target / "tokenizer.json")}
        self.rows = [{"id": "train:0", "input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}]
        self.data = self.root / "data.jsonl"
        self.write_data()
        self.args = argparse.Namespace(target=self.target, target_revision="a" * 40,
                                       draft_config=self.draft_path, data=self.data,
                                       output=self.root / "features", max_length=8,
                                       num_samples=10, dry_run=True)

    def write_data(self):
        self.data.write_text("\n".join(json.dumps(row) for row in [self.header] + self.rows) + "\n")

    def command(self):
        return [sys.executable, "-m", "draftfit.offline_capture.prepare_mlx",
                "--target", str(self.target), "--target-revision", "a" * 40,
                "--draft-config", str(self.draft_path), "--data", str(self.data),
                "--output", str(self.args.output), "--max-length", "8", "--dry-run"]

    def test_real_dry_run_records_exact_weights_and_writes_no_features(self):
        result = subprocess.run(self.command(), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["weights_loaded"])
        self.assertFalse(self.args.output.exists())
        settings = report["manifest"]["producer"]["settings"]
        self.assertEqual(settings["weights_sha256"],
                         {"model.safetensors": sha256(self.target / "model.safetensors")})
        self.assertEqual(settings["data_sha256"], sha256(self.data))
        before = report["manifest"]["manifest_hash"]
        save_file({"identity_fixture": torch.zeros(1)}, str(self.target / "model.safetensors"))
        after, _ = resolve(self.args)
        self.assertNotEqual(before, after["manifest_hash"])

    def test_fingerprint_geometry_taps_and_dataset_errors(self):
        original = copy.deepcopy(self.header)
        for key in ("target_revision", "tokenizer_sha256"):
            self.header = dict(original, **{key: "b" * (40 if key == "target_revision" else 64)})
            self.write_data()
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "fingerprint|revision"):
                resolve(self.args)
        self.header = original
        self.write_data()
        for layers in ([3, 1], [1, 1], [1, 4], [True], [[1]]):
            self.draft["dflash_config"]["target_layer_ids"] = layers
            self.draft_path.write_text(json.dumps(self.draft))
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                resolve(self.args)
        self.draft["dflash_config"]["target_layer_ids"] = [1, 3]
        self.draft["hidden_size"] = 64
        self.draft_path.write_text(json.dumps(self.draft))
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            resolve(self.args)

    def test_manifest_recheck_rejects_changed_weights_and_dataset(self):
        manifest, _ = resolve(self.args)
        save_file({"identity_fixture": torch.zeros(1)}, str(self.target / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            verify_inputs(self.args, manifest)
        manifest, _ = resolve(self.args)
        self.data.write_text(self.data.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "data_sha256"):
            verify_inputs(self.args, manifest)

    def test_staging_failure_and_input_drift_publish_no_partial_directory(self):
        # Deliberately injected CPU producer checks persistence only, not MLX runtime behavior.
        teacher = SimpleNamespace(hidden_size=128, vocab_size=4, num_hidden_layers=4,
                                  set_capture_layers=lambda layers: None)
        def emit(*, input_ids, attention_mask, loss_mask):
            from draftfit.inference.capture import TeacherCaptureBatch
            self.data.write_text(self.data.read_text() + " ")
            return TeacherCaptureBatch(torch.zeros(1, 3, 256), torch.zeros(1, 3, 128),
                                       input_ids, attention_mask, loss_mask)
        teacher.capture = emit
        argv = self.command()[3:-1]  # same real parser, remove --dry-run
        with (patch("draftfit.offline_capture.mlx.OfflineMLXCapture.from_pretrained", return_value=teacher),
              self.assertRaisesRegex(ValueError, "fingerprint changed")):
            main(argv)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(list(self.root.glob("mlx-capture-*")), [])
        self.write_data()
        def fail(**kwargs):
            raise RuntimeError("injected capture failure")
        teacher.capture = fail
        with (patch("draftfit.offline_capture.mlx.OfflineMLXCapture.from_pretrained", return_value=teacher),
              self.assertRaisesRegex(RuntimeError, "injected capture failure")):
            main(argv)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(list(self.root.glob("mlx-capture-*")), [])

    def test_existing_output_is_preserved(self):
        self.args.output.mkdir()
        marker = self.args.output / "keep.txt"
        marker.write_text("existing data")
        result = subprocess.run(self.command(), capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing data")

    def test_empty_data_missing_weights_and_custom_code_fail_before_output(self):
        self.data.write_text("")
        with self.assertRaises(ValueError):
            resolve(self.args)
        self.write_data()
        (self.target / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "safetensors|weight"):
            resolve(self.args)
        save_file({"identity_fixture": torch.ones(1)}, str(self.target / "model.safetensors"))
        self.config["auto_map"] = {"AutoModel": "unreviewed.CustomModel"}
        (self.target / "config.json").write_text(json.dumps(self.config))
        result = subprocess.run(self.command(), capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.args.output.exists())


class MLXInputValidationCPU(unittest.TestCase):
    def test_invalid_inputs_fail_without_importing_an_mlx_runtime(self):
        from draftfit.offline_capture.mlx import OfflineMLXCapture
        # Interface fixture only; a model forward is never called.
        model = SimpleNamespace(args=SimpleNamespace(model_type="qwen3_5_text", hidden_size=128,
                                                     vocab_size=4, num_hidden_layers=2),
                                model=SimpleNamespace(layers=[object(), object()],
                                                      norm=object(), embed_tokens=object()),
                                eval=lambda: None)
        teacher = OfflineMLXCapture(model)
        for taps in ([True], [1, 0], [1, 1], [[1]], [2]):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                teacher.set_capture_layers(taps)
        teacher.set_capture_layers([1])
        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
            for ids, attn, loss in (([[4]], [[1]], [[1]]),
                                    ([[1, 2, 3]], [[1, 0, 1]], [[1, 0, 1]]),
                                    ([[1, 2]], [[1, 0]], [[1, 1]]),
                                    ([[1]], [[0]], [[0]])):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    teacher.capture(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn),
                                    loss_mask=torch.tensor(loss))


if __name__ == "__main__":
    unittest.main()
