"""Recipe and persisted-output behavior, using local tiny models only."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from draftfit.research.config import ADAPTER, load_recipe
from draftfit.research.runner import plan, run
from tests.test_research_methods import draft_model, target_model

ROOT = Path(__file__).resolve().parents[1]


class ResearchCoreTest(unittest.TestCase):
    def test_alr_training_defaults_and_bounds(self):
        from draftfit.config.schema import TrainingConfig

        config = TrainingConfig()
        self.assertEqual(
            (config.alr_topk, config.alr_hard_label_mix, config.alr_tail_bucket),
            (8, 0.0, True),
        )
        for field, value in (
            ("alr_topk", 0),
            ("alr_hard_label_mix", -0.1),
            ("alr_hard_label_mix", 1.1),
            ("alr_hard_label_mix", float("nan")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                TrainingConfig.model_validate({field: value})

    def test_overrides_revalidate_and_refuse_external_runtime_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "recipe.yaml"
            recipe.write_text(
                "method: redline\ndata:\n  input_path: missing\noutput_dir: output\n"
            )
            self.assertEqual(
                load_recipe(str(recipe), ["calibration.alpha=0.2"]).calibration.alpha,
                0.2,
            )
            for override in (
                "calibration.alhpa=0.2",
                "calibration.alpha=1",
                "calibration.alpha=.nan",
                "bad",
            ):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    load_recipe(str(recipe), [override])
        for method in (
            "turbo_dllm",
            "spexis",
            "echo",
            "wavefront",
            "dscale",
            "carryover",
        ):
            with self.subTest(method=method), self.assertRaises(ValueError):
                ADAPTER.validate_python(
                    {"method": method, "backend": {"source_root": "/tmp"}}
                )
        with self.assertRaises(ValueError):
            ADAPTER.validate_python(
                {
                    "method": "redline",
                    "data": {"input_path": "x"},
                    "output_dir": "y",
                    "backend": {"source_root": "z"},
                }
            )

    def test_redline_recipe_persists_lazy_kernel_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "calibration.json"
            data.write_text(
                json.dumps(
                    {
                        "correct": [[1] * 160, [1] * 160],
                        "cost": [1.0, 2.0],
                        "names": ["reference", "fast"],
                    }
                )
            )
            config = ADAPTER.validate_python(
                {
                    "method": "redline",
                    "data": {"input_path": str(data)},
                    "output_dir": str(root / "out"),
                }
            )
            result = run(config)
            self.assertEqual(result["deployed"], "fast")
            self.assertEqual(result["reference"], "reference")
            self.assertEqual(result["n"], 160)
            self.assertEqual(json.loads((root / "out/result.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                run(config)

    def test_invalid_osprey_taps_rejected_before_loading(self):
        for taps in ([0, 1], [0, 0, 1], [-1, 1, 2], [2, 1, 0]):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                ADAPTER.validate_python(
                    {
                        "method": "osprey",
                        "model": {
                            "source_checkpoint": "missing",
                            "target_hidden_size": 16,
                            "target_layer_ids": taps,
                        },
                        "output_dir": "unused",
                    }
                )

    def test_plan_is_read_only_and_catalog_import_is_light(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out"
            config = ADAPTER.validate_python(
                {
                    "method": "cast",
                    "model": {
                        "target_model_path": "missing",
                        "draft_checkpoint_path": "missing",
                    },
                    "data": {"input_path": "missing"},
                    "output_dir": str(path),
                }
            )
            self.assertEqual(plan(config)["execution"], "not-run")
            self.assertFalse(path.exists())
        code = (
            "import sys; from draftfit.research.catalog import source_catalog; "
            "assert {'alr_ira','osprey','cast','redline','ngramrl','specmem','treewy','arr'} <= {r['project'] for r in source_catalog()}; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_target_recipe_prepares_cache_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_model().save_pretrained(root / "target")
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}) + "\n"
            )
            config = ADAPTER.validate_python(
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
            )
            result = run(config)
            self.assertGreater(result["rollouts"], 0)
            manifest = json.loads((root / "features/capture_manifest.json").read_text())
            self.assertEqual(manifest["contract"]["strategy"], "alr_ira")
            before = (root / "features/capture_manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(config)
            self.assertEqual(
                (root / "features/capture_manifest.json").read_bytes(), before
            )

    def test_bad_rollout_mask_leaves_no_partial_destination(self):
        from draftfit.algorithms.alr_ira.prepare import prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2], "loss_mask": [0, 2]}) + "\n"
            )
            config = ADAPTER.validate_python(
                {
                    "method": "alr_ira",
                    "model": {"target_model_path": "tiny"},
                    "data": {"input_path": str(corpus)},
                    "training": {"target_layer_ids": [0, 2], "block_size": 4},
                    "output_dir": str(root / "out"),
                }
            )
            with self.assertRaisesRegex(ValueError, "binary loss mask"):
                prepare(config, target=target_model())
            self.assertFalse((root / "out").exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["corpus.jsonl"]
            )

    def test_osprey_recipe_commits_only_complete_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_model().save_pretrained(root / "source")
            config = ADAPTER.validate_python(
                {
                    "method": "osprey",
                    "model": {
                        "source_checkpoint": str(root / "source"),
                        "num_layers": 3,
                        "target_hidden_size": 16,
                        "target_layer_ids": [0, 1, 2],
                    },
                    "output_dir": str(root / "draft"),
                }
            )
            result = run(config)
            self.assertEqual(result["architecture"], "OspreyQwen3DraftModel")
            self.assertTrue((root / "draft/model.safetensors").is_file())
            with self.assertRaises(FileExistsError):
                run(config)
            bad = config.model_copy(
                update={
                    "output_dir": str(root / "bad"),
                    "model": config.model.model_copy(update={"num_layers": 4}),
                }
            )
            with self.assertRaises(ValueError):
                run(bad)
            self.assertFalse((root / "bad").exists())

    def test_cast_cli_uses_saved_models_and_honors_stop_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = target_model()
            target.save_pretrained(root / "target")
            draft_model(4).save_pretrained(root / "draft")
            prompt = torch.tensor([[1, 2, 3]])
            expected = target.generate(
                prompt,
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )[0, 3:].tolist()
            data = root / "input.jsonl"
            data.write_text(json.dumps({"input_ids": prompt[0].tolist()}) + "\n")
            recipe = root / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "method": "cast",
                        "model": {
                            "target_model_path": str(root / "target"),
                            "draft_checkpoint_path": str(root / "draft"),
                        },
                        "data": {"input_path": str(data)},
                        "generation": {"max_new_tokens": 5},
                        "speculation": {"budget": 7, "branch_k": 2},
                        "output_dir": str(root / "out"),
                    }
                )
            )
            process = subprocess.run(
                [sys.executable, "-m", "draftfit.research", "-c", str(recipe)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            output = json.loads((root / "out/result.json").read_text())
            self.assertEqual(output["answers"][0]["token_ids"], expected)
            before = (root / "out/result.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run(load_recipe(str(recipe)))
            self.assertEqual((root / "out/result.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
