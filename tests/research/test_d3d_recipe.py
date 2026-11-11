import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from draftfit.research.config import ADAPTER, load_recipe

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/configs/research/d3d-corrected-kl.yaml"
FIXTURE = ROOT / "examples/configs/research/data/d3d-corrected-kl.json"


class D3DRecipeTests(unittest.TestCase):
    def cli(self, *overrides, main_cli=False, plan=False):
        command = [sys.executable, "-m"]
        command += ["draftfit.cli", "research"] if main_cli else ["draftfit.research"]
        command += ["-c", str(RECIPE)]
        if plan:
            command.append("--plan")
        command.extend(map(str, overrides))
        return subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_plan_never_opens_missing_input_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "not-created"
            result = self.cli(
                f"data.input_path={directory / 'missing.json'}",
                f"output_dir={output}",
                plan=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            planned = json.loads(result.stdout)
            self.assertEqual(planned["execution"], "not-run")
            self.assertEqual(planned["scope"], "corrected_forward_kl_component_only")
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.iterdir()), [])

    def test_cli_persists_analytic_result_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            result = self.cli(f"output_dir={output}", main_cli=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            persisted = json.loads((output / "result.json").read_text())
            self.assertEqual(json.loads(result.stdout), persisted)
            answer = persisted["answers"][0]
            for actual, expected in zip(
                answer["corrected_teacher_probs"][0], [0.46, 0.36, 0.18]
            ):
                self.assertAlmostEqual(actual, expected, places=12)
            first = sum(p * math.log(p * 3) for p in [0.46, 0.36, 0.18])
            second = sum(
                p * math.log(p / q) for p, q in zip([0.7, 0.2, 0.1], [0.5, 0.25, 0.25])
            )
            self.assertAlmostEqual(
                answer["kl_loss"], (first + 0.5 * second) / 1.5, places=12
            )
            self.assertEqual(answer["active_positions"], 2)
            self.assertGreater(answer["student_gradient_norm"], 0)
            self.assertEqual(persisted["input_kind"], "synthetic")
            self.assertEqual(
                json.loads((output / "recipe.json").read_text())["method"],
                "d3d_corrected_kl",
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"recipe.json", "result.json"},
            )
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            repeat = self.cli(f"output_dir={output}")
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("output already exists", repeat.stderr)
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in output.iterdir()}
            )

    def test_bad_input_cannot_publish_an_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bad_input = directory / "invalid.json"
            row = json.loads(FIXTURE.read_text())
            row["labels"][0] = 9
            bad_input.write_text(json.dumps(row) + "\n")
            output = directory / "result"
            result = self.cli(f"data.input_path={bad_input}", f"output_dir={output}")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob(".draftfit-research-*")), [])

    def test_strict_recipe_rejects_invalid_parameters(self):
        baseline = load_recipe(str(RECIPE)).model_dump()
        for field, values in (
            ("temperature", [0, -1, float("nan"), float("inf"), True, "1"]),
            ("probability_margin", [0, -1, 1, float("inf"), float("nan")]),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    raw = {**baseline, "loss": {**baseline["loss"], field: value}}
                    with self.assertRaises(ValidationError):
                        ADAPTER.validate_python(raw)
        for extra in (
            {"runtime": {"device": "cuda"}},
            {"input_kind": "measured"},
            {"loss": {**baseline["loss"], "automatic_pruning": True}},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                ADAPTER.validate_python({**baseline, **extra})

    def test_positional_recipe_overrides_use_the_public_schema(self):
        config = load_recipe(
            str(RECIPE),
            ["loss.temperature=2.0", "loss.probability_margin=0.2"],
        )
        self.assertEqual(config.loss.temperature, 2)
        self.assertEqual(config.loss.probability_margin, 0.2)
        with self.assertRaises(ValueError):
            load_recipe(str(RECIPE), ["loss.unknown=1"])


if __name__ == "__main__":
    unittest.main()
