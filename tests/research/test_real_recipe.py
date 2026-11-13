"""REAL recipe boundaries and real CPU execution through the public runner."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from pydantic import ValidationError

from draftfit.research.catalog import source_catalog
from draftfit.research.config import ADAPTER, load_recipe
from draftfit.research.real import run_real_loss
from draftfit.research.runner import plan, run

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/configs/research/real-loss.yaml"
FIXTURE = ROOT / "examples/configs/research/data/real-loss.json"


def config_for(input_path, output_dir):
    return load_recipe(
        str(RECIPE), [f"data.input_path={input_path}", f"output_dir={output_dir}"]
    )


class RealRecipeTests(unittest.TestCase):
    def cli(self, *overrides, main_cli=False, planning=False):
        command = [sys.executable, "-m"]
        command += ["draftfit.cli", "research"] if main_cli else ["draftfit.research"]
        command += ["-c", str(RECIPE)]
        if planning:
            command.append("--plan")
        return subprocess.run(
            command + list(map(str, overrides)),
            cwd=ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "WANDB_MODE": "disabled",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )

    def assert_fixture_answer(self, result):
        answer = result["answers"][0]
        self.assertAlmostEqual(answer["mean_acceptance_estimate"], 0.96, places=12)
        self.assertAlmostEqual(answer["real_loss"], 0.52, places=12)
        self.assertAlmostEqual(answer["ce_loss"], math.log(10) / 2, places=12)
        self.assertAlmostEqual(
            answer["loss"], 0.9 * 0.52 + 0.1 * math.log(10) / 2, places=12
        )
        self.assertEqual(answer["active_positions"], 100)
        self.assertEqual(answer["active_blocks"], 50)
        self.assertEqual(answer["gamma"], 2)
        self.assertTrue(math.isfinite(answer["draft_gradient_norm"]))
        self.assertGreater(answer["draft_gradient_norm"], 0)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["dtype"], "float64")
        self.assertEqual(result["ce_target"], "target_logits_argmax")
        self.assertFalse(result["ce_position_decay"])
        self.assertFalse(result["author_wrapper_parity"])
        self.assertFalse(result["sampling_provenance_verified"])

    def test_plan_does_not_read_missing_input_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = config_for(directory / "missing.json", directory / "not-created")
            planned = plan(config)
            self.assertEqual(planned["execution"], "not-run")
            self.assertEqual(
                planned["scope"], "target_sampled_real_branch_component_only"
            )
            self.assertEqual(planned["ce_target"], "target_logits_argmax")
            self.assertFalse(planned["sampling_provenance_verified"])
            self.assertEqual(planned["model_training"], "not_implemented")
            self.assertEqual(planned["gpu_parity"], "not-run")
            for main_cli in (False, True):
                result = self.cli(
                    f"data.input_path={directory / 'missing.json'}",
                    f"output_dir={directory / 'not-created'}",
                    main_cli=main_cli,
                    planning=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stdout), planned)
            self.assertEqual(list(directory.iterdir()), [])

    def test_runner_executes_fixture_and_persists_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            config = config_for(FIXTURE, output)
            with torch.no_grad():
                result = run(config)
            self.assert_fixture_answer(result)
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertEqual(
                json.loads((output / "recipe.json").read_text()), config.model_dump()
            )
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(set(before), {"recipe.json", "result.json"})
            with self.assertRaises(FileExistsError):
                run(config)
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in output.iterdir()}
            )
            self.assertEqual(
                result["provenance"]["input_sha256"],
                hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            )

    def test_main_cli_executes_real_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            result = self.cli(f"output_dir={output}", main_cli=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            parsed = json.loads(result.stdout)
            self.assert_fixture_answer(parsed)
            self.assertEqual(parsed, json.loads((output / "result.json").read_text()))

    def test_fixture_contains_exact_replicated_target_paths(self):
        row = json.loads(FIXTURE.read_text())
        self.assertEqual(
            Counter(tuple(ids) for ids in row["target_token_ids"]),
            {(0, 0): 18, (0, 1): 2, (1, 0): 3, (1, 1): 27},
        )
        result = run_real_loss(config_for(FIXTURE, "unused"))
        self.assert_fixture_answer(result)
        self.assertEqual(len(result["answers"][0]["acceptance_estimate"]), 50)
        for draft, target, ids in zip(
            row["draft_logits"], row["target_logits"], row["target_token_ids"]
        ):
            q1 = [math.exp(x) for x in draft[0]]
            self.assertAlmostEqual(q1[0], 0.8, places=12)
            self.assertAlmostEqual(q1[1], 0.2, places=12)
            self.assertEqual(draft[1], [0, 0])
            expected_target = (0.9, 0.1) if ids[0] == 0 else (0.1, 0.9)
            for actual, expected in zip(target[1], expected_target):
                self.assertAlmostEqual(math.exp(actual), expected, places=12)

    def test_full_input_hash_includes_unused_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "input.json"
            outputs = []
            for tail in (b'{"unused":1}\n', b"not parsed after max_samples\n"):
                data = FIXTURE.read_bytes() + tail
                path.write_bytes(data)
                config = config_for(path, directory / "unused")
                self.assertEqual(config.data.max_samples, 1)
                result = run_real_loss(config)
                self.assertEqual(
                    result["provenance"]["input_sha256"],
                    hashlib.sha256(data).hexdigest(),
                )
                outputs.append(result)
            self.assertEqual(outputs[0]["answers"], outputs[1]["answers"])
            self.assertNotEqual(
                outputs[0]["provenance"]["input_sha256"],
                outputs[1]["provenance"]["input_sha256"],
            )

    def test_raw_json_types_are_not_silently_coerced(self):
        baseline = {
            "draft_logits": [[[0.0, 1.0], [1.0, 0.0]]],
            "target_logits": [[[1.0, 0.0], [0.0, 1.0]]],
            "target_token_ids": [[0, 1]],
            "prediction_mask": [[True, False]],
        }
        cases = []
        for field, values in (
            ("target_token_ids", (1.0, True, "1", None, -1, 2)),
            ("prediction_mask", (1, 0.0, "true", None)),
            ("draft_logits", (True, "1", None, float("nan"), float("inf"))),
            ("target_logits", (False, "0", None, -float("inf"))),
        ):
            for value in values:
                row = copy.deepcopy(baseline)
                if field.endswith("logits"):
                    row[field][0][1][0] = value
                else:
                    row[field][0][1] = value
                cases.append(row)
        cases += [
            [],
            {**baseline, "prediction_mask": None},
            {
                key: value
                for key, value in baseline.items()
                if key != "target_token_ids"
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "invalid.json"
            for index, row in enumerate(cases):
                with self.subTest(case=index):
                    path.write_text(json.dumps(row) + "\n")
                    output = directory / f"result-{index}"
                    with self.assertRaisesRegex(ValueError, "line 1"):
                        run(config_for(path, output))
                    self.assertFalse(output.exists())
                    self.assertEqual(list(directory.glob(".draftfit-research-*")), [])

    def test_empty_and_later_invalid_input_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, output = directory / "input.json", directory / "parent/result"
            for data in (b"", FIXTURE.read_bytes() + b"[]\n"):
                path.write_bytes(data)
                config = config_for(path, output)
                config.data.max_samples = 2
                with self.assertRaises(ValueError):
                    run(config)
                self.assertFalse(output.parent.exists())
                self.assertEqual(list(directory.glob(".draftfit-research-*")), [])

    def test_strict_recipe_and_overrides(self):
        baseline = load_recipe(str(RECIPE)).model_dump()
        for value in (-0.1, 1.1, True, False, "0.9", None, float("nan"), float("inf")):
            with self.subTest(weight=value), self.assertRaises(ValidationError):
                ADAPTER.validate_python({**baseline, "loss": {"real_weight": value}})
        for change in (
            {"sampling": "draft"},
            {"runtime": {"device": "cuda"}},
            {"input_kind": "measured"},
            {"loss": {"real_weight": 0.9, "gamma": 2}},
            {"loss": {"ce_position_decay": True}},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ADAPTER.validate_python({**baseline, **change})
        for weight in (0, 0.25, 1):
            config = load_recipe(
                str(RECIPE), [f"loss.real_weight={weight}", "input_kind=recorded"]
            )
            self.assertEqual(config.loss.real_weight, weight)
            self.assertEqual(config.input_kind, "recorded")
        with self.assertRaises(ValueError):
            load_recipe(str(RECIPE), ["loss.unknown=1"])

    def test_catalog_pins_author_source_without_production_claim(self):
        entry = next(
            item for item in source_catalog() if item["project"] == "real_loss"
        )
        self.assertEqual(entry["revision"], "3dff4579c5f93f692108df8b5178c9e9200b8536")
        self.assertEqual(entry["url"], "https://github.com/King-ty/speculators")
        self.assertEqual(entry["paper_revision"], "2611.14399v1")
        self.assertFalse(entry["production_engine_validated"])
        result = run_real_loss(config_for(FIXTURE, "unused"))
        self.assertEqual(result["provenance"]["source_revision"], entry["revision"])
        self.assertEqual(result["provenance"]["license"], "Apache-2.0")


if __name__ == "__main__":
    unittest.main()
