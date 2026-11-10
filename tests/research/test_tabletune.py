"""Offline TableTune contracts through real CLI processes and fresh outputs."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples/configs/research/tabletune.yaml"
DATA = EXAMPLE.parent / "data"
REVISION = "38170878d901cb06f696a122a0b4140a3d365f02"
SCHEDULE = [[1, 1, 3], [2, 4, 1], [5, 8, 0]]


class TableTuneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="draftfit-tabletune-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for path in DATA.glob("tabletune-*.json"):
            shutil.copyfile(path, self.root / path.name.removeprefix("tabletune-"))
        self.recipe = yaml.safe_load(EXAMPLE.read_text())
        self.recipe["data"]["input_path"] = "acceptance.json"
        self.recipe["table"]["costs_pattern"] = "costs_k*.json"
        self.recipe["output_dir"] = "output"
        self.save_recipe()

    def save_recipe(self):
        (self.root / "recipe.yaml").write_text(yaml.safe_dump(self.recipe))

    def write_acceptance(self, accept, rate=1.0, stored=None):
        data = {"accept_by_pos": accept, "proposal_rate": rate}
        if stored is not None:
            data["tokens_per_step_by_k"] = stored
        (self.root / "acceptance.json").write_text(json.dumps(data))

    def write_cost(self, k, times, batches=(1, 4, 8)):
        data = {
            "rows": [
                {"k": k, "batch_size": b, "step_ms": ms}
                for b, ms in zip(batches, times)
            ]
        }
        (self.root / f"costs_k{k}.json").write_text(json.dumps(data))

    def cli(self, *overrides, plan=False, module=False, example=False):
        command = [
            sys.executable,
            "-m",
            "draftfit.research" if module else "draftfit.cli",
        ]
        if not module:
            command.append("research")
        command.extend(["-c", str(EXAMPLE) if example else "recipe.yaml"])
        if plan:
            command.append("--plan")
        command.extend(overrides)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            command,
            cwd=REPO if example else self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def success(self, *overrides, **kwargs):
        process = self.cli(*overrides, **kwargs)
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def failure(self, expected, *overrides, **kwargs):
        process = self.cli(*overrides, **kwargs)
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertIn(expected, process.stderr)
        self.assertNotIn("Traceback", process.stderr)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse(list(self.root.glob(".draftfit-research-*")))

    def test_plan_does_not_read_profiles_or_write_output(self):
        (self.root / "acceptance.json").unlink()
        for path in self.root.glob("costs_k*.json"):
            path.unlink()
        before = set(self.root.iterdir())
        result = self.success(plan=True)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(result["execution"], "not-run")
        self.assertEqual(result["config"]["method"], "tabletune")
        self.assertEqual(result["source_revision"], REVISION)
        self.assertEqual(result["profile_kind"], "synthetic")
        self.assertFalse(result["limits"]["cost_extrapolation"])
        self.assertEqual(result["limits"]["engine_parity"], "not-run")

    def test_main_cli_matches_hand_calculated_table_and_persists_profiles(self):
        self.success(plan=True)
        result = self.success()
        self.assertEqual(result["schedule"], SCHEDULE)
        self.assertEqual(result["raw_schedule"], SCHEDULE)
        # At B=2, interpolation gives step times 10, 13, 50/3 ms.
        # Tokens per step are 1, 1.6, 2; these fractions cover every B and K.
        expected = [
            [100, 400 / 3, 2000 / 13],
            [200, 3200 / 13, 240],
            [300, 2400 / 7, 18000 / 61],
            [400, 1280 / 3, 1000 / 3],
            [500, 32000 / 69, 1000 / 3],
            [600, 6400 / 13, 1000 / 3],
            [700, 44800 / 87, 1000 / 3],
            [800, 1600 / 3, 1000 / 3],
        ]
        goodput = result["predicted_goodput_tokens_per_second"]
        self.assertEqual(set(goodput), {str(b) for b in range(1, 9)})
        for b, row in enumerate(expected, 1):
            self.assertEqual(set(goodput[str(b)]), {"0", "1", "3"})
            for k, value in zip((0, 1, 3), row):
                self.assertAlmostEqual(goodput[str(b)][str(k)], value, places=10)
        self.assertEqual(result["tokens_per_step_by_k"], {"0": 1, "1": 1.6, "3": 2})
        self.assertEqual(result["tokens_per_step_source"], "stored_rounded")
        self.assertEqual(result["source_revision"], REVISION)
        self.assertEqual(result["source_license"], "Apache-2.0")
        self.assertEqual(result["profile_kind"], "synthetic")
        self.assertFalse(result["serving_backend"])
        self.assertEqual(result["limits"]["max_batch_size"], 8)
        for gate in ("live_controller", "engine_parity", "throughput"):
            self.assertEqual(result["limits"][gate], "not-run")
        self.assertEqual(
            result["cost_coverage_by_k"], {"0": [1, 8], "1": [1, 8], "3": [1, 8]}
        )
        self.assertEqual(
            result["profiles"]["acceptance_sha256"],
            hashlib.sha256((self.root / "acceptance.json").read_bytes()).hexdigest(),
        )
        for k, digest in result["profiles"]["costs_sha256_by_k"].items():
            self.assertEqual(
                digest,
                hashlib.sha256(
                    (self.root / f"costs_k{k}.json").read_bytes()
                ).hexdigest(),
            )
        output = self.root / "output"
        self.assertEqual(json.loads((output / "result.json").read_text()), result)
        self.assertEqual(
            json.loads((output / "recipe.json").read_text())["table"]["max_batch_size"],
            8,
        )
        self.assertEqual(
            {p.name for p in output.iterdir()}, {"recipe.json", "result.json"}
        )

    def test_module_cli_runs_the_shipped_yaml_in_a_fresh_output(self):
        output = self.root / "example-output"
        self.success(f"output_dir={output}", plan=True, module=True, example=True)
        self.assertFalse(output.exists())
        result = self.success(f"output_dir={output}", module=True, example=True)
        self.assertEqual(result["schedule"], SCHEDULE)
        self.assertEqual(json.loads((output / "result.json").read_text()), result)

    def test_profile_kind_defaults_to_synthetic_and_can_be_explicitly_measured(self):
        self.recipe.pop("profile_kind")
        self.save_recipe()
        self.assertEqual(self.success(plan=True)["profile_kind"], "synthetic")
        result = self.success("profile_kind=measured")
        self.assertEqual(result["profile_kind"], "measured")
        self.assertEqual(result["limits"]["throughput"], "not-run")

    def test_k0_is_selected_when_speculation_is_unprofitable(self):
        self.write_acceptance([0, 0, 0])
        result = self.success()
        self.assertEqual(result["schedule"], [[1, 8, 0]])
        self.assertEqual(result["raw_schedule"], [[1, 8, 0]])

    def test_proposal_rate_scales_gains_but_keeps_full_k_costs(self):
        for rate in (0.2, 0.98, 1.003):
            with self.subTest(rate=rate):
                stored = {"0": 1, "1": 1.6, "2": 1.9, "3": 2}
                self.write_acceptance([0.6, 0.3, 0.1], rate, stored)
                result = self.success(f"output_dir=rate-{rate}")
                effective = 0.2 if rate == 0.2 else 1.0
                self.assertEqual(result["proposal_rate_input"], rate)
                self.assertEqual(result["proposal_rate"], effective)
                self.assertAlmostEqual(
                    result["tokens_per_step_by_k"]["1"], 1 + 0.6 * effective
                )
                self.assertAlmostEqual(
                    result["predicted_goodput_tokens_per_second"]["1"]["3"],
                    (1 + effective) * 1000 / 13,
                )
                self.assertEqual(
                    result["schedule"], [[1, 8, 0]] if rate == 0.2 else SCHEDULE
                )
                self.assertEqual(
                    result["tokens_per_step_source"],
                    "prefix_survival" if rate == 0.2 else "stored_rounded",
                )

    def test_rounded_tokens_per_step_control_a_close_decision(self):
        (self.root / "costs_k3.json").unlink()
        self.write_cost(1, [16.002, 16.002, 16.002])
        self.write_acceptance([0.6004], stored={"0": 1, "1": 1.6})
        rounded = self.success("output_dir=rounded")
        self.assertEqual(rounded["schedule"], [[1, 8, 0]])
        self.assertEqual(rounded["tokens_per_step_by_k"]["1"], 1.6)
        self.write_acceptance([0.6004])
        recomputed = self.success("output_dir=recomputed")
        self.assertEqual(recomputed["schedule"], [[1, 8, 1]])
        self.assertEqual(recomputed["tokens_per_step_by_k"]["1"], 1.6004)

    def test_eps_holds_a_nearly_tied_previous_k(self):
        (self.root / "costs_k3.json").unlink()
        self.write_acceptance([1])
        self.write_cost(1, [19, 20.05, 20.05])
        result = self.success("output_dir=hysteresis")
        self.assertEqual(result["raw_schedule"], [[1, 3, 1], [4, 8, 0]])
        self.assertEqual(result["schedule"], [[1, 8, 1]])
        exact = self.success("output_dir=exact", "table.eps=0")
        self.assertEqual(exact["schedule"], exact["raw_schedule"])

    def test_merge_tolerance_keeps_a_profitable_narrow_range(self):
        (self.root / "costs_k3.json").unlink()
        self.write_acceptance([1])
        self.write_cost(1, [22, 22, 22, 22, 22, 18, 22, 22], range(1, 9))
        raw = [[1, 5, 0], [6, 6, 1], [7, 8, 0]]
        merged = self.success("output_dir=merged")
        self.assertEqual(merged["raw_schedule"], raw)
        self.assertEqual(merged["schedule"], [[1, 8, 0]])
        kept = self.success("output_dir=kept", "table.merge_tol=0.02")
        self.assertEqual(kept["schedule"], raw)
        unmerged = self.success("output_dir=unmerged", "table.min_width_frac=0")
        self.assertEqual(unmerged["schedule"], raw)

    def test_repeated_output_is_refused_without_mutating_saved_result(self):
        self.success()
        output = self.root / "output"
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        (self.root / "acceptance.json").write_text("not JSON")
        process = self.cli()
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertIn("output already exists", process.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)
        self.assertFalse(list(self.root.glob(".draftfit-research-*")))

    def test_strict_recipe_and_finite_parameter_validation(self):
        invalid = (
            "table.max_batch_size=0",
            "table.max_batch_size=4097",
            "table.max_batch_size=true",
            "table.max_batch_size=1.5",
            "table.eps=-0.1",
            "table.eps=1",
            "table.eps=.nan",
            "table.eps=.inf",
            "table.min_width_frac=-1",
            "table.min_width_frac=.nan",
            "table.min_width_frac=.inf",
            "table.merge_tol=-0.1",
            "table.merge_tol=1.01",
            "table.merge_tol=.nan",
            "table.merge_tol=.inf",
            "profile_kind=unknown",
            "table.unrecognized=1",
            "data.max_samples=1",
        )
        for override in invalid:
            with self.subTest(override=override):
                process = self.cli(override, plan=True)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertFalse((self.root / "output").exists())
        self.recipe["unexpected"] = True
        self.save_recipe()
        self.failure("extra_forbidden", plan=True)

    def test_missing_input_and_unmatched_cost_pattern_fail_cleanly(self):
        self.failure("no cost files match", "table.costs_pattern=missing*.json")
        self.failure("No such file", "data.input_path=missing.json")

    def test_nonmonotone_prefix_survival_is_rejected(self):
        self.write_acceptance([0.5, 0.7, 0.1])
        self.failure("non-increasing prefix survival")

    def test_every_cost_k_requires_acceptance_support(self):
        self.write_cost(4, [10, 10, 10])
        self.failure("lack acceptance support")

    def test_k0_baseline_is_required(self):
        (self.root / "costs_k0.json").unlink()
        self.failure("K=0 baseline")

    def test_profile_coverage_never_extrapolates(self):
        for batches in ((2, 4, 8), (1, 4, 7)):
            with self.subTest(batches=batches):
                self.write_cost(1, [12, 15, 24], batches)
                self.failure("extrapolation is disabled")
        self.write_cost(1, [12, 15], (9, 10))
        self.failure("no overlapping batch coverage")

    def test_duplicate_cost_profiles_are_rejected(self):
        shutil.copyfile(self.root / "costs_k1.json", self.root / "costs_k1-copy.json")
        self.failure("duplicate cost profiles for K=1")

    def test_malformed_cost_profiles_fail_without_partial_output(self):
        row = {"k": 1, "batch_size": 1, "step_ms": 12}
        invalid = [
            None,
            [],
            {},
            {"rows": None},
            {"rows": []},
            {"rows": [None]},
            {"rows": [{"k": 1, "batch_size": 1}]},
            {"rows": [row, row]},
            {"rows": [row, {**row, "k": 3, "batch_size": 4}]},
        ]
        for field, value in (
            ("k", True),
            ("k", -1),
            ("k", "1"),
            ("batch_size", True),
            ("batch_size", 0),
            ("batch_size", 1.5),
            ("step_ms", True),
            ("step_ms", 0),
            ("step_ms", -1),
            ("step_ms", float("nan")),
            ("step_ms", float("inf")),
        ):
            invalid.append({"rows": [{**row, field: value}]})
        for data in invalid:
            with self.subTest(data=data):
                (self.root / "costs_k1.json").write_text(json.dumps(data))
                self.failure("bad")
        (self.root / "costs_k1.json").write_text("not JSON")
        self.failure("bad profile")

    def test_malformed_acceptance_and_proposal_rates_fail_cleanly(self):
        invalid = [None, [], {}, {"accept_by_pos": None}, {"accept_by_pos": "0.5"}]
        for accept in ([True, 0.3, 0.1], [1.1, 0.3, 0.1], [float("nan"), 0.3, 0.1]):
            invalid.append({"accept_by_pos": accept})
        for rate in (0, -1, True, "1", float("nan"), float("inf")):
            invalid.append({"accept_by_pos": [0.6, 0.3, 0.1], "proposal_rate": rate})
        for data in invalid:
            with self.subTest(data=data):
                (self.root / "acceptance.json").write_text(json.dumps(data))
                self.failure("bad")

    def test_stored_tokens_must_be_supported_consistent_and_cover_all_costs(self):
        original = {"0": 1, "1": 1.6, "2": 1.9, "3": 2}
        invalid = [None, [], {}, {"0": 1, "1": 1.6}]
        for key, value in (
            ("0", 1.0001),
            ("1", 1.7),
            ("1", True),
            ("1", "1.6"),
            ("1", float("nan")),
            ("1", float("inf")),
            ("-1", 1),
            ("01", 1.6),
            ("4", 2),
        ):
            invalid.append({**original, key: value})
        for stored in invalid:
            with self.subTest(stored=stored):
                self.write_acceptance([0.6, 0.3, 0.1], stored=stored)
                if stored is None:
                    data = json.loads((self.root / "acceptance.json").read_text())
                    data["tokens_per_step_by_k"] = None
                    (self.root / "acceptance.json").write_text(json.dumps(data))
                self.failure("bad acceptance file")
        self.write_acceptance([0.6, 0.3, 0.1], rate=0.2, stored={**original, "3": 9})
        self.failure("rounded prefix sums")

    def test_duplicate_json_fields_are_not_silently_overwritten(self):
        (self.root / "acceptance.json").write_text(
            '{"accept_by_pos": [0.6, 0.3, 0.1], "proposal_rate": 1, "proposal_rate": 0.2}'
        )
        self.failure("duplicate JSON key")

    def test_unsorted_cost_rows_are_supported(self):
        data = json.loads((self.root / "costs_k1.json").read_text())
        data["rows"].reverse()
        (self.root / "costs_k1.json").write_text(json.dumps(data))
        self.assertEqual(self.success()["schedule"], SCHEDULE)

    def test_numerically_unrepresentable_goodput_fails_cleanly(self):
        self.write_cost(0, [5e-324, 5e-324, 5e-324])
        self.failure("invalid predicted goodput")

    def test_source_catalog_and_unmodified_vendor_bytes(self):
        process = subprocess.run(
            [sys.executable, "-m", "draftfit.cli", "research", "--sources"],
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        source = next(
            row for row in json.loads(process.stdout) if row["project"] == "tabletune"
        )
        self.assertEqual(source["revision"], REVISION)
        self.assertEqual(source["paper_revision"], "2610.38968v2")
        self.assertEqual(source["url"], "https://github.com/asm582/tabletune")
        self.assertFalse(source["production_engine_validated"])
        vendor = REPO / "draftfit/research/_vendor"
        self.assertEqual(
            hashlib.sha256((vendor / "tabletune.py").read_bytes()).hexdigest(),
            "8d4c7e0d8b5d7a04f85f00fe7c28d80f1f76afd4c497657430b412df744ee80c",
        )
        self.assertEqual(
            hashlib.sha256((vendor / "TABLETUNE_LICENSE").read_bytes()).hexdigest(),
            "c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c",
        )


if __name__ == "__main__":
    unittest.main()
