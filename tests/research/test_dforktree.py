import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import UserList
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from draftfit.assets import export_assets
from draftfit.research.catalog import source_catalog
from draftfit.research.config import (
    ADAPTER,
    DForkTreeScoringConfig,
    InputData,
    load_recipe,
)
from draftfit.research.dforktree import run_dforktree_scoring, score_candidates
from draftfit.research.runner import plan, run

ROOT = Path(__file__).resolve().parents[2]
RECIPE = "examples/configs/research/dforktree-scoring.yaml"
FIXTURE = "examples/configs/research/data/dforktree-scoring.json"
EXPECTED = (
    ("first", 0.0, (0.5, 0.3), 0.2),
    ("spine", 0.3, (0.41633919763052696, 0.37863133148174183), 0.20502947088773132),
    ("fork", 0.6, (0.33503928046719367, 0.4618308843984667), 0.20312983513433963),
)


def evidence(**changes):
    values = {
        "candidate_ids": [101, 102],
        "marginals": [0.5, 0.3],
        "conditional": [0.2, 0.4],
        "unigram": [0.4, 0.2],
        "position": 2,
        "parent_path": "spine",
    }
    values.update(changes)
    return values


def row(**changes):
    values = {"id": "example", **evidence()}
    values.update(changes)
    return values


def recipe(root, **changes):
    values = {
        "method": "dforktree_scoring",
        "input_kind": "synthetic",
        "data": {"input_path": str(root / "input.json")},
        "output_dir": str(root / "published" / "result"),
    }
    values.update(changes)
    return ADAPTER.validate_python(values)


class DForkTreeNumericalTests(unittest.TestCase):
    def assert_distribution(self, result):
        values = result.candidate_probs + (result.tail_probability,)
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in values))
        self.assertAlmostEqual(math.fsum(values), 1.0, places=15)

    def test_analytic_first_spine_and_fork_with_tail_and_ranking(self):
        for name, weight, expected, tail in EXPECTED:
            with self.subTest(context=name):
                result = score_candidates(
                    **evidence(
                        position=1 if name == "first" else 2,
                        parent_path="forked" if name == "fork" else "spine",
                    )
                )
                self.assertEqual(result.candidate_ids, (101, 102))
                self.assertEqual(result.weight, weight)
                for actual, value in zip(result.candidate_probs, expected):
                    self.assertAlmostEqual(actual, value, places=13)
                self.assertAlmostEqual(result.tail_probability, tail, places=13)
                self.assertAlmostEqual(result.clipped_pmi[0], -math.log(2), places=15)
                self.assertAlmostEqual(result.clipped_pmi[1], math.log(2), places=15)
                ranking = sorted(
                    result.candidate_ids,
                    key=dict(zip(result.candidate_ids, result.candidate_probs)).get,
                    reverse=True,
                )
                self.assertEqual(ranking, [102, 101] if name == "fork" else [101, 102])
                self.assertLess(math.fsum(result.candidate_probs), 1)
                self.assert_distribution(result)

    def test_first_position_and_zero_weights_are_exact_identities(self):
        for context in ("spine", "forked"):
            for changes in ({"position": 1}, {"spine_weight": 0, "fork_weight": 0}):
                with self.subTest(context=context, changes=changes):
                    result = score_candidates(
                        **evidence(parent_path=context, **changes)
                    )
                    self.assertEqual(result.candidate_probs, (0.5, 0.3))
                    self.assertEqual(result.tail_probability, 1 - math.fsum([0.5, 0.3]))
                    self.assertEqual(result.weight, 0)

    def test_zero_clip_and_neutral_evidence_are_exact_identities(self):
        for changes in ({"pmi_clip": 0}, {"conditional": [0.4, 0.2]}):
            with self.subTest(changes=changes):
                result = score_candidates(**evidence(**changes))
                self.assertEqual(result.clipped_pmi, (0.0, 0.0))
                self.assertEqual(result.candidate_probs, (0.5, 0.3))
                self.assert_distribution(result)

    def test_zero_marginal_support_and_all_zero_candidates(self):
        result = score_candidates(**evidence(marginals=[0, 0.25], spine_weight=1))
        self.assertEqual(result.candidate_probs[0], 0.0)
        self.assertAlmostEqual(result.candidate_probs[1], 0.4, places=15)
        self.assertAlmostEqual(result.tail_probability, 0.6)
        for position in (1, 2):
            result = score_candidates(**evidence(marginals=[0, 0], position=position))
            self.assertEqual(result.candidate_probs, (0.0, 0.0))
            self.assertEqual(result.tail_probability, 1.0)

    def test_full_support_preserves_a_true_zero_tail(self):
        for q in ([0.1, 0.9], [0.5, 0.5]):
            with self.subTest(marginals=q):
                result = score_candidates(**evidence(marginals=q))
                self.assertEqual(result.tail_probability, 0)
                self.assert_distribution(result)
        result = score_candidates(
            range(10),
            marginals=[0.1] * 10,
            conditional=[0.1] * 10,
            unigram=[0.1] * 10,
            position=2,
            parent_path="spine",
        )
        self.assertEqual(result.tail_probability, 0)
        self.assertEqual(result.candidate_probs, (0.1,) * 10)

    def test_clipping_both_signs_matches_rational_probabilities(self):
        result = score_candidates(
            **evidence(
                marginals=[0.25, 0.25],
                conditional=[0.5, 0.25],
                unigram=[0.125, 0.5],
                spine_weight=1,
                pmi_clip=math.log(2),
            )
        )
        self.assertEqual(result.clipped_pmi, (math.log(2), -math.log(2)))
        self.assertAlmostEqual(result.candidate_probs[0], 4 / 9, places=15)
        self.assertAlmostEqual(result.candidate_probs[1], 1 / 9, places=15)
        self.assertAlmostEqual(result.tail_probability, 4 / 9, places=15)
        self.assert_distribution(result)

    def test_sequence_snapshot_preserves_identity_order_and_is_frozen(self):
        ids, q = UserList([9, 3]), UserList([0.5, 0.3])
        result = score_candidates(**evidence(candidate_ids=ids, marginals=q))
        ids[0], q[0] = 27, 0
        self.assertEqual(result.candidate_ids, (9, 3))
        self.assertAlmostEqual(result.candidate_probs[0], EXPECTED[1][2][0], places=13)
        with self.assertRaises(FrozenInstanceError):
            result.weight = 4

    def test_equal_pmi_keeps_unequal_masses_at_largest_finite_weight(self):
        for weight, expected, tail in (
            (1, (4 / 15, 2 / 5), 1 / 3),
            (sys.float_info.max, (0.4, 0.6), 0),
        ):
            with self.subTest(weight=weight):
                result = score_candidates(
                    **evidence(
                        marginals=[0.2, 0.3],
                        conditional=[0.25, 0.25],
                        unigram=[0.125, 0.125],
                        spine_weight=weight,
                    )
                )
                for actual, value in zip(result.candidate_probs, expected):
                    self.assertAlmostEqual(actual, value, places=15)
                self.assertAlmostEqual(result.tail_probability, tail, places=15)
                self.assert_distribution(result)

    def test_equal_negative_pmi_with_no_tail_keeps_unequal_masses(self):
        result = score_candidates(
            **evidence(
                marginals=[0.1, 0.9],
                conditional=[0.125, 0.125],
                unigram=[0.25, 0.25],
                spine_weight=sys.float_info.max,
            )
        )
        self.assertAlmostEqual(result.candidate_probs[0], 0.1, places=15)
        self.assertAlmostEqual(result.candidate_probs[1], 0.9, places=15)
        self.assertEqual(result.tail_probability, 0)
        self.assert_distribution(result)

    def test_scale_equivalent_evidence_keeps_equal_corrections(self):
        tiny = math.ulp(0.0)
        for conditional, unigram in (
            ([0.5, 0.25], [0.25, 0.125]),
            ([0.375, 0.28125], [0.125, 0.09375]),
            ([0.5, 0.25], [2 * tiny, tiny]),
            ([2 * tiny, tiny], [0.5, 0.25]),
            ([2 * tiny, 4 * tiny], [5 * tiny, 10 * tiny]),
        ):
            for weight in (1e16, sys.float_info.max):
                with self.subTest(conditional=conditional, weight=weight):
                    result = score_candidates(
                        **evidence(
                            marginals=[0.25, 0.75],
                            conditional=conditional,
                            unigram=unigram,
                            spine_weight=weight,
                            pmi_clip=sys.float_info.max,
                        )
                    )
                    self.assertEqual(result.clipped_pmi[0], result.clipped_pmi[1])
                    self.assertAlmostEqual(result.candidate_probs[0], 0.25, places=15)
                    self.assertAlmostEqual(result.candidate_probs[1], 0.75, places=15)
                    self.assertEqual(result.tail_probability, 0)

    def test_near_neutral_evidence_matches_high_precision_reference(self):
        for base in (1e-100, 0.5):
            with self.subTest(base=base), localcontext() as context:
                context.prec = 110
                conditional = math.nextafter(base, math.inf)
                correction = (
                    Decimal.from_float(conditional) / Decimal.from_float(base)
                ).ln() * Decimal("1e16")
                corrected = Decimal.from_float(0.2) * correction.exp()
                other = Decimal.from_float(0.3)
                tail = Decimal(1) - Decimal.from_float(0.2) - other
                total = corrected + other + tail
                result = score_candidates(
                    **evidence(
                        marginals=[0.2, 0.3],
                        conditional=[conditional, 0.25],
                        unigram=[base, 0.25],
                        spine_weight=1e16,
                    )
                )
                self.assertGreater(result.clipped_pmi[0], 0)
                self.assertAlmostEqual(
                    result.candidate_probs[0], float(corrected / total), places=14
                )
                self.assertAlmostEqual(
                    result.candidate_probs[1], float(other / total), places=14
                )
                self.assertAlmostEqual(
                    result.tail_probability, float(tail / total), places=14
                )
                self.assert_distribution(result)

    def test_extreme_finite_evidence_and_weights_stay_normalized(self):
        tiny, huge = math.ulp(0.0), sys.float_info.max
        cases = (
            ({"conditional": [tiny, 0.5], "unigram": [0.5, tiny]}, (0.0, 1.0), 0),
            (
                {
                    "marginals": [1, 0],
                    "conditional": [0.125, 0.5],
                    "unigram": [0.5, 0.125],
                },
                (1.0, 0.0),
                0,
            ),
            (
                {
                    "marginals": [0, 0.25],
                    "conditional": [0.5, 0.125],
                    "unigram": [0.125, 0.5],
                },
                (0.0, 0.0),
                1,
            ),
            (
                {
                    "marginals": [tiny, 0],
                    "conditional": [0.5, 0.125],
                    "unigram": [tiny, 0.5],
                },
                (1.0, 0.0),
                0,
            ),
        )
        for changes, expected, tail in cases:
            with self.subTest(changes=changes):
                result = score_candidates(
                    **evidence(
                        spine_weight=huge,
                        pmi_clip=huge,
                        **changes,
                    )
                )
                self.assertEqual(result.candidate_probs, expected)
                self.assertEqual(result.tail_probability, tail)
                self.assert_distribution(result)

    def test_smallest_positive_evidence_does_not_overflow_a_ratio(self):
        tiny = math.ulp(0.0)
        result = score_candidates(
            **evidence(
                marginals=[0.25, 0.25],
                conditional=[0.5, tiny],
                unigram=[tiny, 0.5],
                spine_weight=1,
                pmi_clip=sys.float_info.max,
            )
        )
        self.assertEqual(result.candidate_probs, (1.0, 0.0))
        self.assertEqual(result.tail_probability, 4 * tiny)
        self.assert_distribution(result)

    def test_rejects_nonsequences_and_nested_values(self):
        for field in ("candidate_ids", "marginals", "conditional", "unigram"):
            for value in (
                None,
                "12",
                b"12",
                bytearray(b"12"),
                memoryview(b"12"),
                {0: 0.2},
                {0.2, 0.4},
                iter([0.2, 0.4]),
                [[0.2], [0.4]],
            ):
                with self.subTest(field=field, value=type(value).__name__):
                    with self.assertRaises(ValueError):
                        score_candidates(**evidence(**{field: value}))

    def test_rejects_empty_misaligned_duplicate_and_noninteger_ids(self):
        class IntegerSubclass(int):
            pass

        for ids in (
            [],
            [101],
            [101, 101],
            [-1, 102],
            [True, 102],
            [101.0, 102],
            ["101", 102],
            [IntegerSubclass(101), 102],
        ):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    score_candidates(**evidence(candidate_ids=ids))
        for field in ("marginals", "conditional", "unigram"):
            for values in ([], [0.2], [0.2, 0.2, 0.2]):
                with self.subTest(field=field, values=values):
                    with self.assertRaises(ValueError):
                        score_candidates(**evidence(**{field: values}))

    def test_rejects_invalid_scalars_probability_ranges_and_subset_mass(self):
        for field in ("marginals", "conditional", "unigram"):
            for value in (
                True,
                "0.2",
                None,
                Decimal("0.2"),
                Fraction(1, 5),
                math.nan,
                math.inf,
                -math.inf,
                -0.1,
                1.1,
                10**400,
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        score_candidates(**evidence(**{field: [value, 0.2]}))
            with self.subTest(field=field, mass="over_one"):
                with self.assertRaises(ValueError):
                    score_candidates(**evidence(**{field: [0.6, 0.5]}))

    def test_neutral_cases_still_validate_every_input(self):
        invalid = (
            {"conditional": [0, 0.4]},
            {"unigram": [0.4, 0]},
            {"marginals": [0.8, 0.8]},
            {"candidate_ids": [101, 101]},
            {"position": True},
            {"position": 0},
            {"position": 1.0},
            {"position": "1"},
            {"parent_path": True},
            {"parent_path": "child"},
            {"spine_weight": True},
            {"fork_weight": "0.6"},
            {"spine_weight": -0.1},
            {"fork_weight": math.inf},
            {"pmi_clip": math.nan},
            {"pmi_clip": -1},
        )
        for neutral in (
            {"position": 1},
            {"spine_weight": 0},
            {"pmi_clip": 0},
            {"marginals": [0, 0]},
        ):
            for bad in invalid:
                with self.subTest(neutral=neutral, bad=bad):
                    with self.assertRaises(ValueError):
                        score_candidates(**evidence(**(neutral | bad)))

    def test_all_public_options_reject_coercion_and_nonfinite_values(self):
        for field in ("spine_weight", "fork_weight", "pmi_clip"):
            for value in (
                True,
                "0.3",
                None,
                Decimal("0.3"),
                math.nan,
                math.inf,
                -math.inf,
                -1,
                10**400,
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        score_candidates(**evidence(**{field: value}))


class DForkTreeRecipeTests(unittest.TestCase):
    def cli(self, root, *args, module="draftfit.research"):
        pythonpath = os.pathsep.join(
            filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))
        )
        return subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            cwd=root,
            env={**os.environ, "PYTHONPATH": pythonpath},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def assert_analytic_answers(self, result):
        self.assertEqual(result["processed_count"], 3)
        self.assertEqual(len(result["answers"]), 3)
        for answer, (name, weight, probabilities, tail) in zip(
            result["answers"], EXPECTED
        ):
            self.assertEqual(answer["id"], name)
            self.assertEqual(answer["candidate_ids"], [101, 102])
            self.assertEqual(answer["weight"], weight)
            for actual, expected in zip(answer["candidate_probs"], probabilities):
                self.assertAlmostEqual(actual, expected, places=13)
            self.assertAlmostEqual(answer["tail_probability"], tail, places=13)

    def test_strict_new_options_and_sample_count_leave_old_recipes_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = recipe(
                root, scoring={"spine_weight": 0, "fork_weight": 1, "pmi_clip": 2}
            )
            self.assertIsInstance(config, DForkTreeScoringConfig)
            for field in ("spine_weight", "fork_weight", "pmi_clip"):
                for value in (True, False, "0.3", -1, math.nan, math.inf):
                    with self.subTest(field=field, value=value):
                        with self.assertRaises(ValueError):
                            recipe(root, scoring={field: value})
            for value in (True, False, "3", 3.0, 0, -1):
                with self.subTest(max_samples=value):
                    with self.assertRaises(ValueError):
                        recipe(
                            root, data={"input_path": "unused", "max_samples": value}
                        )
            for extra in (
                {"model": {}},
                {"runtime": {"device": "cuda"}},
                {"device": "cpu"},
                {"input_kind": "measured"},
            ):
                with self.subTest(extra=extra):
                    with self.assertRaises(ValueError):
                        recipe(root, **extra)
            self.assertEqual(
                InputData(input_path="old", max_samples="2").max_samples, 2
            )
            minimal = recipe(root).model_dump()
            del minimal["output_dir"]
            self.assertEqual(
                ADAPTER.validate_python(minimal).output_dir,
                "outputs/research-dforktree-scoring",
            )

    def test_overrides_reject_coercion_and_accept_native_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "recipe.json"
            source.write_text(recipe(Path(temporary)).model_dump_json())
            valid = load_recipe(
                str(source), ("data.max_samples=2", "scoring.spine_weight=0")
            )
            self.assertEqual(valid.data.max_samples, 2)
            self.assertEqual(valid.scoring.spine_weight, 0)
            for override in (
                "data.max_samples=true",
                'data.max_samples="2"',
                "scoring.pmi_clip=false",
                'scoring.fork_weight="0.6"',
            ):
                with self.subTest(override=override):
                    with self.assertRaises(ValueError):
                        load_recipe(str(source), (override,))

    def test_nonblank_limit_hashes_unparsed_binary_suffix_from_full_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = recipe(
                root,
                input_kind="recorded",
                data={
                    "input_path": str(root / "input.json"),
                    "max_samples": 2,
                },
            )
            prefix = b"\n \r\n" + json.dumps(row(id="first")).encode() + b"\n\n"
            prefix += (
                json.dumps(row(id="second", parent_path="forked")).encode() + b"\n"
            )
            suffix = b"not JSON\xff" + b"x" * (1024 * 1024 + 17)
            source = root / "input.json"
            source.write_bytes(prefix + suffix)
            result = run_dforktree_scoring(config)
            self.assertEqual(
                [answer["id"] for answer in result["answers"]], ["first", "second"]
            )
            self.assertEqual(result["processed_count"], 2)
            self.assertEqual(result["input_kind"], "recorded")
            self.assertEqual(result["input_digest_scope"], "full_file")
            self.assertEqual(
                result["input_sha256"], hashlib.sha256(prefix + suffix).hexdigest()
            )
            self.assertFalse(Path(config.output_dir).exists())
            source.write_bytes(prefix + suffix + b"changed")
            changed = run_dforktree_scoring(config)
            self.assertEqual(changed["answers"], result["answers"])
            self.assertNotEqual(changed["input_sha256"], result["input_sha256"])
            config.data.max_samples = 3
            with self.assertRaises(ValueError):
                run(config)
            self.assertFalse(Path(config.output_dir).parent.exists())

    def test_empty_and_closed_row_schema_fail_without_publishing(self):
        bad_rows = [
            [],
            None,
            {},
            row(extra=True),
            row(id=""),
            row(id=1),
            row(position=True),
            row(parent_path="unknown"),
            row(marginals=["0.5", 0.3]),
            row(conditional=[0, 0.4]),
        ]
        missing = row()
        del missing["unigram"]
        bad_rows.append(missing)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = recipe(root)
            source = root / "input.json"
            for payload in (b"", b"\n \t\r\n", b"broken JSON\n", b"\xff\n"):
                with self.subTest(payload=payload):
                    source.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        run(config)
                    self.assertFalse(Path(config.output_dir).parent.exists())
            for invalid in bad_rows:
                with self.subTest(row=invalid):
                    source.write_text(json.dumps(invalid) + "\n")
                    with self.assertRaises(ValueError):
                        run(config)
                    self.assertFalse(Path(config.output_dir).parent.exists())

    def test_late_invalid_row_keeps_output_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = recipe(root)
            (root / "input.json").write_text(json.dumps(row()) + "\n{}\n")
            with self.assertRaises(ValueError):
                run(config)
            self.assertFalse(Path(config.output_dir).parent.exists())

    def test_existing_output_is_refused_before_input_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = recipe(root)
            destination = Path(config.output_dir)
            destination.mkdir(parents=True)
            marker = destination / "customized.txt"
            marker.write_bytes(b"keep these bytes\n")
            with self.assertRaises(FileExistsError):
                run(config)
            self.assertEqual(marker.read_bytes(), b"keep these bytes\n")
            self.assertEqual(list(destination.iterdir()), [marker])

    def test_plan_is_input_independent_and_catalog_carries_source_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = recipe(Path(temporary))
            result = plan(config)
            self.assertEqual(
                result["implementation"], "cpu_probability_scoring_reference"
            )
            self.assertEqual(result["execution"], "not-run")
            self.assertFalse(Path(config.output_dir).parent.exists())
        source = next(
            item for item in source_catalog() if item["project"] == "dforktree_scoring"
        )
        self.assertEqual(source["revision"], "eeed8b698b8822aa2767ec5c191efc4e1b6cdbe3")
        self.assertEqual(source["url"], "https://github.com/Perseus037/DForkTree")
        self.assertEqual(source["license"], "Apache-2.0")
        self.assertFalse(source["production_engine_validated"])

    def test_exported_recipe_cli_plans_runs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = export_assets(Path(temporary) / "recipes")
            destination = root / "outputs/research-dforktree-scoring"
            planned = self.cli(
                root, "-c", RECIPE, "--plan", "data.input_path=missing.json"
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertEqual(json.loads(planned.stdout)["execution"], "not-run")
            self.assertFalse(destination.exists())
            executed = self.cli(root, "research", "-c", RECIPE, module="draftfit.cli")
            self.assertEqual(executed.returncode, 0, executed.stderr)
            result_path = destination / "result.json"
            result_bytes = result_path.read_bytes()
            result = json.loads(result_bytes)
            self.assertEqual(json.loads(executed.stdout), result)
            self.assert_analytic_answers(result)
            self.assertEqual(
                result["implementation_scope"], "cpu_probability_scoring_reference"
            )
            self.assertEqual(result["input_kind"], "synthetic")
            self.assertEqual(result["input_digest_scope"], "full_file")
            self.assertEqual(
                result["input_sha256"],
                hashlib.sha256((root / FIXTURE).read_bytes()).hexdigest(),
            )
            for field in (
                "ngram_counting",
                "corpus_smoothing",
                "tree_construction",
                "verifier",
                "serving_integration",
                "training",
            ):
                self.assertEqual(result[field], "not_implemented")
            self.assertEqual(result["tail_convention"], "true_zero_no_floor")
            self.assertFalse(result["production_engine_validated"])
            self.assertFalse(result["throughput_claim"])
            source = next(
                item
                for item in source_catalog()
                if item["project"] == "dforktree_scoring"
            )
            self.assertEqual(result["source"]["sha256"], source["sha256"])
            self.assertEqual(result["source"]["revision"], source["revision"])
            resolved = json.loads((destination / "recipe.json").read_text())
            self.assertEqual(resolved, load_recipe(str(root / RECIPE)).model_dump())
            refused = self.cli(root, "-c", RECIPE)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("output already exists", refused.stderr)
            self.assertEqual(result_path.read_bytes(), result_bytes)
            failed = self.cli(
                root,
                "-c",
                RECIPE,
                "data.input_path=missing.json",
                "output_dir=failed/result",
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / "failed").exists())


if __name__ == "__main__":
    unittest.main()
