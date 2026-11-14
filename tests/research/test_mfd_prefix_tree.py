import copy
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from draftfit.research.mfd import PrefixNode, PrefixTree, select_prefixes

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/configs/research/mfd-prefix-tree.yaml"
FIXTURE = ROOT / "examples/configs/research/data/mfd-prefix-tree.json"
SOURCE_REVISION = "95040343de00f15aef8c6350a578c35e0cd53475"
PAPER_REVISION = "2611.13017v1"


def exhaustive(probabilities, priors=None):
    k, depth, vocab = (
        len(probabilities),
        len(probabilities[0]),
        len(probabilities[0][0]),
    )
    weights = [Fraction(1, k)] * k if priors is None else list(map(Fraction, priors))
    ranked = []
    for size in range(1, depth + 1):
        for prefix in itertools.product(range(vocab), repeat=size):
            score = max(
                weights[mode]
                * math.prod(
                    Fraction(probabilities[mode][i][token])
                    for i, token in enumerate(prefix)
                )
                for mode in range(k)
            )
            if score > 0:
                ranked.append((prefix, score))
    return sorted(ranked, key=lambda item: (-item[1], len(item[0]), item[0]))


class MFDSelectionTest(unittest.TestCase):
    def assert_matches_oracle(self, probabilities, priors=None, budgets=None):
        expected = exhaustive(probabilities, priors)
        if budgets is None:
            budgets = range(min(len(expected) + 1, 256) + 1)
        for budget in budgets:
            with self.subTest(budget=budget):
                tree = select_prefixes(probabilities, priors=priors, budget=budget)
                self.assertIsInstance(tree, PrefixTree)
                self.assertIsInstance(tree.nodes, tuple)
                self.assertEqual(
                    [node.prefix for node in tree.nodes],
                    [p for p, _ in expected[:budget]],
                )
                self.assertEqual(len(tree.nodes), min(budget, len(expected)))
                parents = {(): 0}
                for index, (node, (prefix, product)) in enumerate(
                    zip(tree.nodes, expected), 1
                ):
                    self.assertIsInstance(node, PrefixNode)
                    self.assertEqual(node.node_id, index)
                    self.assertEqual(node.parent, parents[prefix[:-1]])
                    self.assertLess(node.parent, node.node_id)
                    self.assertTrue(math.isfinite(node.log_score))
                    self.assertAlmostEqual(
                        node.log_score, math.log(float(product)), places=12
                    )
                    parents[prefix] = index

    def test_coherent_modes_differ_observably_from_collapsed_marginals(self):
        row = json.loads(FIXTURE.read_text())
        probabilities, priors = row["probabilities"], row["priors"]
        collapsed = [
            [
                [
                    math.fsum(priors[k] * probabilities[k][i][t] for k in range(2))
                    for t in range(2)
                ]
                for i in range(2)
            ]
        ]
        actual = select_prefixes(probabilities, priors=priors, budget=4)
        independent = select_prefixes(collapsed, budget=4)
        names = lambda tree: [
            "".join("AB"[t] for t in node.prefix) for node in tree.nodes
        ]
        self.assertEqual(names(actual), ["A", "AA", "B", "BB"])
        self.assertEqual(names(independent), ["A", "B", "AA", "AB"])
        for node, expected in zip(actual.nodes, (0.54, 0.486, 0.36, 0.324)):
            self.assertAlmostEqual(math.exp(node.log_score), expected, places=14)
        self.assertNotAlmostEqual(math.exp(actual.nodes[0].log_score), 0.58)
        self.assert_matches_oracle(probabilities, priors)
        self.assert_matches_oracle(collapsed)

    def test_randomized_dyadic_tables_and_all_budgets(self):
        rng = random.Random(261113017)

        def distribution(size):
            weights = [1.0]
            while len(weights) < size:
                split = rng.randrange(len(weights))
                value = weights[split] / 2
                weights[split : split + 1] = [value, value]
            if size > 1 and rng.randrange(3) == 0:
                source, destination = rng.sample(range(size), 2)
                weights[destination] += weights[source]
                weights[source] = 0.0
                if weights[destination] not in (0.0, 0.25, 0.5, 1.0):
                    return distribution(size)
            rng.shuffle(weights)
            return weights

        for case in range(72):
            modes, depth, vocab = (
                rng.randint(1, 3),
                rng.randint(1, 3),
                rng.randint(1, 3),
            )
            probabilities = [
                [distribution(vocab) for _ in range(depth)] for _ in range(modes)
            ]
            priors = None if case % 3 == 0 else distribution(modes)
            with self.subTest(case=case, probabilities=probabilities, priors=priors):
                self.assert_matches_oracle(probabilities, priors)

    def test_single_mode_zeros_ties_and_input_is_unchanged(self):
        probabilities = [[[0.5, 0.25, 0.25], [0, 0.5, 0.5], [0.5, 0, 0.5]]]
        original = copy.deepcopy(probabilities)
        self.assert_matches_oracle(probabilities)
        self.assertEqual(probabilities, original)
        self.assertEqual(select_prefixes(probabilities, budget=0), PrefixTree(()))

    def test_duplicates_expand_with_their_own_mode_parent_score(self):
        probabilities = [
            [[0.75, 0.25], [0.75, 0.25]],
            [[0.5, 0.5], [0.0, 1.0]],
        ]
        self.assert_matches_oracle(probabilities)
        scores = {
            node.prefix: math.exp(node.log_score)
            for node in select_prefixes(probabilities, budget=6).nodes
        }
        self.assertAlmostEqual(scores[(0, 1)], 0.25)
        self.assertAlmostEqual(scores[(1, 1)], 0.25)

    def test_uniform_mode_duplication_preserves_selection_and_shifts_scores(self):
        probabilities = [
            [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]],
            [[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],
        ]
        for budget in range(14):
            original = select_prefixes(probabilities, budget=budget)
            duplicated = select_prefixes(probabilities * 2, budget=budget)
            self.assertEqual(
                [n.prefix for n in duplicated.nodes], [n.prefix for n in original.nodes]
            )
            for before, after in zip(original.nodes, duplicated.nodes):
                self.assertAlmostEqual(
                    after.log_score, before.log_score - math.log(2), places=14
                )
        self.assert_matches_oracle(probabilities * 2)

    def test_mode_permutations_with_nonuniform_and_zero_priors(self):
        probabilities = [
            [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]],
            [[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],
            [[1, 0, 0], [0, 0, 1]],
        ]
        for priors in ([0.5, 0.25, 0.25], [0.5, 0.5, 0]):
            self.assert_matches_oracle(probabilities, priors)
            for order in itertools.permutations(range(3)):
                permuted = [probabilities[k] for k in order]
                weights = [priors[k] for k in order]
                for budget in range(14):
                    self.assertEqual(
                        select_prefixes(permuted, priors=weights, budget=budget),
                        select_prefixes(probabilities, priors=priors, budget=budget),
                    )
        active = select_prefixes(probabilities[:2], budget=12)
        ignored = select_prefixes(probabilities, priors=[0.5, 0.5, 0], budget=12)
        self.assertEqual(active, ignored)

    def test_deterministic_chains_and_shorter_first_ties(self):
        deterministic = [[[0.0, 1.0]] * 16] * 8
        nodes = select_prefixes(deterministic, budget=256).nodes
        self.assertEqual(len(nodes), 16)
        for index, node in enumerate(nodes, 1):
            self.assertEqual(
                node, PrefixNode(index, index - 1, (1,) * index, math.log(1 / 8))
            )
        probabilities = [[[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]]]
        self.assertEqual(
            [n.prefix for n in select_prefixes(probabilities, budget=6).nodes],
            [(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)],
        )
        self.assert_matches_oracle(probabilities)

    def test_rounded_total_scores_use_global_token_order(self):
        probabilities = [[[1e-300, 1.0], [0.4999999999999999, 0.5000000000000001]]]
        ranked = []
        for depth in (1, 2):
            for prefix in itertools.product(range(2), repeat=depth):
                score = math.fsum(
                    math.log(probabilities[0][position][token])
                    for position, token in enumerate(prefix)
                )
                ranked.append((prefix, score))
        ranked.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
        self.assertEqual(
            [prefix for prefix, _ in ranked],
            [(1,), (1, 1), (1, 0), (0,), (0, 0), (0, 1)],
        )
        self.assertEqual(ranked[-2][1], ranked[-1][1])
        self.assertEqual(ranked[-1][1], -691.4686750787736)
        for budget in (5, 6):
            with self.subTest(budget=budget):
                tree = select_prefixes(probabilities, budget=budget)
                self.assertEqual(
                    [(node.prefix, node.log_score) for node in tree.nodes],
                    ranked[:budget],
                )
                self.assertEqual(
                    [node.parent for node in tree.nodes],
                    [0, 1, 1, 0, 4, 4][:budget],
                )

    def test_full_budget_and_geometry_limits(self):
        self.assert_matches_oracle([[[0.5, 0.5]] * 8], budgets=(0, 1, 255, 256))
        row = [1.0] + [0.0] * 4095
        probabilities = [[row] * 8] * 8
        tree = select_prefixes(probabilities, budget=256)
        self.assertEqual(len(tree.nodes), 8)
        self.assertTrue(all(n.prefix == (0,) * n.node_id for n in tree.nodes))

    def test_small_positive_support_and_tolerance_without_renormalization(self):
        tiny = float.fromhex("0x0.0000000000001p-1022")
        tree = select_prefixes([[[1.0, tiny]] * 2], budget=6)
        self.assertEqual(len(tree.nodes), 6)
        self.assertTrue(all(math.isfinite(n.log_score) for n in tree.nodes))
        by_prefix = {n.prefix: n for n in tree.nodes}
        self.assertEqual(by_prefix[(1, 1)].log_score, 2 * math.log(tiny))
        near = 0.4 - 5e-9
        tree = select_prefixes([[[0.6, near]]], priors=[1 - 5e-9], budget=2)
        for node, value in zip(tree.nodes, (0.6, near)):
            self.assertAlmostEqual(
                math.exp(node.log_score), value * (1 - 5e-9), places=15
            )

    def test_invalid_geometry_probabilities_and_priors_even_at_zero_budget(self):
        invalid = [
            None,
            [],
            {},
            "bad",
            [None],
            [[]],
            [[None]],
            [[[]]],
            [[[1]]] * 9,
            [[[1]] * 17],
            [[[1] + [0] * 4096]],
            [[[1] + [0] * 2048] * 16] * 8,
            [[[1]], [[1], [1]]],
            [[[1], [0.5, 0.5]]],
            [[[0.2, 0.2]]],
            [[[0, 0]]],
            [[[1 + 2e-8]]],
        ]
        for value in (
            True,
            False,
            "1",
            None,
            math.nan,
            math.inf,
            -math.inf,
            -0.1,
            1.1,
            10**1000,
            1j,
            [],
            {},
        ):
            invalid.append([[[value, 0]]])
        for probabilities in invalid:
            for budget in (0, 2):
                with self.subTest(
                    probabilities=str(probabilities)[:100], budget=budget
                ):
                    with self.assertRaises(ValueError):
                        select_prefixes(probabilities, budget=budget)
        for priors in (
            [],
            [1],
            [0, 0],
            [0.4, 0.4],
            [True, False],
            [math.inf, 0],
            [math.nan, 0],
            ["0.5", 0.5],
            [1.1, -0.1],
            "bad",
            {},
            1,
        ):
            for budget in (0, 2):
                with self.subTest(priors=priors, budget=budget):
                    with self.assertRaises(ValueError):
                        select_prefixes([[[1]], [[1]]], priors=priors, budget=budget)
        with self.assertRaises(ValueError):
            select_prefixes([[[0.5, 0.5 - 2e-8]]], budget=0)

    def test_invalid_budgets(self):
        for budget in (-1, 257, True, False, 1.0, "1", None):
            with (
                self.subTest(budget=budget),
                self.assertRaisesRegex(ValueError, "budget"),
            ):
                select_prefixes([[[1]]], budget=budget)


class MFDRecipeTest(unittest.TestCase):
    def config(self, input_path, output, **extra):
        from draftfit.research.config import ADAPTER

        return ADAPTER.validate_python(
            {
                "method": "mfd_prefix_tree",
                "data": {"input_path": str(input_path)},
                "tree": {"budget": 4},
                "output_dir": str(output),
                **extra,
            }
        )

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "draftfit.cli", "research", *map(str, args)],
            cwd=ROOT,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
                WANDB_MODE="disabled",
            ),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def assert_metadata(self, result):
        self.assertEqual(result["implementation"], "cpu_reference")
        self.assertEqual(result["source_revision"], SOURCE_REVISION)
        self.assertEqual(result["paper_revision"], PAPER_REVISION)
        self.assertIs(result["model_execution_performed"], False)
        self.assertEqual(result["serving_validation"], "not-run")

    def test_recipe_and_strict_config_limits(self):
        from draftfit.research.config import ADAPTER, load_recipe

        config = load_recipe(str(RECIPE))
        self.assertEqual(config.method, "mfd_prefix_tree")
        self.assertEqual(
            config.data.input_path,
            "examples/configs/research/data/mfd-prefix-tree.json",
        )
        raw = config.model_dump()
        for field, values in (
            ("budget", [-1, 257, True, 1.0, "1"]),
            ("max_samples", [0, 33, True, 1.0, "1"]),
        ):
            for value in values:
                invalid = copy.deepcopy(raw)
                invalid["tree" if field == "budget" else "data"][field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    ADAPTER.validate_python(invalid)
        for field in ("model", "runtime", "unknown"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ADAPTER.validate_python({**raw, field: {}})
        for budget in (0, 256):
            self.assertEqual(
                load_recipe(str(RECIPE), [f"tree.budget={budget}"]).tree.budget, budget
            )

    def test_real_cli_plan_run_atomic_result_and_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "output"
            plan = self.cli(
                "-c",
                RECIPE,
                "--plan",
                f"output_dir={destination}",
                f"data.input_path={root / 'missing'}",
            )
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertEqual(json.loads(plan.stdout)["execution"], "not-run")
            self.assert_metadata(json.loads(plan.stdout))
            self.assertFalse(destination.exists())
            execution = self.cli("-c", RECIPE, f"output_dir={destination}")
            self.assertEqual(execution.returncode, 0, execution.stderr)
            result = json.loads(execution.stdout)
            self.assert_metadata(result)
            self.assertEqual(result["score_semantics"], "max_over_weighted_modes")
            saved = json.loads((destination / "result.json").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(
                {p.name for p in destination.iterdir()}, {"recipe.json", "result.json"}
            )
            recipe = json.loads((destination / "recipe.json").read_text())
            self.assertEqual(recipe["tree"]["budget"], 4)
            answer = result["answers"][0]
            self.assertEqual(answer["id"], "coherent-two-mode")
            self.assertEqual(
                [n["prefix"] for n in answer["nodes"]], [[0], [0, 0], [1], [1, 1]]
            )
            self.assertEqual([n["node_id"] for n in answer["nodes"]], [1, 2, 3, 4])
            self.assertEqual([n["parent"] for n in answer["nodes"]], [0, 1, 0, 3])
            self.assertTrue(
                all(
                    set(n) == {"node_id", "parent", "prefix", "log_score"}
                    for n in answer["nodes"]
                )
            )
            before = {p.name: p.read_bytes() for p in destination.iterdir()}
            collision = self.cli("-c", RECIPE, f"output_dir={destination}")
            self.assertEqual(collision.returncode, 2)
            self.assertIn("output already exists", collision.stderr)
            self.assertEqual(
                {p.name: p.read_bytes() for p in destination.iterdir()}, before
            )
            self.assertFalse(list(root.glob(".draftfit-research-*")))

    def test_real_cli_bad_later_row_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root / "rows.json", root / "new-parent" / "out"
            data.write_text(FIXTURE.read_text() + '{"probabilities": [[[]]]}\n')
            result = self.cli(
                "-c", RECIPE, f"data.input_path={data}", f"output_dir={output}"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("MFD input row 2", result.stderr)
            self.assertIn("vocabulary", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.parent.exists())
            self.assertFalse(list(root.glob(".draftfit-research-*")))

    def test_empty_and_malformed_input_fails_without_output(self):
        from draftfit.research.runner import run

        payloads = (
            b"",
            b"\n",
            b"{}\n",
            b"[]\n",
            b"null\n",
            b"{bad\n",
            b"\xff\n",
            b'{"probabilities": null}\n',
            b'{"probabilities": [[[NaN]]]}\n',
            b'{"probabilities": [[[1]]], "id": NaN}\n',
            b'{"probabilities": [[[1]]], "extra": Infinity}\n',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, payload in enumerate(payloads):
                data, output = root / "data.json", root / f"out-{index}"
                data.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    run(self.config(data, output))
                self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".draftfit-research-*")))

    def test_jsonl_row_and_byte_limits(self):
        from draftfit.research.runner import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "rows.json"
            row = b'{"probabilities": [[[1]]]}\n'
            data.write_bytes(row * 32)
            result = run(self.config(data, root / "full"))
            self.assertEqual(len(result["answers"]), 32)
            self.assertEqual([r["id"] for r in result["answers"]], list(range(32)))
            data.write_bytes(row * 33)
            with self.assertRaisesRegex(ValueError, "max_samples"):
                run(self.config(data, root / "overflow"))
            self.assertFalse((root / "overflow").exists())
            data.write_bytes(row * 2)
            config = self.config(
                data,
                root / "lower-cap",
                data={"input_path": str(data), "max_samples": 1},
            )
            with self.assertRaisesRegex(ValueError, "max_samples=1"):
                run(config)
            self.assertFalse((root / "lower-cap").exists())
            limit = 8 * 1024 * 1024
            data.write_bytes(row[:-1] + b" " * (limit - len(row)) + b"\n")
            self.assertEqual(
                len(run(self.config(data, root / "exact-byte-limit"))["answers"]), 1
            )
            data.write_bytes(row[:-1] + b" " * (limit - len(row) + 1) + b"\n")
            with self.assertRaisesRegex(ValueError, "exceeds 8 MiB"):
                run(self.config(data, root / "too-long"))
            self.assertFalse((root / "too-long").exists())
            self.assertFalse(list(root.glob(".draftfit-research-*")))

    def test_zero_budget_cli_still_validates_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, data = root / "zero", root / "bad.json"
            result = self.cli("-c", RECIPE, "tree.budget=0", f"output_dir={output}")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["answers"][0]["nodes"], [])
            data.write_text('{"probabilities": [[[0.2, 0.2]]]}\n')
            result = self.cli(
                "-c",
                RECIPE,
                "tree.budget=0",
                f"data.input_path={data}",
                f"output_dir={root / 'bad'}",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("sum to 1", result.stderr)
            self.assertFalse((root / "bad").exists())

    def test_source_metadata_is_lazy_and_cli_visible(self):
        code = (
            "import json, sys; from draftfit.research.catalog import source_catalog; "
            "print(json.dumps({'sources': source_catalog(), 'loaded': "
            "[m for m in ('torch', 'transformers', 'draftfit.research.mfd') if m in sys.modules]}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        decoded = json.loads(result.stdout)
        self.assertEqual(decoded["loaded"], [])
        source = next(
            r for r in decoded["sources"] if r["project"] == "mfd_prefix_tree"
        )
        self.assertEqual(source["revision"], SOURCE_REVISION)
        self.assertEqual(source["paper_revision"], PAPER_REVISION)
        self.assertEqual(
            source["url"], "https://github.com/hhhhhhy5/multi-future-drafting"
        )
        self.assertFalse(source["production_engine_validated"])
        result = self.cli("--sources")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(source, json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main(verbosity=2)
