"""CPU behavior of the session lookup proposal and its public recipe."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path

from draftfit.cli import main
from draftfit.research.papers.session_lookup.method import propose

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples/configs/research/session-lookup.yaml"
)


def expected_proposal(tokens, k, minimum, maximum, match):
    """Rank complete continuations by suffix length, then by position."""
    candidates = []
    for cut in range(1, len(tokens) - k + 1):
        width = 0
        # Leave at least one position between the cut and the query start.
        limit = min(maximum, cut, len(tokens) - cut - 1)
        while width < limit and tokens[cut - width - 1] == tokens[-width - 1]:
            width += 1
        if width >= minimum:
            candidates.append((width, cut))
    if not candidates:
        return []
    _, cut = max(
        candidates,
        key=lambda item: (item[0], item[1] if match == "recent" else -item[1]),
    )
    return list(tokens[cut : cut + k])


class SessionLookupTest(unittest.TestCase):
    def test_hand_calculated_proposals(self):
        cases = [
            # Adjacent equal pairs: the continuation would start at the query.
            ([1, 2, 1, 2], 1, 2, 2, "earliest", []),
            # The continuation can end at the last token, but may not be short.
            ([4, 5, 4], 2, 1, 3, "recent", [5, 4]),
            ([4, 5, 4], 3, 1, 3, "recent", []),
            ([0, 1, 0, 2, 0], 3, 1, 1, "recent", [1, 0, 2]),
            # A longer suffix wins even though there is a more recent unigram.
            ([1, 2, 8, 1, 2, 9, 2, 7, 1, 2], 2, 1, 2, "recent", [9, 2]),
            ([1, 2, 8, 1, 2, 9, 2, 7, 1, 2], 2, 1, 2, "earliest", [8, 1]),
            ([2, 8, 1, 2], 1, 1, 2, "recent", [8]),
            ([2, 8, 1, 2], 1, 2, 2, "recent", []),
        ]
        for tokens, k, minimum, maximum, match, expected in cases:
            with self.subTest(tokens=tokens, k=k, minimum=minimum, match=match):
                self.assertEqual(
                    propose(
                        tokens, k=k, min_ngram=minimum, max_ngram=maximum, match=match
                    ),
                    expected,
                )

    def test_all_binary_contexts_through_length_seven(self):
        bounds = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3))
        for size in range(8):
            for tokens in product((0, 1), repeat=size):
                for minimum, maximum in bounds:
                    for k, match in product((1, 2, 3), ("earliest", "recent")):
                        self.assertEqual(
                            propose(
                                list(tokens),
                                k=k,
                                min_ngram=minimum,
                                max_ngram=maximum,
                                match=match,
                            ),
                            expected_proposal(tokens, k, minimum, maximum, match),
                            (tokens, k, minimum, maximum, match),
                        )


class SessionLookupCLITest(unittest.TestCase):
    def invoke(self, recipe, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["research", "--config", str(recipe), *args]), 0)
        return json.loads(output.getvalue())

    def test_example_plan_reads_inline_context_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            plan = self.invoke(EXAMPLE, "--plan", f"output_dir={destination}")
            self.assertEqual(plan["project"], "session_lookup")
            self.assertEqual(plan["execution"], "not-run")
            self.assertIsInstance(plan["config"]["context_ids"], list)
            self.assertTrue(plan["config"]["context_ids"])
            self.assertFalse(destination.exists())

    def test_run_persists_proposed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            result = self.invoke(
                EXAMPLE,
                "context_ids=[0,4,0,6,0]",
                "min_ngram=1",
                "max_ngram=1",
                "k=2",
                "match=recent",
                f"output_dir={destination}",
            )
            self.assertEqual(result["proposed_ids"], [6, 0])
            saved = json.loads((destination / "result.json").read_text())
            self.assertEqual(saved["proposed_ids"], [6, 0])

    def test_rejects_unordered_yaml_context(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(
                [
                    "research",
                    "-c",
                    str(EXAMPLE),
                    "--plan",
                    "context_ids=!!set {3, 1, 2}",
                ]
            )
        self.assertEqual(exc.exception.code, 2)

    def test_invalid_recipe_fields_are_rejected_by_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out"
            recipe = Path(directory) / "recipe.json"
            valid = dict(
                method="session_lookup",
                context_ids=[0, 3, 0],
                k=1,
                min_ngram=1,
                max_ngram=2,
                match="earliest",
                output_dir=str(destination),
            )
            recipe.write_text(json.dumps(valid))
            self.assertEqual(self.invoke(recipe, "--plan")["project"], "session_lookup")
            for field, value in (
                ("context_ids", 1),
                ("context_ids", "0,1"),
                ("context_ids", [-1]),
                ("context_ids", [True]),
                ("context_ids", [1.0]),
                ("context_ids", ["1"]),
                ("k", 0),
                ("k", True),
                ("min_ngram", 0),
                ("min_ngram", True),
                ("min_ngram", 3),
                ("max_ngram", 0),
                ("max_ngram", True),
                ("match", "last"),
                ("output_dir", ""),
            ):
                with self.subTest(field=field, value=value):
                    recipe.write_text(json.dumps({**valid, field: value}))
                    with (
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as exc,
                    ):
                        main(["research", "--config", str(recipe), "--plan"])
                    self.assertEqual(exc.exception.code, 2)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
