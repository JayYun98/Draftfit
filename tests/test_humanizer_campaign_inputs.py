"""Behavioral checks for the Humanizer input-only campaign sidecar."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from draftfit.data.prepare_presets import process_sharegpt_row
from scripts.prepare_humanizer_campaign_inputs import collect, split_rows, write_outputs

PROSE = (
    "The morning started with a quiet walk through the neighborhood. "
    "Several people were already opening their shops, and the air smelled of fresh bread. "
    "I stopped to talk with a neighbor about the garden beside our building. "
    "She explained how the new plants had survived a week of heavy rain. "
    "By the time I got home, I had a clearer idea of what to do with my afternoon."
)


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return text.split()


def row(
    index, answer=PROSE, prompt="Write a short personal story about the neighborhood."
):
    return {
        "id": index,
        "source": "fixture",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }


class CampaignInputsTests(unittest.TestCase):
    def test_group_and_draft_dedup_before_split(self):
        rows = [
            row(0),
            row(1),
            row(2, prompt="Write another story about the neighborhood."),
            row(3, answer=PROSE + "\n\n```python\nprint(1)\n```"),
            {
                "id": 4,
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "tool", "value": PROSE},
                ],
            },
        ]
        counts = Counter()
        selected = collect(rows, process_sharegpt_row, Tokenizer(), 5, counts)
        self.assertEqual(1, len(selected))
        self.assertEqual(1, counts["duplicate_parent"])
        self.assertEqual(1, counts["duplicate_draft"])
        self.assertEqual(1, counts["code_math_or_tool_conversation"])
        self.assertEqual(1, counts["tool_or_nonassistant_ending"])
        self.assertEqual(
            "source_assistant_prose_draft_not_target_rewrite", selected[0]["input_kind"]
        )

    def test_fixed_splits_and_manifest(self):
        rows = [
            row(
                i,
                prompt=f"Write story number {i} about the neighborhood.",
                answer=PROSE + f" Today I thought about garden number {i}.",
            )
            for i in range(6)
        ]
        selected = collect(rows, process_sharegpt_row, Tokenizer(), 6, Counter())
        splits = split_rows(selected, 2, 1, 1, 2, 42)
        self.assertEqual(
            {
                "train": 2,
                "validation": 1,
                "test": 1,
                "train_replacements": 1,
                "validation_replacements": 1,
                "test_replacements": 0,
            },
            {k: len(v) for k, v in splits.items()},
        )
        self.assertEqual(
            6, len({r["parent_hash"] for part in splits.values() for r in part})
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            manifest = write_outputs(
                output,
                splits,
                Counter(source_rows_seen=6),
                source_revision="abc",
                seed=42,
            )
            self.assertEqual("abc", manifest["source_revision"])
            self.assertEqual(2, len((output / "train.jsonl").read_text().splitlines()))
            self.assertEqual(
                "input_drafts_only_no_target_generated_rewrites",
                json.loads((output / "manifest.json").read_text())["kind"],
            )

    def test_insufficient_rows_fails(self):
        with self.assertRaisesRegex(ValueError, "insufficient usable"):
            split_rows([], 2, 1, 1, 0, 42)

    def test_rejects_math_source_and_arithmetic_answer(self):
        math_source = row(1)
        math_source["source"] = "meta-math/MetaMathQA"
        arithmetic = row(2, answer=PROSE + " We find that 30 * 10 = 300.")
        selected = collect(
            [math_source, arithmetic], process_sharegpt_row, Tokenizer(), 2, Counter()
        )
        self.assertEqual([], selected)


if __name__ == "__main__":
    unittest.main()
