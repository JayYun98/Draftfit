import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from draftfit.data.prepare import (
    load_local_rows,
    normalize_row,
    prepare_dataset,
    split_rows,
)


class LocalDatasetPreparationTest(unittest.TestCase):
    def test_openai_messages_are_normalized_and_developer_is_preserved(self):
        row = normalize_row(
            {
                "messages": [
                    {"role": "developer", "content": "Use terse answers."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        )

        self.assertRegex(row["id"], r"^[0-9a-f]{16}$")
        self.assertEqual(
            [message["role"] for message in row["conversations"]],
            ["developer", "user", "assistant"],
        )
        self.assertEqual(row["conversations"][-1]["content"], "4")

    def test_sharegpt_top_level_system_is_kept(self):
        row = normalize_row(
            {
                "id": "with-system",
                "system": "Answer in JSON.",
                "conversations": [
                    {"from": "human", "value": "Give me an object."},
                    {"from": "gpt", "value": "{}"},
                ],
            },
            data_format="sharegpt",
        )

        self.assertEqual(
            row["conversations"][0],
            {"role": "system", "content": "Answer in JSON."},
        )

    def test_semantic_metadata_is_rejected_instead_of_dropped(self):
        with self.assertRaisesRegex(ValueError, "'weight'"):
            normalize_row(
                {
                    "messages": [
                        {"role": "user", "content": "q", "weight": 0},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )

        with self.assertRaisesRegex(ValueError, "'source'"):
            normalize_row(
                {
                    "source": "private-export",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            )

    def test_sharegpt_rows_are_converted_without_dropping_tool_metadata(self):
        row = normalize_row(
            {
                "id": 7,
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "conversations": [
                    {"from": "human", "value": "Look up the answer."},
                    {
                        "from": "gpt",
                        "value": "",
                        "tool_calls": [{"id": "call-1"}],
                    },
                    {"from": "tool", "value": "The answer is 42."},
                    {"from": "gpt", "value": "42"},
                ],
            },
            data_format="sharegpt",
        )

        self.assertEqual(row["id"], "7")
        self.assertEqual(
            [message["role"] for message in row["conversations"]],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(row["conversations"][1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(row["tools"][0]["function"]["name"], "lookup")

    def test_prepare_supports_json_arrays_and_deterministic_eval_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "examples.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "messages": [
                                {"role": "user", "content": f"q-{index}"},
                                {"role": "assistant", "content": f"a-{index}"},
                            ]
                        }
                        for index in range(20)
                    ]
                ),
                encoding="utf-8",
            )

            first = prepare_dataset(
                input_path,
                root / "first_train.jsonl",
                split_eval=True,
            )
            second = prepare_dataset(
                input_path,
                root / "second_train.jsonl",
                split_eval=True,
            )

            self.assertEqual(
                (first.input_rows, first.train_rows, first.eval_rows), (20, 19, 1)
            )
            self.assertEqual(
                (root / "first_train.jsonl").read_text(),
                (root / "second_train.jsonl").read_text(),
            )
            self.assertEqual(
                (root / "first_test.jsonl").read_text(),
                (root / "second_test.jsonl").read_text(),
            )

    def test_split_prefers_small_prompt_groups_near_the_target_ratio(self):
        rows = [
            normalize_row(
                {
                    "id": str(index),
                    "messages": [
                        {"role": "user", "content": "repeated prompt"},
                        {"role": "assistant", "content": f"answer-{index}"},
                    ],
                }
            )
            for index in range(19)
        ]
        rows.append(
            normalize_row(
                {
                    "id": "unique",
                    "messages": [
                        {"role": "user", "content": "unique prompt"},
                        {"role": "assistant", "content": "unique answer"},
                    ],
                }
            )
        )

        train, evaluation = split_rows(rows)

        self.assertEqual({row["id"] for row in evaluation}, {"unique"})
        self.assertEqual(len(train), 19)

    def test_max_rows_does_not_consume_the_next_source_row(self):
        consumed = []

        def source_rows():
            for index in range(3):
                consumed.append(index)
                yield index + 1, {
                    "messages": [
                        {"role": "user", "content": f"q-{index}"},
                        {"role": "assistant", "content": f"a-{index}"},
                    ]
                }

        with patch(
            "draftfit.data.prepare._input_rows",
            return_value=source_rows(),
        ):
            rows = load_local_rows("unused.jsonl", max_rows=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(consumed, [0, 1])

    def test_large_json_file_requires_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            path.write_text("[]", encoding="utf-8")

            with patch("draftfit.data.prepare.MAX_JSON_FILE_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "use JSONL"):
                    load_local_rows(path)

    def test_invalid_rows_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bad.jsonl"
            input_path.write_text(
                json.dumps(
                    {"messages": [{"role": "user", "content": "missing answer"}]}
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "requires at least one user and assistant"
            ):
                prepare_dataset(input_path, root / "out" / "train.jsonl")
            self.assertFalse((root / "out").exists())

    def test_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "data.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "train.jsonl"
            output_path.write_text("keep\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_dataset(input_path, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep\n")

    def test_eval_output_cannot_replace_input_before_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "data.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "train.jsonl"

            with self.assertRaisesRegex(ValueError, "differ from input"):
                prepare_dataset(
                    input_path,
                    output_path,
                    split_eval=True,
                    eval_output_path=input_path,
                )
            self.assertFalse(output_path.exists())

    def test_reserved_outputs_are_cleaned_on_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "data.jsonl"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "q-1"},
                        {"role": "assistant", "content": "a-1"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "q-2"},
                        {"role": "assistant", "content": "a-2"},
                    ]
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            train_path = root / "train.jsonl"
            eval_path = root / "test.jsonl"

            with patch(
                "draftfit.data.prepare._write_jsonl",
                side_effect=RuntimeError("write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    prepare_dataset(
                        input_path,
                        train_path,
                        split_eval=True,
                        eval_output_path=eval_path,
                    )

            self.assertFalse(train_path.exists())
            self.assertFalse(eval_path.exists())

    def test_cleanup_does_not_remove_a_raced_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "data.jsonl"
            rows = [
                {
                    "messages": [
                        {"role": "user", "content": "q-1"},
                        {"role": "assistant", "content": "a-1"},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "q-2"},
                        {"role": "assistant", "content": "a-2"},
                    ]
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            train_path = root / "train.jsonl"
            eval_path = root / "test.jsonl"

            def replace_then_fail(handle, _rows):
                train_path.unlink()
                train_path.write_text("owned by another process\n", encoding="utf-8")
                raise RuntimeError("write failed")

            with patch(
                "draftfit.data.prepare._write_jsonl",
                side_effect=replace_then_fail,
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    prepare_dataset(
                        input_path,
                        train_path,
                        split_eval=True,
                        eval_output_path=eval_path,
                    )

            self.assertEqual(
                train_path.read_text(encoding="utf-8"),
                "owned by another process\n",
            )
            self.assertFalse(eval_path.exists())

    def test_duplicate_conversations_cannot_cross_an_eval_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "duplicates.jsonl"
            row = {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            input_path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate rows"):
                prepare_dataset(input_path, root / "train.jsonl", split_eval=True)
            self.assertFalse((root / "train.jsonl").exists())

    def test_same_prompt_with_different_answers_stays_in_one_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "alternates.jsonl"
            rows = [
                {
                    "id": "a",
                    "messages": [
                        {"role": "user", "content": "same prompt"},
                        {"role": "assistant", "content": "answer A"},
                    ],
                },
                {
                    "id": "b",
                    "messages": [
                        {"role": "user", "content": "same prompt"},
                        {"role": "assistant", "content": "answer B"},
                    ],
                },
                {
                    "id": "c",
                    "messages": [
                        {"role": "user", "content": "different prompt"},
                        {"role": "assistant", "content": "answer C"},
                    ],
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            prepare_dataset(input_path, root / "train.jsonl", split_eval=True)
            train_ids = {
                json.loads(line)["id"]
                for line in (root / "train.jsonl").read_text().splitlines()
            }
            eval_ids = {
                json.loads(line)["id"]
                for line in (root / "test.jsonl").read_text().splitlines()
            }
            self.assertFalse({"a", "b"} & train_ids and {"a", "b"} & eval_ids)

    def test_cli_data_prepare_dispatches_without_source_only_imports(self):
        from draftfit.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "messages.jsonl"
            output_path = root / "prepared.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "data",
                            "prepare",
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                        ]
                    ),
                    0,
                )
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["train_rows"], 1)
            self.assertEqual(Path(summary["train_path"]), output_path)

    def test_cli_data_prepare_help_does_not_treat_percent_as_formatting(self):
        from draftfit.cli import main

        with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as exited:
            main(["data", "prepare", "--help"])
        self.assertEqual(exited.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
