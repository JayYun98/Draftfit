import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from draftfit.data import prepare_presets as prepare_data
from draftfit.data.prepare import normalize_row


class TestPrepareData(unittest.TestCase):
    def test_parser_exposes_all_supported_presets(self):
        parser = prepare_data.build_parser()
        dataset_action = next(
            action for action in parser._actions if action.dest == "dataset"
        )

        self.assertEqual(prepare_data.SUPPORTED_DATASETS, tuple(dataset_action.choices))
        self.assertTrue(
            {
                "ultrachat",
                "sharegpt",
                "opc",
                "gsm8k",
                "hendrycks_math",
                "codealpaca-20k",
                "camel",
            }.issubset(dataset_action.choices)
        )
        self.assertTrue(
            prepare_data.UNSUPPORTED_VLM_DATASETS.isdisjoint(dataset_action.choices)
        )

    def test_parser_help_renders_literal_eval_split_percentage(self):
        help_text = prepare_data.build_parser().format_help()

        self.assertIn("Write a deterministic 5% evaluation split.", help_text)

    def test_vlm_presets_are_explicitly_unsupported(self):
        for dataset_name in prepare_data.UNSUPPORTED_VLM_DATASETS:
            with (
                self.subTest(dataset=dataset_name),
                self.assertRaisesRegex(ValueError, "VLM.*not supported"),
            ):
                prepare_data.load_dataset_preset(dataset_name)

    def test_parser_restores_eval_split_and_opc_subset(self):
        args = prepare_data.parse_args(
            [
                "--dataset",
                "opc",
                "--opc-subset",
                "all",
                "--split-eval",
            ]
        )

        self.assertEqual("all", args.opc_subset)
        self.assertTrue(args.split_eval)

    def test_custom_data_path_is_limited_to_sharegpt_json(self):
        for suffix in (".json", ".jsonl"):
            with self.subTest(suffix=suffix):
                args = prepare_data.parse_args(
                    [
                        "--dataset",
                        "sharegpt",
                        "--data-path",
                        f"custom{suffix}",
                    ]
                )
                self.assertEqual(Path(f"custom{suffix}"), args.data_path)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "ultrachat", "--data-path", "custom.jsonl"]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "sharegpt", "--data-path", "custom.csv"]
            )

    def test_custom_json_and_jsonl_use_the_hugging_face_json_loader(self):
        sentinel_dataset = object()
        for suffix in (".json", ".jsonl"):
            data_path = Path(f"custom{suffix}")
            with (
                self.subTest(suffix=suffix),
                patch.object(
                    prepare_data,
                    "_load_hf_dataset",
                    return_value=sentinel_dataset,
                ) as load_dataset,
            ):
                dataset = prepare_data.load_dataset_from_path(data_path)

                self.assertIs(sentinel_dataset, dataset)
                load_dataset.assert_called_once_with(
                    "json",
                    data_files=str(data_path),
                    split="train",
                )

    def test_sharegpt_conversion_normalizes_roles(self):
        row, skipped_count = prepare_data.process_sharegpt_row(
            {
                "id": 7,
                "conversations": [
                    {"from": "system", "value": "ignored"},
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )

        self.assertEqual(1, skipped_count)
        self.assertEqual(
            {
                "id": "7",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            row,
        )

    def test_math_and_code_presets_produce_conversation_rows(self):
        math_row, skipped = prepare_data.process_gsm8k_row(
            {"question": "1 + 1?", "answer": "2"}
        )
        code_row, code_skipped = prepare_data.process_codealpaca_row(
            {"instruction": "return one", "input": "", "output": "return 1"}
        )

        self.assertEqual(0, skipped)
        self.assertEqual(0, code_skipped)
        self.assertEqual("1 + 1?", math_row["conversations"][0]["content"])
        self.assertEqual("2", math_row["conversations"][1]["content"])
        self.assertEqual("return one", code_row["conversations"][0]["content"])
        self.assertEqual("return 1", code_row["conversations"][1]["content"])

    def test_codealpaca_conversion_keeps_the_non_empty_input(self):
        empty_input_row, _ = prepare_data.process_codealpaca_row(
            {
                "instruction": "Count the characters.",
                "input": "",
                "output": "12",
            }
        )
        filled_input_row, _ = prepare_data.process_codealpaca_row(
            {
                "instruction": "Count the characters.",
                "input": 'text = "Hello World!"',
                "output": "12",
            }
        )
        repeated_row, _ = prepare_data.process_codealpaca_row(
            {
                "instruction": "Count the characters.",
                "input": 'text = "Hello World!"',
                "output": "12",
            }
        )

        self.assertEqual(
            "Count the characters.",
            empty_input_row["conversations"][0]["content"],
        )
        self.assertEqual(
            'Count the characters.\n\ntext = "Hello World!"',
            filled_input_row["conversations"][0]["content"],
        )
        self.assertNotEqual(empty_input_row["id"], filled_input_row["id"])
        self.assertEqual(filled_input_row["id"], repeated_row["id"])

    def test_gsm8k_preset_dispatches_to_its_hosted_dataset(self):
        sentinel_dataset = object()
        with patch.object(
            prepare_data,
            "_train_split",
            return_value=sentinel_dataset,
        ) as load_train:
            dataset, processor = prepare_data.load_dataset_preset("gsm8k")

        self.assertIs(sentinel_dataset, dataset)
        self.assertIs(prepare_data.process_gsm8k_row, processor)
        load_train.assert_called_once_with("openai/gsm8k", "main")

    def test_conversion_writes_only_the_training_jsonl(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            output_path = prepare_data.process_and_save_dataset(
                [
                    {
                        "prompt_id": "prompt-1",
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                ],
                output_directory,
                prepare_data.process_ultrachat_row,
                "ultrachat",
            )

            self.assertEqual(output_directory / "ultrachat_train.jsonl", output_path)
            self.assertEqual([output_path], list(output_directory.iterdir()))
            self.assertEqual(
                {
                    "id": "prompt-1",
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ],
                },
                json.loads(output_path.read_text()),
            )

    def test_conversion_can_write_the_restored_eval_split(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            prepare_data.process_and_save_dataset(
                [
                    {
                        "prompt_id": "train",
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                ],
                output_directory,
                prepare_data.process_ultrachat_row,
                "ultrachat",
                eval_dataset=[
                    {
                        "prompt_id": "eval",
                        "messages": [
                            {"role": "user", "content": "eval question"},
                            {"role": "assistant", "content": "eval answer"},
                        ],
                    }
                ],
            )

            self.assertTrue((output_directory / "ultrachat_train.jsonl").exists())
            self.assertTrue((output_directory / "ultrachat_test.jsonl").exists())

    def test_mixture_ratios_and_exact_output(self):
        from datasets import Dataset

        source_rows = {
            name: Dataset.from_list(
                [
                    {
                        "id": str(i),
                        "conversations": [
                            {"role": "user", "content": f"{name} {i}"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                    for i in range(20)
                ]
            )
            for name in prepare_data.MIX_SOURCES
        }

        def load(name, *, streaming=False):
            self.assertTrue(streaming)
            return source_rows[name], lambda row, _: (dict(row), 0)

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(prepare_data, "load_dataset_preset", side_effect=load),
        ):
            root = Path(temporary_directory)
            path = prepare_data.prepare_mixture(
                "agent-aware", 13, (70, 10, 10, 10), root, seed=7
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(13, len(rows))
            self.assertEqual(
                [9, 2, 1, 1],
                [
                    sum(row["id"].startswith(name + ":") for row in rows)
                    for name in prepare_data.MIX_SOURCES
                ],
            )
            self.assertEqual(len(rows), len({row["id"] for row in rows}))
            self.assertEqual(
                rows, [json.loads(line) for line in path.read_text().splitlines()]
            )
            with self.assertRaises(FileExistsError):
                prepare_data.prepare_mixture("agent-aware", 13, (70, 10, 10, 10), root)

    def test_mixture_rejects_invalid_ratios_and_insufficient_source(self):
        with self.assertRaisesRegex(ValueError, "sum to 100"):
            prepare_data.parse_ratios("perfectblend-dedup=70,opencodeinstruct=20")
        with self.assertRaisesRegex(ValueError, "repeated"):
            prepare_data.parse_ratios("perfectblend-dedup=50,perfectblend-dedup=50")
        from datasets import Dataset

        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                prepare_data,
                "load_dataset_preset",
                return_value=(Dataset.from_list([{"id": "a"}]), None),
            ),
        ):
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "requested 2"):
                prepare_data.prepare_mixture("mix", 2, (100, 0, 0, 0), root)
            self.assertFalse(list(root.iterdir()))

    def test_mixture_discards_incomplete_dialogues_and_replenishes(self):
        from datasets import Dataset

        def dialogue(row_id, final):
            return {
                "id": row_id,
                "conversations": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                    final,
                ],
            }

        dataset = Dataset.from_list(
            [
                dialogue("tool-ended", {"role": "tool", "content": "Result"}),
                dialogue("empty", {"role": "assistant", "content": "   "}),
                {
                    "id": "good-1",
                    "conversations": [
                        {"role": "user", "content": "Q1"},
                        {"role": "assistant", "content": "A1"},
                    ],
                },
                {
                    "id": "good-2",
                    "conversations": [
                        {"role": "user", "content": "Q2"},
                        {"role": "assistant", "content": "A2"},
                    ],
                },
            ]
        )
        with (
            TemporaryDirectory() as temporary_directory,
            patch.object(
                prepare_data,
                "load_dataset_preset",
                return_value=(dataset, lambda row, _: (dict(row), 0)),
            ),
        ):
            path = prepare_data.prepare_mixture(
                "mix", 2, (100, 0, 0, 0), Path(temporary_directory), seed=1
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                {"perfectblend-dedup:good-1", "perfectblend-dedup:good-2"},
                {row["id"] for row in rows},
            )

    def test_agentic_converters_keep_tool_calls(self):
        apigen, _ = prepare_data.process_apigen_mt_row(
            {
                "system": "Use tools",
                "tools": '[{"name":"lookup"}]',
                "conversations": [
                    {"from": "human", "value": "Look up x"},
                    {
                        "from": "function_call",
                        "value": '{"name":"lookup","arguments":{"x":1}}',
                    },
                    {"from": "observation", "value": "found"},
                    {"from": "gpt", "value": "Done"},
                ],
            }
        )
        self.assertEqual(
            "lookup", apigen["conversations"][2]["tool_calls"][0]["function"]["name"]
        )
        self.assertEqual("call_1", apigen["conversations"][3]["tool_call_id"])
        normalize_row(apigen)
        nemotron, _ = prepare_data.process_nemotron_agentic_row(
            {
                "uuid": "u1",
                "tools": [],
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "ok", "tool_call_id": "c1"},
                    {"role": "assistant", "content": "Done"},
                ],
            }
        )
        self.assertEqual("c1", nemotron["conversations"][2]["tool_call_id"])
        normalize_row(nemotron)


if __name__ == "__main__":
    unittest.main(verbosity=2)
