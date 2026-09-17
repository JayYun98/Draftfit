import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dspark.benchmarks import sglang


class SGLangBenchmarkTest(unittest.TestCase):
    def test_output_is_reserved_before_inference_and_preserved_on_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text("baseline")
            with mock.patch.object(sglang, "_run_sglang") as run:
                for output in (path, Path(directory) / "missing" / "result.json"):
                    with self.assertRaises(OSError):
                        sglang.run(SimpleNamespace(output_json=str(output)))
                run.assert_not_called()
            self.assertEqual(path.read_text(), "baseline")

    def test_failed_measurement_or_write_cleans_only_owned_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            args = SimpleNamespace(output_json=str(path))
            result = sglang.BenchmarkResult("sglang", "test", 1, 1, 1, 1)
            with mock.patch.object(
                sglang, "_run_sglang", side_effect=RuntimeError("server")
            ):
                with self.assertRaises(RuntimeError):
                    sglang.run(args)
            self.assertFalse(path.exists())
            with (
                mock.patch.object(sglang, "_run_sglang", return_value=result),
                mock.patch.object(sglang.os, "fsync", side_effect=OSError("disk")),
            ):
                with self.assertRaises(OSError):
                    sglang.run(args)
            self.assertFalse(path.exists())

            def replace_then_fail(_args):
                path.unlink()
                path.write_text("another writer")
                raise RuntimeError("server")

            with mock.patch.object(
                sglang, "_run_sglang", side_effect=replace_then_fail
            ):
                with self.assertRaises(RuntimeError):
                    sglang.run(args)
            self.assertEqual(path.read_text(), "another writer")

    def test_output_success_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = sglang.BenchmarkResult("sglang", "test", 1, 1, 1, 1)
            with (
                mock.patch.object(sglang, "_run_sglang", return_value=result),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(sglang.run(SimpleNamespace(output_json=str(path))), 0)
            self.assertEqual(json.loads(path.read_text())["output_tokens"], 1)

    def test_mt_bench_prompts_preserve_turns(self):
        rows = [{"prompt": ["first", "second"]}]
        with mock.patch("datasets.load_dataset", return_value=rows):
            prompts = sglang._load_prompts("mt-bench", max_samples=None)
        self.assertEqual(prompts, [["first", "second"]])

    def test_prompt_loader_rejects_empty_inputs(self):
        with mock.patch("datasets.load_dataset", return_value=[]):
            with self.assertRaisesRegex(ValueError, "did not contain any prompts"):
                sglang._load_prompts("gsm8k", max_samples=None)
        with self.assertRaisesRegex(ValueError, "--max-samples must be positive"):
            sglang._load_prompts("gsm8k", max_samples=0)

    def test_messages_jsonl_drops_answer_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "held-out.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "first"},
                            {"role": "assistant", "content": "history"},
                            {"role": "user", "content": "follow-up"},
                            {"role": "assistant", "content": "held-out answer"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            prompts = sglang._load_messages_jsonl(path, max_samples=None)

        self.assertEqual(
            prompts,
            [
                [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "history"},
                    {"role": "user", "content": "follow-up"},
                ]
            ],
        )

    def test_messages_jsonl_rejects_non_openai_rows_before_server(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"from": "human", "value": "question"},
                            {"from": "gpt", "value": "answer"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 1"):
                sglang._load_messages_jsonl(path, max_samples=None)

    def test_messages_jsonl_rejects_unsupported_schema_before_tokenizer(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "tools": [{"type": "function"}],
            },
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "history",
                        "tool_calls": [{"type": "function"}],
                    },
                    {"role": "user", "content": "follow-up"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "history",
                        "function_call": {"name": "lookup"},
                    },
                    {"role": "user", "content": "follow-up"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "system": "rules kept outside messages",
            },
        ]
        args = SimpleNamespace(
            model="/models/Inkling",
            trust_remote_code=False,
            dataset=None,
            messages_jsonl=None,
            max_samples=None,
            num_prompts=1,
            concurrency=1,
            enable_thinking=False,
            base_url="http://127.0.0.1:30000",
            timeout_seconds=30,
        )

        with tempfile.TemporaryDirectory() as directory:
            for row in rows:
                path = Path(directory) / "bad.jsonl"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                args.messages_jsonl = str(path)
                with (
                    mock.patch(
                        "transformers.AutoTokenizer.from_pretrained"
                    ) as tokenizer,
                    mock.patch("requests.get") as get,
                    mock.patch("requests.post") as post,
                ):
                    with self.assertRaisesRegex(ValueError, "line 1"):
                        sglang._run_sglang(args)
                tokenizer.assert_not_called()
                get.assert_not_called()
                post.assert_not_called()

    def test_local_messages_jsonl_mocked_server_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "held-out.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "first"},
                            {"role": "assistant", "content": "history"},
                            {"role": "user", "content": "follow-up"},
                            {"role": "assistant", "content": "do not prompt this"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            args = SimpleNamespace(
                model="/models/Inkling",
                trust_remote_code=False,
                dataset=None,
                messages_jsonl=str(path),
                max_samples=None,
                num_prompts=1,
                concurrency=1,
                enable_thinking=False,
                max_new_tokens=16,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                base_url="http://127.0.0.1:30000",
                timeout_seconds=30,
            )
            rendered = []

            class FakeTokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    rendered.append(messages)
                    return "rendered"

            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "meta_info": {
                    "completion_tokens": 2,
                    "spec_verify_ct": 1,
                    "spec_accept_length": 3.0,
                }
            }
            flush_response = mock.Mock()
            flush_response.raise_for_status.return_value = None
            with (
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=FakeTokenizer(),
                ),
                mock.patch("requests.get", return_value=flush_response),
                mock.patch("requests.post", return_value=response) as post,
            ):
                result = sglang._run_sglang(args)

        self.assertEqual(post.call_count, 2)  # one warmup plus one measured request
        self.assertEqual(rendered[0][-1]["content"], "follow-up")
        self.assertNotIn(
            "do not prompt this", [message["content"] for message in rendered[0]]
        )
        self.assertEqual(result.dataset, "messages-jsonl")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.metadata["source"]["kind"], "messages_jsonl")
        self.assertEqual(
            result.metadata["source"]["sha256"],
            source_sha256,
        )
        self.assertEqual(result.metadata["generation"]["max_new_tokens"], 16)

    def test_sglang_path_excludes_warmup_from_reported_totals(self):
        args = SimpleNamespace(
            model="thinkingmachines/Inkling",
            trust_remote_code=False,
            dataset="gsm8k",
            max_samples=None,
            num_prompts=3,
            concurrency=2,
            enable_thinking=False,
            base_url="http://127.0.0.1:30000",
            timeout_seconds=30,
        )
        response = {
            "meta_info": {
                "completion_tokens": 2,
                "spec_verify_ct": 1,
                "spec_accept_length": 3.0,
            }
        }
        flush_response = mock.Mock()
        flush_response.raise_for_status.return_value = None
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(sglang, "_load_prompts", return_value=[["prompt"]]),
            mock.patch.object(sglang, "_apply_chat_template", return_value="rendered"),
            mock.patch.object(sglang, "_send_sglang", return_value=response) as send,
            mock.patch("requests.get", return_value=flush_response) as flush,
        ):
            result = sglang._run_sglang(args)

        self.assertEqual(send.call_count, args.num_prompts + args.concurrency)
        self.assertEqual(result.samples, 3)
        self.assertEqual(result.output_tokens, 6)
        self.assertEqual(result.spec_verify_count, 3)
        self.assertEqual(result.average_acceptance_length, 3.0)
        flush.assert_called_once_with(
            "http://127.0.0.1:30000/flush_cache",
            timeout=30,
        )

    def test_shared_cli_dispatches_sglang_benchmark(self):
        from dspark.cli import main

        with mock.patch.object(sglang, "run", return_value=7) as run:
            status = main(
                [
                    "benchmark",
                    "--model",
                    "thinkingmachines/Inkling",
                    "--dataset",
                    "gsm8k",
                ]
            )

        self.assertEqual(status, 7)
        self.assertEqual(run.call_args.args[0].model, "thinkingmachines/Inkling")

    def test_cli_help_describes_the_backend_not_an_algorithm(self):
        from dspark.cli import main

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exited:
            main(["benchmark", "--help"])

        self.assertEqual(exited.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("a running SGLang server", help_text)
        self.assertNotIn("DSpark", help_text)
        self.assertNotIn("DFlash", help_text)


if __name__ == "__main__":
    unittest.main()
