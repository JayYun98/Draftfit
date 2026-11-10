"""Offline local-target generation, row commits, and safe continuation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from draftfit.cli import main
from draftfit.data.regenerate import parse_arguments


class LocalRegenerationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="local-regeneration-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "target"
        vocabulary = {
            word: index
            for index, word in enumerate(
                [
                    "[PAD]",
                    "[EOS]",
                    "[UNK]",
                    "system",
                    "user",
                    "assistant",
                    "hello",
                    "world",
                    "again",
                    "reply",
                ]
            )
        }
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        )
        tokenizer.save_pretrained(self.model)
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)
        torch.manual_seed(7)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=128,
                pad_token_id=0,
                eos_token_id=None,
                bos_token_id=None,
            )
        )
        model.generation_config.suppress_tokens = [0, 1, 2]
        model.save_pretrained(self.model)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.rows = [
            dict(
                id=str(index),
                conversations=[
                    dict(role="user", content="hello world"),
                    dict(role="assistant", content="OLD RESPONSE"),
                    dict(role="user", content="again"),
                ],
            )
            for index in range(2)
        ]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def arguments(self, *extra):
        return [
            "data",
            "regenerate",
            "--backend",
            "transformers",
            "--model",
            str(self.model),
            "--input-file-path",
            str(self.input),
            "--output-file-path",
            str(self.output),
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--local-files-only",
            "--batch-size",
            "2",
            "--max-tokens",
            "2",
            *extra,
        ]

    def test_actual_cli_multiturn_generation_and_resume(self):
        self.assertEqual(main(self.arguments()), 0)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["0", "1"])
        for row in rows:
            self.assertEqual(
                [m["role"] for m in row["conversations"]],
                ["user", "assistant", "user", "assistant"],
            )
            self.assertNotIn("OLD RESPONSE", str(row["conversations"]))
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["regeneration"]["backend"], "transformers")
        before = self.output.read_bytes()
        with mock.patch(
            "draftfit.data.local_regenerate._load_target",
            side_effect=AssertionError("already done"),
        ):
            self.assertEqual(main(self.arguments("--resume")), 0)
        self.assertEqual(before, self.output.read_bytes())

    def test_resume_changed_settings_or_input_fails_before_writes(self):
        self.assertEqual(main(self.arguments()), 0)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "contract differs"):
            main(self.arguments("--resume", "--temperature", "0.5"))
        self.assertEqual(before, self.output.read_bytes())
        self.input.write_text(self.input.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "contract differs"):
            main(self.arguments("--resume"))
        self.assertEqual(before, self.output.read_bytes())

    def test_derived_output_cannot_overwrite_input(self):
        original = self.input.read_bytes()
        self.output = self.root / "input.jsonl"
        self.input = self.root / "input_error.jsonl"
        self.input.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "separate"):
            main(self.arguments())
        self.assertEqual(self.input.read_bytes(), original)

    def test_resume_repairs_only_incomplete_tail(self):
        self.assertEqual(main(self.arguments("--num-samples", "1")), 0)
        committed = self.output.read_bytes()
        with self.output.open("ab") as handle:
            handle.write(b'{"id": "unfinished')
        self.assertEqual(main(self.arguments("--resume")), 0)
        self.assertTrue(self.output.read_bytes().startswith(committed))
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["0", "1"])

    def test_resume_rejects_corrupt_complete_record_without_truncation(self):
        self.assertEqual(main(self.arguments("--num-samples", "1")), 0)
        with self.output.open("ab") as handle:
            handle.write(b"{broken}\n")
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "corrupt completed"):
            main(self.arguments("--resume"))
        self.assertEqual(self.output.read_bytes(), before)

    def test_resume_preserves_valid_unterminated_record(self):
        self.assertEqual(main(self.arguments("--num-samples", "1")), 0)
        self.output.write_bytes(self.output.read_bytes().rstrip(b"\n"))
        self.assertEqual(main(self.arguments("--resume")), 0)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["0", "1"])

    def test_model_failure_does_not_commit_pending_rows(self):
        with mock.patch(
            "draftfit.data.generation_batching.AdaptiveGenerationBatcher.generate",
            side_effect=RuntimeError("native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                main(self.arguments())
        self.assertEqual(self.output.read_text(), "")
        progress = json.loads(self.output.with_suffix(".progress.json").read_text())
        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["committed_this_run"]["success"], 0)

    def test_expired_deadline_never_loads_target(self):
        with mock.patch(
            "draftfit.data.local_regenerate._load_target",
            side_effect=AssertionError("expired"),
        ):
            with self.assertRaises(TimeoutError):
                main(self.arguments("--deadline-unix", "1"))
        self.assertEqual(self.output.read_text(), "")

    def test_server_default_and_local_option_boundary(self):
        args = parse_arguments(self.arguments()[2:])
        self.assertEqual(args.backend, "transformers")
        server = parse_arguments(
            [
                "--model",
                "unused",
                "--input-file-path",
                str(self.input),
                "--output-file-path",
                str(self.output),
            ]
        )
        self.assertEqual(server.backend, "server")
        with self.assertRaisesRegex(ValueError, "auto-batch requires"):
            main(
                [
                    "data",
                    "regenerate",
                    "--model",
                    "unused",
                    "--auto-batch",
                    "--input-file-path",
                    str(self.input),
                    "--output-file-path",
                    str(self.output),
                ]
            )

    def test_partial_completed_row_resume_has_no_duplicates(self):
        self.assertEqual(main(self.arguments("--num-samples", "1")), 0)
        self.assertEqual(main(self.arguments("--resume")), 0)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
