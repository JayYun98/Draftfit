"""Completion contract tests; opt-in pinned tokenizer integration needs cache only."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from draftfit.data.humanizer import (
    MODEL_ID, PROMPT_FINGERPRINT, REVISION, load_prompt_format,
    prepare_jsonl, prepare_record, render_prompt,
)
from draftfit.data.prompt_builder import prepare_prompt_tasks

# Verbatim fields from the pinned upstream prompt_format.json (jialinyyzz/humanizer).
FORMAT = {
    "style": "legacy", "chat": False, "fingerprint": PROMPT_FINGERPRINT,
    "instr": "Rewrite the text below so it reads like a person wrote it, not a language model.\n\n"
    "Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,\n"
    "throat-clearing, and any sentence that only announces what comes next.\n"
    "Prefer the concrete word over the abstract one. It is fine to sound uneven.\n\n"
    "Every fact, number, unit, date, name and quotation must survive unchanged.",
    "sep": "\n\n### Rewritten:\n\n",
}


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=True, truncation=False,
                              return_offsets_mapping=True, return_special_tokens_mask=True)
        return {"input_ids": [1] + [ord(c) for c in text] + [2],
                "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)],
                "special_tokens_mask": [1] + [0] * len(text) + [1]}

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("chat rendering is forbidden")


class MergingTokenizer(CharacterTokenizer):
    """Merge the last prompt newline with the first rewrite character."""

    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        boundary = text.index(FORMAT["sep"]) + len(FORMAT["sep"])
        # BOS shifts indexes by one: boundary is the final prompt character.
        encoded["offset_mapping"][boundary] = (boundary - 1, boundary + 1)
        for field in encoded.values():
            del field[boundary + 1]
        return encoded


class HumanizerDataTests(unittest.TestCase):
    def test_official_prompt_and_tamper(self):
        prompt = render_prompt(" \nX\t", FORMAT)
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest()[:16], PROMPT_FINGERPRINT)
        for mutation in ({"instr": "wrong"}, {"sep": "wrong"}, {"chat": True}, {"fingerprint": "wrong"}):
            with self.assertRaises(ValueError):
                render_prompt("X", FORMAT | mutation)

    def test_masks_and_whitespace(self):
        rewrite = " \nHello 世界🙂\n"
        row = prepare_record({"draft": " X ", "rewrite": rewrite}, CharacterTokenizer(), FORMAT)
        self.assertEqual(sum(row["loss_mask"]), len(rewrite))
        self.assertEqual(row["loss_mask"][-1], 0)
        self.assertEqual(row["loss_mask"][:len(render_prompt("X", FORMAT)) + 1],
                         [0] * (len(render_prompt("X", FORMAT)) + 1))
        self.assertEqual("".join(chr(i) for i, mask in zip(row["input_ids"], row["loss_mask"]) if mask), rewrite)

    def test_boundary_merge_and_zero_supervision(self):
        row = prepare_record({"draft": "X", "rewrite": "abc"}, MergingTokenizer(), FORMAT)
        self.assertEqual(row["loss_mask"][-4:], [0, 1, 1, 0])
        with self.assertRaisesRegex(ValueError, "no wholly contained"):
            prepare_record({"draft": "X", "rewrite": "a"}, MergingTokenizer(), FORMAT)

    def test_invalid_rows_and_limits(self):
        tokenizer = CharacterTokenizer()
        for record in ({}, [], {"draft": "X"}, {"rewrite": "a"},
                       {"draft": " ", "rewrite": "a"}, {"draft": "X", "rewrite": "\n"},
                       {"draft": 2, "rewrite": "a"}, {"draft": "X", "rewrite": None}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                prepare_record(record, tokenizer, FORMAT)
        record = {"draft": "X", "rewrite": "abc"}
        length = len(prepare_record(record, tokenizer, FORMAT)["input_ids"])
        self.assertEqual(len(prepare_record(record, tokenizer, FORMAT, max_length=length)["input_ids"]), length)
        for limit in (length - 1, 0, True):
            with self.assertRaises(ValueError):
                prepare_record(record, tokenizer, FORMAT, max_length=limit)
        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            prepare_record(record, tokenizer, FORMAT)

    def test_atomic_output_and_existing_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.jsonl", Path(directory) / "out.jsonl"
            source.write_text(json.dumps({"draft": "X", "rewrite": "abc"}) + "\n")
            self.assertEqual(prepare_jsonl(source, output, CharacterTokenizer(), FORMAT), 1)
            row = json.loads(output.read_text())
            tasks = prepare_prompt_tasks(
                output, None, chat_template=None, max_length=8192, is_preformatted=True,
                train_only_last_turn=False, cache_dir=None, cache_key=None, num_proc=None,
            )
            self.assertEqual(tasks[0]["payload"], row)
            with self.assertRaises(FileExistsError):
                prepare_jsonl(source, output, CharacterTokenizer(), FORMAT)
            output.unlink()
            for content in ("", "\n", source.read_text() + '{"draft":"X"}\n', "null\n", "invalid\n"):
                source.write_text(content)
                with self.assertRaises(ValueError):
                    prepare_jsonl(source, output, CharacterTokenizer(), FORMAT)
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(directory).glob(".humanizer-*")), [])


@unittest.skipUnless(os.environ.get("HUMANIZER_TOKENIZER_TEST") == "1", "set HUMANIZER_TOKENIZER_TEST=1 for cached real tokenizer")
class OfficialTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        path = hf_hub_download(MODEL_ID, "prompt_format.json", revision=REVISION, local_files_only=True)
        cls.prompt_format = load_prompt_format(path)
        cls.tokenizer = AutoTokenizer.from_pretrained(str(Path(path).parent), local_files_only=True, trust_remote_code=False)

    def test_actual_tokenization_and_masks(self):
        for rewrite in ("A clearer version.", "\n\nA new start.", "中文改写，保留42。🙂", "  indented text\n", "\nHello"):
            with self.subTest(rewrite=rewrite):
                prompt = render_prompt("  Draft has 42. 中文。 ", self.prompt_format)
                encoded = self.tokenizer(prompt + rewrite, return_offsets_mapping=True, return_special_tokens_mask=True)
                row = prepare_record({"draft": "  Draft has 42. 中文。 ", "rewrite": rewrite}, self.tokenizer, self.prompt_format)
                self.assertEqual(row["input_ids"], encoded["input_ids"])
                self.assertTrue(any(row["loss_mask"]))
                for mask, (start, end), special in zip(row["loss_mask"], encoded["offset_mapping"], encoded["special_tokens_mask"]):
                    self.assertEqual(mask, int(not special and len(prompt) <= start < end))

    def test_real_boundary_merge(self):
        prompt = render_prompt("X", self.prompt_format)
        rewrite = "\nHello there."
        encoded = self.tokenizer(prompt + rewrite, return_offsets_mapping=True)
        crossings = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < len(prompt) < b]
        self.assertTrue(crossings, "fixture must reproduce an actual cross-boundary merge")
        row = prepare_record({"draft": "X", "rewrite": rewrite}, self.tokenizer, self.prompt_format)
        for index in crossings:
            self.assertEqual(row["loss_mask"][index], 0)

    def test_actual_cli_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "pairs.jsonl", Path(directory) / "tokens.jsonl"
            source.write_text(json.dumps({"draft": "The event was held on 12 May.", "rewrite": "The event took place on 12 May."}) + "\n")
            command = [sys.executable, "-m", "draftfit.data.humanizer", str(source), str(target), "--local-files-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            tasks = prepare_prompt_tasks(target, None, chat_template=None, max_length=8192,
                                         is_preformatted=True, train_only_last_turn=False,
                                         cache_dir=None, cache_key=None, num_proc=None)
            self.assertEqual(len(tasks), 1)
            self.assertGreater(sum(tasks[0]["payload"]["loss_mask"]), 1)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)


if __name__ == "__main__":
    unittest.main()
