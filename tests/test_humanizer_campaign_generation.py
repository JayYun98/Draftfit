"""CPU-only checks for bounded generation decisions and resumable publication."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.generate_humanizer_campaign_pairs import (
    MAX_LENGTH, assess_completion, estimate, read_inputs, run,
)
from draftfit.data.humanizer import PROMPT_FINGERPRINT
from tests.test_humanizer_data import FORMAT, CharacterTokenizer


class MockTokenizer(CharacterTokenizer):
    eos_token_id = 2
    pad_token_id = 2
    eos_token = '<eos>'
    pad_token = '<eos>'
    all_special_ids = [1, 2]
    padding_side = 'right'

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids)


class FakeConfig:
    model_type = 'gemma4_unified'

    def get_text_config(self):
        return argparse.Namespace(vocab_size=100000)


class FakeModel:
    config = FakeConfig()


class BatchTokenizer(MockTokenizer):
    def encode(self, text, add_special_tokens=False):
        return [1, 3] if add_special_tokens else super().encode(text, add_special_tokens=False)

    def __call__(self, text, **kwargs):
        class Tensor:
            shape = (2, 2)
            def to(self, _): return self
        if isinstance(text, list):
            return {'input_ids': Tensor(), 'attention_mask': Tensor()}
        return super().__call__(text, **kwargs)


class BatchModel(FakeModel):
    device = 'cpu'

    def __init__(self, on_generate=None):
        self.on_generate = on_generate

    def generate(self, **kwargs):
        if self.on_generate:
            self.on_generate()
        class Output:
            def __getitem__(self, _): return self
            def tolist(self): return [[65, 2], [65, 2]]
        return Output()


def fake_torch():
    return patch.dict('sys.modules', {'torch': SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None))})


class GenerationTests(unittest.TestCase):
    def test_filters_eos_empty_invalid_and_length(self):
        tok = MockTokenizer()
        row = {'id': 'a', 'draft': 'X', 'source': 'fixture'}
        for ids, reason in [([], 'invalid_tokens'), ([65], 'no_eos'),
                            ([2], 'invalid_tokens'), ([1, 65, 2], 'invalid_tokens'),
                            ([100001, 2], 'invalid_tokens')]:
            self.assertEqual(assess_completion(row, ids, tok, FORMAT, eos_token_id=2,
                                               vocab_size=100000)[1], reason)
        result, reason = assess_completion(row, [65, 66, 2, 2], tok, FORMAT,
                                           eos_token_id=2, vocab_size=100000)
        self.assertIsNone(reason)
        self.assertEqual(result['pair']['rewrite'], 'AB')
        self.assertEqual(result['pair']['source'], 'fixture')
        self.assertTrue(any(result['tokens']['loss_mask']))
        self.assertLessEqual(len(result['tokens']['input_ids']), MAX_LENGTH)
        _, reason = assess_completion(row, [65] * 300 + [2], tok, FORMAT,
                                      eos_token_id=2, vocab_size=100000)
        self.assertEqual(reason, 'over_512')

    def test_duplicate_input_and_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'train.inputs.jsonl'
            p.write_text('{"id":"x","draft":"a"}\n{"id":"x","draft":"b"}\n')
            with self.assertRaisesRegex(ValueError, 'duplicate id'):
                read_inputs(p)
        metric = estimate([{'status': 'accepted'}] * 3, 6, 4, 30, 1.5,
                          warmup_generation_seconds=2, warmup_accepted=1)
        self.assertEqual(metric['accepted_per_second'], 1.0)
        self.assertEqual(metric['projected_remaining_seconds'], 27)
        self.assertAlmostEqual(metric['projected_total_usd'], .01375)
        unknown = estimate([{'status': 'rejected'}], 5, 0, 30, 1.5)
        self.assertIsNone(unknown['projected_total_usd'])
        json.dumps(unknown, allow_nan=False)

    def test_pilot_resume_and_exact_final(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'train.inputs.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'draft': 'X', 'source': 's'}) + '\n'
                                      for i in range(3)))
            args = argparse.Namespace(input=source, output_dir=root / 'out', required_count=2,
                                      batch_size=1, max_batches=1, deadline_unix=None,
                                      hourly_usd=1, max_projected_usd=3, local_files_only=True)
            tok = MockTokenizer()
            def generate(*_, **__):
                return [[65, 2]]
            with fake_torch():
                # The fake tensor container supplies only the access used by the runner.
                class Tensor:
                    shape = (1, 2)
                    def to(self, _): return self
                class Output:
                    def __getitem__(self, _):
                        return self
                    def tolist(self): return [[65, 2]]
                tok.__call__ = lambda *a, **k: None  # instance override does not affect special method
                class BatchTokenizer(MockTokenizer):
                    def encode(self, text, add_special_tokens=False):
                        if add_special_tokens:
                            return [1, 3]
                        return super().encode(text, add_special_tokens=False)

                    def __call__(self, text, **kwargs):
                        if isinstance(text, list): return {'input_ids': Tensor(), 'attention_mask': Tensor()}
                        return super().__call__(text, **kwargs)
                class Model(FakeModel):
                    device = 'cpu'
                    generate = lambda self, **kwargs: Output()
                tok = BatchTokenizer()
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=Model()), 3)
                self.assertFalse((args.output_dir / 'train.tokens.jsonl').exists())
                self.assertEqual(len((args.output_dir / 'train.partial.jsonl').read_text().splitlines()), 1)
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=Model()), 0)
            tokens = [json.loads(x) for x in (args.output_dir / 'train.tokens.jsonl').read_text().splitlines()]
            pairs = [json.loads(x) for x in (args.output_dir / 'train.pairs.jsonl').read_text().splitlines()]
            manifest = json.loads((args.output_dir / 'train.manifest.json').read_text())
            self.assertEqual(len(tokens), 2)
            self.assertEqual([p['id'] for p in pairs], ['0', '1'])
            self.assertEqual(manifest['count'], 2)
            self.assertEqual(manifest['provenance'][0]['source'], 's')
            self.assertIn('exact RNG replay is not guaranteed', manifest['sampling_resume'])

    def test_staged_accepted_threshold_resumes_without_duplicate_or_early_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'train.inputs.jsonl'
            original = ''.join(json.dumps({'id': str(i), 'draft': 'X'}) + '\n' for i in range(6))
            source.write_text(original)
            args = argparse.Namespace(input=source, output_dir=root / 'out', required_count=5,
                                      batch_size=2, max_batches=None, stop_after_accepted=2,
                                      stop_request_file=None, deadline_unix=None, hourly_usd=1,
                                      max_projected_usd=1e9, local_files_only=True)
            tok, model = BatchTokenizer(), BatchModel()
            journal = args.output_dir / 'train.partial.jsonl'
            progress = args.output_dir / 'train.progress.json'
            with fake_torch():
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=model), 2)
                self.assertEqual(json.loads(progress.read_text())['required_count'], 5)
                self.assertEqual(json.loads(progress.read_text())['projection_target_accepted'], 2)
                self.assertEqual(json.loads(progress.read_text())['pause_reason'], 'stop_after_accepted')
                initial_input_sha = json.loads(progress.read_text())['input_sha256']
                self.assertEqual(json.loads(progress.read_text())['prompt_fingerprint'], PROMPT_FINGERPRINT)
                self.assertEqual(len(journal.read_text().splitlines()), 2)
                self.assertFalse((args.output_dir / 'train.manifest.json').exists())
                self.assertFalse((args.output_dir / 'train.tokens.jsonl').exists())
                first = journal.read_bytes()
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=model), 2)
                self.assertEqual(journal.read_bytes(), first)
                source.write_text(original.replace('"draft": "X"', '"draft": "Y"', 1))
                with self.assertRaisesRegex(ValueError, 'resume input or required count differs'):
                    run(args, tokenizer=tok, prompt_format=FORMAT, model=model)
                source.write_text(original)
                args.required_count = 4
                with self.assertRaisesRegex(ValueError, 'resume input or required count differs'):
                    run(args, tokenizer=tok, prompt_format=FORMAT, model=model)
                args.required_count = 5
                args.stop_after_accepted = 3
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=model), 2)
                self.assertEqual(len(journal.read_text().splitlines()), 4)
                self.assertTrue(journal.read_bytes().startswith(first))
                self.assertEqual(json.loads(progress.read_text())['projection_target_accepted'], 3)
                self.assertEqual(json.loads(progress.read_text())['input_sha256'], initial_input_sha)
                self.assertEqual(json.loads(progress.read_text())['required_count'], 5)
                self.assertFalse((args.output_dir / 'train.manifest.json').exists())
                args.stop_after_accepted = None
                self.assertEqual(run(args, tokenizer=tok, prompt_format=FORMAT, model=model), 0)
            entries = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([entry['id'] for entry in entries], ['0', '1', '2', '3', '4'])
            self.assertEqual(len({entry['id'] for entry in entries}), 5)
            manifest = json.loads((args.output_dir / 'train.manifest.json').read_text())
            self.assertEqual(manifest['count'], 5)
            self.assertEqual(manifest['input_sha256'], initial_input_sha)
            self.assertEqual(manifest['prompt_fingerprint'], PROMPT_FINGERPRINT)

    def test_live_stop_file_waits_for_persisted_batch_then_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'train.inputs.jsonl'
            source.write_text(''.join(json.dumps({'id': str(i), 'draft': 'X'}) + '\n' for i in range(4)))
            marker = root / 'stop-now'
            args = argparse.Namespace(input=source, output_dir=root / 'out', required_count=3,
                                      batch_size=2, max_batches=None, stop_after_accepted=None,
                                      stop_request_file=marker, deadline_unix=None, hourly_usd=1,
                                      max_projected_usd=1e9, local_files_only=True)
            with fake_torch():
                self.assertEqual(run(args, tokenizer=BatchTokenizer(), prompt_format=FORMAT,
                                     model=BatchModel(lambda: marker.touch())), 2)
                self.assertEqual(len((args.output_dir / 'train.partial.jsonl').read_text().splitlines()), 2)
                self.assertEqual(json.loads((args.output_dir / 'train.progress.json').read_text())['processed'], 2)
                self.assertEqual(json.loads((args.output_dir / 'train.progress.json').read_text())['pause_reason'], 'stop_requested')
                self.assertFalse((args.output_dir / 'train.manifest.json').exists())
                marker.unlink()
                self.assertEqual(run(args, tokenizer=BatchTokenizer(), prompt_format=FORMAT,
                                     model=BatchModel()), 0)
            entries = [json.loads(line) for line in (args.output_dir / 'train.partial.jsonl').read_text().splitlines()]
            self.assertEqual([entry['id'] for entry in entries], ['0', '1', '2'])
