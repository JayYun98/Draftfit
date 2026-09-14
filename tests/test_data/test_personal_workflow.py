"""Offline integration of reviewed mock conversations, not an OTel collector."""
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from specforge.benchmarks import sglang
from specforge.cli import main


class PersonalWorkflowTest(unittest.TestCase):
    def test_reviewed_mock_export_to_heldout_benchmark(self):
        # Independent conversations; this does NOT simulate session-aware splitting.
        rows = [{"messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": f"Explain function number {i}."},
            {"role": "assistant", "content": f"REFERENCE_ONLY_{i}"},
        ]} for i in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, train, holdout = (root / name for name in
                                      ("reviewed.jsonl", "train.jsonl", "holdout.jsonl"))
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "data", "prepare", "--input", str(source), "--output", str(train),
                    "--split-eval", "--eval-output", str(holdout),
                ]), 0)
            training = [json.loads(line) for line in train.read_text().splitlines()]
            evaluation = [json.loads(line) for line in holdout.read_text().splitlines()]
            self.assertEqual((len(training), len(evaluation)), (19, 1))
            self.assertFalse({row["id"] for row in training} &
                             {row["id"] for row in evaluation})
            args = SimpleNamespace(
                model="mock-target", trust_remote_code=False, dataset=None,
                messages_jsonl=str(holdout), max_samples=None, num_prompts=1,
                concurrency=1, enable_thinking=False, max_new_tokens=16,
                temperature=0.0, top_p=1.0, top_k=1,
                base_url="http://127.0.0.1:30000", timeout_seconds=1,
            )
            tokenizer = Mock()
            tokenizer.apply_chat_template.side_effect = lambda messages, **kw: json.dumps(messages)
            response = Mock()
            response.json.return_value = {"meta_info": {"completion_tokens": 2}}
            with (patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                  patch("requests.get", return_value=Mock()),
                  patch("requests.post", return_value=response) as send):
                result = sglang._run_sglang(args)
            self.assertEqual(send.call_count, 2)  # warmup + measured request
            for call in tokenizer.apply_chat_template.call_args_list:
                self.assertEqual(call.args[0], evaluation[0]["conversations"][:-1])
                self.assertNotIn("REFERENCE_ONLY", json.dumps(call.args[0]))
            self.assertEqual(result.metadata["source"]["sha256"],
                             hashlib.sha256(holdout.read_bytes()).hexdigest())
            self.assertEqual(result.output_tokens, 2)
            # Never use mocked latency or acceptance as a speedup result.
            self.assertIsNone(result.average_acceptance_length)


if __name__ == "__main__":
    unittest.main()
