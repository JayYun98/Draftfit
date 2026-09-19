"""Synthetic schema checks only: these fixtures are not performance evidence."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from draftfit.benchmarks.compare import compare, main


def report(rate=100):
    return {
        "backend": "sglang",
        "dataset": "messages-jsonl",
        "samples": 2,
        "output_tokens": 100,
        "latency_seconds": 100 / rate,
        "throughput_tokens_per_second": rate,
        "metadata": {
            "model": "target",
            "tokenizer": "target",
            "seed": 42,
            "source": {"kind": "messages_jsonl", "sha256": "a" * 64, "sample_count": 2},
            "concurrency": 1,
            "warmup_prompts": 1,
            "num_prompts_requested": 2,
            "max_samples": None,
            "trust_remote_code": False,
            "timeout_seconds": 60,
            "generation": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 100,
                "enable_thinking": False,
            },
        },
    }


class CompareTest(unittest.TestCase):
    def test_median_and_no_certification(self):
        result = compare([report(50), report(100), report(150)], [report(200)])
        self.assertEqual(result["throughput_ratio"], 2)
        self.assertEqual(result["production_gate"], "not_evaluated")
        self.assertEqual(result["baseline_runs"], 3)

    def test_rejects_bad_measurements(self):
        for key, value in [
            ("samples", True),
            ("output_tokens", 0),
            ("latency_seconds", -1),
            ("latency_seconds", float("inf")),
            ("output_tokens", 10**1000),
            ("throughput_tokens_per_second", float("nan")),
            ("throughput_tokens_per_second", 200),
            ("spec_verify_count", 1.5),
        ]:
            with self.subTest(key=key, value=value):
                candidate = report()
                candidate[key] = value
                with self.assertRaises(ValueError):
                    compare([report()], [candidate])

    def test_missing_and_incompatible_evidence(self):
        mutations = [
            lambda r: r["metadata"].pop("generation"),
            lambda r: r["metadata"]["source"].pop("sha256"),
            lambda r: r["metadata"]["source"].update(sha256="b" * 64),
            lambda r: r["metadata"].update(concurrency=2),
            lambda r: r["metadata"].update(model="other"),
            lambda r: r["metadata"].update(num_prompts_requested=3),
            lambda r: r["metadata"]["generation"].update(temperature=0.5),
            lambda r: r["metadata"]["generation"].update(top_p=float("nan")),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(report())
            mutate(candidate)
            with self.assertRaises(ValueError):
                compare([report()], [candidate])
        with self.assertRaises(ValueError):
            compare([], [report()])
        for malformed in (None, [], {}, {"metadata": None}):
            with self.assertRaises(ValueError):
                compare([report()], [malformed])

    def test_cli_rejects_missing_file_and_duplicate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.json"
            path.write_text(json.dumps(report()))
            for candidate in (str(path), str(path) + ".missing"):
                with self.assertRaises(SystemExit) as caught:
                    main(["--baseline", str(path), "--candidate", candidate])
                self.assertEqual(caught.exception.code, 2)

    def test_cli_output_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            candidate = Path(directory) / "candidate.json"
            baseline.write_text(json.dumps(report()))
            candidate.write_text(json.dumps(report(80)))
            args = ["--baseline", str(baseline), "--candidate", str(candidate)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.getvalue())["throughput_ratio"], 0.8)
            candidate.write_text("{")
            with self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 2)

    def test_cli_rejects_oversized_report(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            candidate = Path(directory) / "candidate.json"
            baseline.write_text(json.dumps(report()))
            candidate.write_bytes(b" " * (1024 * 1024 + 1))
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                main(["--baseline", str(baseline), "--candidate", str(candidate)])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
