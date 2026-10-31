"""Synthetic schema checks only: these fixtures are not performance evidence."""

import contextlib
import copy
import io
import json
import subprocess
import sys
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


def runtime():
    """Synthetic declarations only; no real runtime or artifact evidence."""
    return {
        "target_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "draft_revision": "none",
        "backend_version": "sglang synthetic build",
        "hardware": "synthetic CPU fixture, not GPU evidence",
        "common_server_config_sha256": "c" * 64,
    }


class CompareTest(unittest.TestCase):
    def test_legacy_reports_are_explicitly_weak_evidence(self):
        result = compare([report()], [report(200)])
        self.assertEqual(result["evidence_strength"], "weak")
        self.assertEqual(result["runtime_identity"], "not_recorded")
        self.assertEqual(result["rendered_prompt_identity"], "not_recorded")
        self.assertEqual(result["status"], "throughput_comparison_only")
        self.assertEqual(result["production_gate"], "not_evaluated")

    def test_builtin_workload_and_rendered_prompt_identity(self):
        baseline = report()
        baseline["dataset"] = "gsm8k"
        baseline["metadata"]["source"].update(kind="builtin", name="gsm8k")
        baseline["metadata"]["rendered_prompts_sha256"] = "f" * 64
        candidate = copy.deepcopy(baseline)
        self.assertEqual(compare([baseline], [candidate])["throughput_ratio"], 1)
        for mutate in (
            lambda r: r["metadata"].update(rendered_prompts_sha256="e" * 64),
            lambda r: r["metadata"].pop("rendered_prompts_sha256"),
            lambda r: r["metadata"]["source"].pop("sha256"),
            lambda r: r["metadata"]["source"].update(name="other"),
            lambda r: r.update(dataset="other"),
        ):
            changed = copy.deepcopy(candidate)
            mutate(changed)
            with self.assertRaises(ValueError):
                compare([baseline], [changed])
        baseline["metadata"].pop("rendered_prompts_sha256")
        with self.assertRaises(ValueError):
            compare([baseline], [baseline])

    def test_runtime_matching_and_candidate_artifact_identity(self):
        baseline, candidate = report(), report(200)
        for item in (baseline, candidate):
            item["metadata"].update(runtime=runtime(), rendered_prompts_sha256="f" * 64)
        candidate["metadata"]["runtime"]["draft_revision"] = "d" * 64
        result = compare([baseline, baseline], [candidate, candidate])
        self.assertEqual(result["runtime_identity"], "operator_recorded_match")
        self.assertEqual(result["rendered_prompt_identity"], "recorded_match")
        self.assertEqual(result["evidence_strength"], "operator_recorded")
        self.assertEqual(result["production_gate"], "not_evaluated")
        self.assertTrue(
            any("not independently verified" in s for s in result["limitations"])
        )
        for key, value in (
            ("target_revision", "e" * 40),
            ("tokenizer_revision", "e" * 40),
            ("hardware", "different hardware"),
            ("backend_version", "different build"),
            ("common_server_config_sha256", "e" * 64),
        ):
            changed = copy.deepcopy(candidate)
            changed["metadata"]["runtime"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare([baseline], [changed])
        for side in ("baseline", "candidate"):
            changed = copy.deepcopy(baseline if side == "baseline" else candidate)
            changed["metadata"]["runtime"]["draft_revision"] = "e" * 64
            with (
                self.subTest(side=side),
                self.assertRaisesRegex(ValueError, "within each side"),
            ):
                compare(
                    [baseline, changed] if side == "baseline" else [baseline],
                    [candidate, changed] if side == "candidate" else [candidate],
                )

    def test_optional_identities_cannot_be_dropped_on_one_side_or_run(self):
        for key, value in (
            ("runtime", runtime()),
            ("rendered_prompts_sha256", "f" * 64),
        ):
            identified = report()
            identified["metadata"][key] = value
            for baseline, candidate in (
                ([report()], [identified]),
                ([identified], [report()]),
                ([identified, report()], [identified]),
                ([identified], [identified, report()]),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    compare(baseline, candidate)
            result = compare([identified], [identified])
            self.assertEqual(result["evidence_strength"], "weak")

    def test_malformed_optional_identities_reject_even_when_both_sides_match(self):
        for value in (None, False, {}, "", "f" * 40, "F" * 64, "g" * 64):
            changed = report()
            changed["metadata"]["rendered_prompts_sha256"] = value
            with self.subTest(prompt=value), self.assertRaises(ValueError):
                compare([changed], [changed])
        bad_runtimes = [None, False, [], {}, {**runtime(), "extra": "field"}]
        for key, value in (
            ("target_revision", "main"),
            ("target_revision", "none"),
            ("tokenizer_revision", "f" * 64),
            ("tokenizer_revision", "latest"),
            ("draft_revision", "release-v1"),
            ("draft_revision", ""),
            ("hardware", "  "),
            ("backend_version", 1),
            ("common_server_config_sha256", "f" * 40),
            ("common_server_config_sha256", None),
        ):
            bad_runtimes.append({**runtime(), key: value})
        for value in bad_runtimes:
            changed = report()
            changed["metadata"]["runtime"] = value
            with self.subTest(runtime=value), self.assertRaises(ValueError):
                compare([changed], [changed])

    def test_actual_compare_cli_reports_weak_evidence_and_rejects_one_sided_identity(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            candidate = Path(directory) / "candidate.json"
            baseline.write_text(json.dumps(report()), encoding="utf-8")
            candidate.write_text(json.dumps(report(200)), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "draftfit.benchmarks.compare",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["evidence_strength"], "weak")
            identified = report(200)
            identified["metadata"]["runtime"] = runtime()
            candidate.write_text(json.dumps(identified), encoding="utf-8")
            rejected = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("comparison rejected", rejected.stderr)
            baseline.write_text(json.dumps(identified), encoding="utf-8")
            accepted = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(
                json.loads(accepted.stdout)["runtime_identity"],
                "operator_recorded_match",
            )

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
