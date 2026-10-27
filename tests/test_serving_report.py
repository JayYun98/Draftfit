"""CPU replay of recorded SGLang output; no server, network or fake CUDA.

Token IDs, seconds and acceptance below are copied from main checkout's
.planning/unattended-e2e-a100-20261017/evidence/serving-retry/receipt.json
(target, dspark, ce; selector failed and is deliberately not a passing arm).
See docs/reports/unattended-e2e-2026-10-17.md in main. The old smoke receipt
does NOT contain the new independent bindings: those are explicit synthetic
test metadata, never retroactive certification of the recorded GPU run.
"""

import json
import unittest
from copy import deepcopy

from draftfit.training.joint_policy import decide, fingerprint
from draftfit.training.serving_report import (
    ACCEPTANCE_SCOPE,
    SCHEMA,
    TIMING_SCOPE,
    normalize_serving_report,
)
from tests.test_joint_policy import artifact, policy

RECORDED = {
    "chat": (
        [8, 220, 122895, 33, 285, 5390, 8471, 559],
        0.08306900039315224,
        0.1780275907367468,
        0.10561742447316647,
    ),
    "code": (
        [8, 220, 1741, 457, 2231, 317, 4900, 280],
        0.08346084877848625,
        0.17738572880625725,
        0.10889239609241486,
    ),
    "math": (
        [8, 220, 1741, 457, 2925, 47, 458, 2928],
        0.07994505204260349,
        0.17707658000290394,
        0.10942981205880642,
    ),
}


def fixture():
    domains = list(RECORDED)
    bindings = dict(
        target=dict(
            model="MiniCPM5-1B",
            revision="87179e5c1f455ef22e6223592d2d61351b525bfc",
            weights_sha256=fingerprint("test target"),
            tokenizer_sha256=fingerprint("test tokenizer"),
        ),
        runtime=dict(
            engine="sglang",
            version="0.5.19",
            build_sha256=fingerprint("test runtime"),
            device="A100 SXM4 80GB",
            dtype="bfloat16",
        ),
        validation_data_sha256=fingerprint("test held-out data"),
        config=dict(
            temperature=0,
            max_new_tokens=8,
            trials=1,
            concurrency=1,
            warmup_requests=1,
            cache_policy="disabled",
        ),
        prompts=[
            dict(
                row_index=i,
                domain=d,
                prompt_sha256=fingerprint(["test rendered prompt", d]),
            )
            for i, d in enumerate(domains)
        ],
    )
    artifacts = {
        name: artifact(
            10,
            family=family,
            checkpoint_sha256=fingerprint([name, "checkpoint"]),
            export_sha256=fingerprint([name, "export"]),
        )
        for name, family in (("dspark", "dspark"), ("ce", "dflash"))
    }
    comparison = fingerprint(bindings)

    def arm(name, timing_index, proposed):
        rows = [
            dict(
                trial=0,
                **p,
                token_ids=RECORDED[p["domain"]][0][:],
                generated_tokens=8,
                seconds=RECORDED[p["domain"]][timing_index],
                accepted_tokens=0,
                proposed_tokens=proposed,
            )
            for p in bindings["prompts"]
        ]
        result = dict(
            name=name, status="complete", comparison_fingerprint=comparison, runs=rows
        )
        if name in artifacts:
            result["artifact"] = deepcopy(artifacts[name])
        return result

    report = dict(
        schema=SCHEMA,
        status="complete",
        split="validation",
        production_engine_speed=True,
        speed_decisions_allowed=False,
        timing_scope=TIMING_SCOPE,
        acceptance_scope=ACCEPTANCE_SCOPE,
        bindings=deepcopy(bindings),
        comparison_fingerprint=comparison,
        baseline=arm("target-only", 1, 0),
        candidates=[arm("dspark", 2, 28), arm("ce", 3, 21)],
    )
    return report, dict(
        artifacts=artifacts, bindings=bindings, domains=domains, min_requests=3
    )


class ServingReportTests(unittest.TestCase):
    def test_recorded_measurements_roundtrip_into_real_policy(self):
        report, kwargs = fixture()
        before = deepcopy((report, kwargs))
        normalized = normalize_serving_report(json.loads(json.dumps(report)), **kwargs)
        self.assertEqual((report, kwargs), before)
        self.assertEqual(normalized["source_report_sha256"], fingerprint(report))
        for serving in normalized["candidates"]:
            current = kwargs["artifacts"][serving["name"]]
            cfg = policy(
                domains=kwargs["domains"],
                min_requests=3,
                comparison_fingerprint=normalized["comparison_fingerprint"],
            )
            decision = decide(
                cfg,
                [dict(artifact=current, loss=1.0, serving=serving)],
                current=current,
                budget={},
            )
            self.assertTrue(decision["current_serving_validated"])
            self.assertFalse(decision["rejected"])
            self.assertNotIn(
                "speed", decision["best"]
            )  # Recorded smoke is not a speed gate.
            self.assertEqual(decision["best"]["acceptance"]["value"], 0.0)
            self.assertEqual(sum(d["output_tokens"] for d in serving["domains"]), 24)
        normalized["baseline"]["runs"][0]["token_ids"][0] = 999
        self.assertEqual((report, kwargs), before)

    def test_speed_uses_measured_sum_and_requires_explicit_permission(self):
        report, kwargs = fixture()
        report["speed_decisions_allowed"] = True
        self.assertFalse(
            normalize_serving_report(report, **kwargs)["candidates"][0][
                "speed_decisions_allowed"
            ]
        )
        normalized = normalize_serving_report(
            report, allow_speed_decisions=True, **kwargs
        )
        serving = next(c for c in normalized["candidates"] if c["name"] == "dspark")
        current = kwargs["artifacts"]["dspark"]
        decision = decide(
            policy(
                domains=kwargs["domains"],
                min_requests=3,
                comparison_fingerprint=normalized["comparison_fingerprint"],
            ),
            [dict(artifact=current, loss=1.0, serving=serving)],
            current=current,
            budget={},
        )
        self.assertAlmostEqual(
            decision["best"]["speed"]["value"],
            sum(r[1] for r in RECORDED.values()) / sum(r[2] for r in RECORDED.values()),
        )
        self.assertLess(decision["best"]["speed"]["value"], 1.0)

    def test_stale_checkpoint_export_and_training_contracts_rejected(self):
        for field in (
            "artifact_sha256",
            "checkpoint_sha256",
            "export_sha256",
            "step",
            "family",
            "preset",
            "training_fingerprint",
            "objective_fingerprint",
            "capture_fingerprint",
        ):
            with self.subTest(field=field):
                report, kwargs = fixture()
                report["candidates"][0]["artifact"][field] = (
                    11 if field == "step" else "f" * 64
                )
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    normalize_serving_report(report, **kwargs)

    def test_actual_runtime_target_data_and_prompt_binding_are_independent(self):
        for part, field, value in (
            ("runtime", "version", "fake"),
            ("runtime", "dtype", "float32"),
            ("runtime", "build_sha256", "f" * 64),
            ("target", "revision", "stale"),
            ("target", "tokenizer_sha256", "f" * 64),
            ("config", "max_new_tokens", 9),
        ):
            with self.subTest(part=part, field=field):
                report, kwargs = fixture()
                report["bindings"][part][field] = value
                report["comparison_fingerprint"] = fingerprint(report["bindings"])
                with self.assertRaisesRegex(ValueError, "comparison mismatch"):
                    normalize_serving_report(report, **kwargs)
        report, kwargs = fixture()
        report["bindings"]["validation_data_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            normalize_serving_report(report, **kwargs)

    def test_reference_cannot_be_relabelled_production(self):
        changes = [
            dict(schema="draftfit-joint-generation-v1", production_engine_speed=True),
            dict(production_engine_speed=False),
            dict(preset="greedy-full-prefix-reference"),
            dict(timing_scope="full_generation_including_prefill_no_kv_cache"),
        ]
        for change in changes:
            with self.subTest(change=change):
                report, kwargs = fixture()
                report.update(change)
                with self.assertRaises(ValueError):
                    normalize_serving_report(report, **kwargs)
        for location in ("baseline", "candidate", "row"):
            report, kwargs = fixture()
            obj = (
                report["baseline"]
                if location == "baseline"
                else report["candidates"][0]
            )
            if location == "row":
                obj = obj["runs"][0]
            obj["production_engine_speed"] = False
            with self.assertRaisesRegex(ValueError, "reference"):
                normalize_serving_report(report, **kwargs)
        report, kwargs = fixture()
        kwargs["bindings"]["runtime"]["engine"] = "native-reference"
        report["bindings"] = deepcopy(kwargs["bindings"])
        with self.assertRaisesRegex(ValueError, "engine"):
            normalize_serving_report(report, **kwargs)

    def test_baseline_and_all_expected_candidates_required(self):
        for change in (
            lambda r: r.pop("baseline"),
            lambda r: r.update(baseline=None),
            lambda r: r["candidates"].pop(),
            lambda r: r["candidates"].append(deepcopy(r["candidates"][0])),
            lambda r: r["candidates"][0].update(name="selector", status="failed"),
            lambda r: r["baseline"].update(name="existing-draft"),
            lambda r: r["baseline"].update(status="failed"),
            lambda r: r.update(exact_token_parity=False),
            lambda r: r["baseline"].update(tokenparity=False),
            lambda r: r["candidates"][0]["runs"][0].update(status="failed"),
            lambda r: r["candidates"][0].update(comparison_fingerprint="f" * 64),
        ):
            report, kwargs = fixture()
            change(report)
            with self.assertRaises(ValueError):
                normalize_serving_report(report, **kwargs)

    def test_missing_duplicate_and_empty_domains_rejected(self):
        for domains in (
            [],
            ["chat", "chat"],
            ["chat", "code"],
            ["chat", "code", "math", "empty"],
            [""],
        ):
            with self.subTest(domains=domains):
                report, kwargs = fixture()
                kwargs["domains"] = domains
                with self.assertRaises(ValueError):
                    normalize_serving_report(report, **kwargs)
        report, kwargs = fixture()
        kwargs["bindings"]["prompts"] = []
        with self.assertRaises(ValueError):
            normalize_serving_report(report, **kwargs)

    def test_request_pairing_is_complete_not_zip_or_positional(self):
        for side in ("baseline", "candidate"):
            for change in (
                lambda a: a["runs"].pop(),
                lambda a: a["runs"].__setitem__(1, deepcopy(a["runs"][0])),
                lambda a: a["runs"][0].update(trial=1),
                lambda a: a["runs"][0].update(row_index=9),
                lambda a: a["runs"][0].update(domain="code"),
                lambda a: a["runs"][0].update(prompt_sha256="f" * 64),
            ):
                with self.subTest(side=side, change=change):
                    report, kwargs = fixture()
                    change(
                        report["baseline"]
                        if side == "baseline"
                        else report["candidates"][0]
                    )
                    with self.assertRaises(ValueError):
                        normalize_serving_report(report, **kwargs)
        report, kwargs = fixture()
        for c in report["candidates"]:
            c["runs"].reverse()
        self.assertEqual(
            len(normalize_serving_report(report, **kwargs)["candidates"]), 2
        )

    def test_exact_ids_counts_and_explicit_failure_flags(self):
        for change in (
            dict(token_ids=[999] * 8),
            dict(generated_tokens=7),
            dict(token_ids=[]),
            dict(token_ids=[True] * 8),
            dict(tokenparity=False),
            dict(exact_token_parity=False),
        ):
            with self.subTest(change=change):
                report, kwargs = fixture()
                report["candidates"][0]["runs"][0].update(change)
                with self.assertRaises(ValueError):
                    normalize_serving_report(report, **kwargs)

    def test_invalid_numbers_cannot_enter_policy(self):
        cases = {
            "seconds": [
                0,
                -1,
                float("nan"),
                float("inf"),
                True,
                "0.1",
                5e-324,
                1.7e308,
            ],
            "generated_tokens": [True, 8.0, -1],
            "accepted_tokens": [-1, 29, 9, 0.1, True],
            "proposed_tokens": [-1, 0, 0.1, True],
            "trial": [False, 0.5],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    report, kwargs = fixture()
                    for row in report["candidates"][0]["runs"]:
                        row[field] = value
                    with self.assertRaises(ValueError):
                        normalize_serving_report(report, **kwargs)

    def test_acceptance_is_counts_never_rate_or_training_accuracy(self):
        report, kwargs = fixture()
        for row in report["candidates"][0]["runs"]:
            row.pop("accepted_tokens")
            row.update(acceptance_rate=0.9, training_accuracy=0.99)
        with self.assertRaises(ValueError):
            normalize_serving_report(report, **kwargs)

    def test_request_thresholds_are_not_lowered_to_fit_smoke(self):
        report, kwargs = fixture()
        kwargs.pop("min_requests")
        with self.assertRaisesRegex(ValueError, "insufficient_requests"):
            normalize_serving_report(report, **kwargs)
        kwargs.update(min_requests=3, min_domain_requests=2)
        with self.assertRaisesRegex(ValueError, "insufficient_domain_requests"):
            normalize_serving_report(report, **kwargs)

    def test_existing_draft_is_separate_bound_and_optional(self):
        report, kwargs = fixture()
        self.assertIsNone(normalize_serving_report(report, **kwargs)["existing_draft"])
        existing = deepcopy(report["candidates"][0])
        existing.update(name="prior-draft")
        existing["artifact"].update(
            step=1, artifact_sha256="1" * 64, export_sha256="2" * 64
        )
        report["existing_draft"] = existing
        with self.assertRaisesRegex(ValueError, "existing_draft"):
            normalize_serving_report(report, **kwargs)
        kwargs["existing_draft"] = deepcopy(existing["artifact"])
        normalized = normalize_serving_report(report, **kwargs)
        self.assertEqual(normalized["existing_draft"]["step"], 1)
        self.assertEqual(
            {c["name"] for c in normalized["candidates"]}, {"ce", "dspark"}
        )
        report["existing_draft"]["artifact"]["export_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            normalize_serving_report(report, **kwargs)
        report.pop("existing_draft")
        with self.assertRaisesRegex(ValueError, "existing_draft"):
            normalize_serving_report(report, **kwargs)

    def test_multiple_trials_require_every_prompt(self):
        report, kwargs = fixture()
        kwargs["bindings"]["config"]["trials"] = 2
        report["bindings"] = deepcopy(kwargs["bindings"])
        comparison = fingerprint(kwargs["bindings"])
        report["comparison_fingerprint"] = comparison
        for arm in [report["baseline"], *report["candidates"]]:
            arm["comparison_fingerprint"] = comparison
            arm["runs"] += [dict(deepcopy(r), trial=1) for r in arm["runs"]]
        normalized = normalize_serving_report(report, **kwargs)
        self.assertEqual(normalized["candidates"][0]["domains"][0]["requests"], 2)
        report["candidates"][0]["runs"][-1]["trial"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            normalize_serving_report(report, **kwargs)

    def test_malformed_json_structures_fail_with_value_error(self):
        for report in (None, [], {}, {"schema": SCHEMA}, "report"):
            _, kwargs = fixture()
            with self.subTest(report=report), self.assertRaises(ValueError):
                normalize_serving_report(report, **kwargs)


if __name__ == "__main__":
    unittest.main()
