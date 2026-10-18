"""Deterministic policy decisions from synthetic, explicitly bound observations."""

import json
import unittest
from copy import deepcopy

from draftfit.training.joint_policy import (
    decide,
    execute_registered_phase,
    observation_key,
)


def artifact(step, **extra):
    return {
        "artifact_sha256": f"{step:064x}",
        "family": "dspark",
        "preset": "base",
        "step": step,
        "training_fingerprint": "train-v1",
        "objective_fingerprint": "ce-l1-confidence",
        "capture_fingerprint": "teacher-taps-dtype-v1",
        **extra,
    }


def policy(**extra):
    return {
        "schema": "draftfit-joint-policy-v1",
        "revision": "test-v1",
        "domains": ["short", "long"],
        "comparison_fingerprint": "cuda-bf16-workload-v1",
        "patience": 2,
        "min_requests": 4,
        "next_steps": 10,
        "next_cost_usd": 0.02,
        "limits": {
            "max_usd": 1.0,
            "reserve_usd": 0.1,
            "max_steps": 1000,
            "max_pilot_usd": 0.1,
            "max_pilot_steps": 50,
            "max_exploration_usd": 0.2,
            "max_pilots": 2,
            "max_unvalidated_steps": 20,
        },
        **extra,
    }


def observation(step, loss=1.0, *, speed=None, acceptance=50):
    a = artifact(step)
    report = None
    if speed is not None:
        report = {k: a[k] for k in ("artifact_sha256", "family", "preset", "step")}
        report.update(
            comparison_fingerprint="cuda-bf16-workload-v1",
            split="validation",
            exact_token_parity=True,
            domains=[
                {
                    "name": name,
                    "requests": 2,
                    "matched_requests": 2,
                    "exact_token_parity": True,
                    "output_tokens": 100,
                    "baseline_output_tokens": 100,
                    "seconds": 10 / speed,
                    "baseline_seconds": 10.0,
                    "accepted_tokens": acceptance,
                    "proposed_tokens": 100,
                }
                for name in ("short", "long")
            ],
        )
    return {"artifact": a, "loss": loss, "serving": report}


def preset(name="head-only", **extra):
    return {
        "id": name,
        "family": "dspark",
        "training_fingerprint": "head-only-train",
        "objective_fingerprint": "confidence-only",
        "capture_fingerprint": "teacher-taps-dtype-v1",
        "when": ["loss_plateau"],
        "reason": "predeclared confidence head pilot",
        "pilot_steps": 20,
        "pilot_cost_usd": 0.05,
        "preflight_passed": True,
        "settings": {"head_only": True},
        **extra,
    }


class JointPolicyTests(unittest.TestCase):
    def decide(self, rows, *, config=None, budget=None, current=None, tried=()):
        return decide(
            config or policy(),
            rows,
            current=current or rows[-1]["artifact"],
            budget={} if budget is None else budget,
            tried_presets=tried,
        )

    def test_loss_small_improvements_accumulate_and_best_is_raw_minimum(self):
        rows = [
            observation(i, loss) for i, loss in enumerate((1.0, 0.9994, 0.9988, 0.9982))
        ]
        result = self.decide(rows, config=policy(patience=3))
        self.assertEqual(result["action"], "continue")
        self.assertEqual(result["plateau"]["loss"]["stale"], 1)
        self.assertEqual(result["plateau"]["loss"]["meaningful_best"], 0.9988)
        self.assertEqual(result["best"]["loss"]["value"], 0.9982)
        self.assertNotIn("speed", result["best"])
        self.assertFalse(result["current_serving_validated"])
        self.assertEqual(result["reason"], "loss_only_not_serving_validated")

    def test_plateau_counts_observations_not_step_distance(self):
        rows = [observation(i) for i in (0, 250, 1000)]
        result = self.decide(rows)
        self.assertEqual((result["action"], result["reason"]), ("stop", "loss_plateau"))
        self.assertEqual(result["plateau"]["loss"]["stale"], 2)

    def test_roundtrip_permutation_and_duplicate_reports_are_deterministic(self):
        rows = [observation(i, 1 - 0.1 * i, speed=1 + 0.1 * i) for i in range(3)]
        original = deepcopy(rows)
        first = self.decide(rows)
        self.assertEqual(
            first,
            self.decide(
                list(reversed(json.loads(json.dumps(rows)))), current=artifact(2)
            ),
        )
        repeated = self.decide(rows + [deepcopy(rows[-1])])
        for key in ("action", "reason", "best", "plateau"):
            self.assertEqual(first[key], repeated[key])
        self.assertEqual(original, rows)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.decide(rows + [observation(2, 0.2)])

    def test_acceptance_aggregates_actual_counts_not_average_percent_or_accuracy(self):
        row = observation(0, speed=2)
        row["serving"]["domains"][0].update(accepted_tokens=1, proposed_tokens=2)
        row["serving"]["domains"][1].update(accepted_tokens=90, proposed_tokens=100)
        row["serving"]["train_accuracy"] = 0.99
        result = self.decide([row])
        self.assertAlmostEqual(result["best"]["acceptance"]["value"], 91 / 102)
        self.assertEqual(result["best"]["speed"]["value"], 2)
        self.assertTrue(result["current_serving_validated"])

    def test_speed_aggregates_matched_time_and_reports_domain_floor(self):
        row = observation(0, speed=2)
        row["serving"]["domains"][1].update(seconds=20.0)
        result = self.decide([row], config=policy(min_domain_speedup=0.9))
        self.assertAlmostEqual(result["best"]["speed"]["value"], 20 / 25)
        self.assertEqual(result["reason"], "below_serving_threshold")

    def test_nonfinite_loss_timing_and_aggregate_overflow_are_rejected_with_json_receipt(
        self,
    ):
        for loss in (float("nan"), float("inf")):
            result = self.decide([observation(0, loss)])
            self.assertEqual(result["reason"], "current_loss_required")
            json.dumps(result, allow_nan=False)
        for seconds in (float("nan"), float("inf"), 5e-324, 1.7e308):
            with self.subTest(seconds=seconds):
                row = observation(0, speed=2)
                for domain in row["serving"]["domains"]:
                    domain["seconds"] = seconds
                result = self.decide([row])
                self.assertFalse(result["current_serving_validated"])
                self.assertTrue(result["rejected"])
                json.dumps(result, allow_nan=False)

    def test_speed_meaningful_baseline_accumulates_small_relative_gains(self):
        rows = [
            observation(i, 1 - 0.1 * i, speed=s)
            for i, s in enumerate((1.0, 1.006, 1.012, 1.018))
        ]
        result = self.decide(rows, config=policy(patience=3))
        self.assertEqual(result["plateau"]["speed"]["stale"], 1)
        self.assertAlmostEqual(result["plateau"]["speed"]["meaningful_best"], 1.012)

    def test_report_binding_and_runtime_fingerprints_cannot_substitute_current_weights(
        self,
    ):
        for field, value in (
            ("artifact_sha256", "f" * 64),
            ("family", "dflash"),
            ("preset", "different"),
            ("step", 20),
            ("comparison_fingerprint", "mlx-fp32-workload-v1"),
            ("split", "test"),
        ):
            with self.subTest(field=field):
                row = observation(10, speed=2)
                row["serving"][field] = value
                result = self.decide([row])
                self.assertFalse(result["current_serving_validated"])
                self.assertNotIn("speed", result["best"])
                self.assertTrue(result["rejected"])

    def test_family_preset_and_training_contract_mismatches_do_not_affect_patience(
        self,
    ):
        for field in (
            "family",
            "preset",
            "training_fingerprint",
            "objective_fingerprint",
            "capture_fingerprint",
        ):
            with self.subTest(field=field):
                old = observation(0)
                old["artifact"][field] = "another-contract"
                result = self.decide([old, observation(10)])
                self.assertEqual(result["plateau"]["loss"]["stale"], 0)
                self.assertEqual(len(result["rejected"]), 1)

    def test_same_step_different_artifact_and_future_loss_are_invalidated(self):
        for a in (artifact(10, artifact_sha256="f" * 64), artifact(11)):
            result = self.decide([{"artifact": a, "loss": 0.01}], current=artifact(10))
            self.assertEqual(result["reason"], "current_loss_required")
            self.assertTrue(result["rejected"])

    def test_late_report_remains_bound_to_old_artifact_and_wait_is_bounded(self):
        rows = [observation(0, speed=2), observation(10, 0.9)]
        cfg = policy(require_serving=True)
        result = self.decide(rows, config=cfg)
        self.assertFalse(result["current_serving_validated"])
        self.assertEqual(result["best"]["speed"]["artifact"]["step"], 0)
        self.assertIn(
            "late_report_for_current", [r["reason"] for r in result["rejected"]]
        )
        bounded = self.decide(rows, config=cfg, budget={"unvalidated_steps": 15})
        self.assertEqual(bounded["next_steps"], 5)
        blocked = self.decide(rows, config=cfg, budget={"unvalidated_steps": 20})
        self.assertEqual(blocked["reason"], "fresh_serving_required")

    def test_missing_duplicate_or_unmatched_domains_and_invalid_counts_cannot_claim_speed(
        self,
    ):
        def missing(report):
            report["domains"].pop()

        def duplicate(report):
            report["domains"][1]["name"] = "short"

        def unmatched(report):
            report["domains"][0]["matched_requests"] = 1

        def count(report):
            report["domains"][0]["accepted_tokens"] = 101

        def zero(report):
            report["domains"][0]["proposed_tokens"] = 0

        def proxy(report):
            report["domains"][0].pop("accepted_tokens")

        def fractional(report):
            report["domains"][0]["output_tokens"] = 1.5

        def parity(report):
            report["domains"][0]["exact_token_parity"] = False

        for change in (
            missing,
            duplicate,
            unmatched,
            count,
            zero,
            proxy,
            fractional,
            parity,
        ):
            with self.subTest(change=change.__name__):
                row = observation(0, speed=2)
                change(row["serving"])
                result = self.decide([row])
                self.assertNotIn("speed", result["best"])
                self.assertTrue(result["rejected"])

    def test_parity_failure_stops_before_branch_even_with_loss_improvement(self):
        rows = [observation(0), observation(1, 0.9, speed=3)]
        rows[-1]["serving"]["exact_token_parity"] = False
        result = self.decide(rows, config=policy(presets=[preset()]))
        self.assertEqual(
            (result["action"], result["reason"]), ("stop", "current_parity_failure")
        )

    def test_registered_branch_selection_is_ordered_preflighted_and_weights_only(self):
        choices = [
            preset("unapproved", preflight_passed=False),
            preset("head"),
            preset("dflash2", family="dflash2", objective_fingerprint="selector-ce"),
        ]
        result = self.decide(
            [observation(i) for i in range(3)], config=policy(presets=choices)
        )
        self.assertEqual(result["action"], "branch")
        self.assertEqual(result["allowed_presets"], ["head", "dflash2"])
        self.assertEqual(result["next_preset"], "head")
        self.assertEqual(result["optimizer_mode"], "weights_only")
        self.assertFalse(result["optimizer_migration_allowed"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["new_phase_required"])
        self.assertEqual(result["selection_reason"], choices[1]["reason"])
        again = self.decide(
            [observation(i) for i in range(3)],
            config=policy(presets=choices),
            tried=("head",),
        )
        self.assertEqual(again["next_preset"], "dflash2")

    def test_acceptance_plateau_and_speed_plateau_use_separate_measured_triggers(self):
        cfg = policy(
            presets=[
                preset(
                    when=[
                        "loss_improving_acceptance_plateau",
                        "acceptance_improving_speed_plateau",
                    ]
                )
            ]
        )
        rows = [observation(i, 1 - 0.1 * i, speed=1.2, acceptance=50) for i in range(3)]
        result = self.decide(rows, config=cfg)
        self.assertEqual(result["reason"], "loss_improving_acceptance_plateau")
        rows = [
            observation(i, 1 - 0.1 * i, speed=1.2, acceptance=50 + 5 * i)
            for i in range(3)
        ]
        result = self.decide(rows, config=cfg)
        self.assertEqual(result["reason"], "acceptance_improving_speed_plateau")

    def test_cumulative_pilot_exploration_and_step_bounds(self):
        rows = [observation(i) for i in range(3)]
        cfg = policy(presets=[preset()])
        cases = [
            ({"spent_usd": 0.95}, "cumulative_budget_exhausted"),
            ({"spent_usd": 0.85, "reserved_usd": 0.06}, "cumulative_budget_exhausted"),
            ({"steps_used": 1000}, "cumulative_budget_exhausted"),
            ({"pilots_used": 2}, "pilot_count_bound"),
            ({"steps_used": 990}, "pilot_step_bound"),
            ({"spent_usd": 0.88}, "pilot_cost_bound"),
            ({"exploration_spent_usd": 0.18}, "exploration_budget_bound"),
        ]
        for budget, reason in cases:
            with self.subTest(reason=reason):
                result = self.decide(rows, config=cfg, budget=budget)
                self.assertEqual(result["action"], "stop")
                self.assertTrue(
                    result["reason"] == reason
                    or reason in [p["reason"] for p in result["excluded_presets"]]
                )
        cfg["limits"]["max_pilot_usd"] = 0.04
        self.assertEqual(
            self.decide(rows, config=cfg)["excluded_presets"][0]["reason"],
            "pilot_cost_bound",
        )
        cfg["limits"]["max_pilot_steps"] = 19
        self.assertEqual(
            self.decide(rows, config=cfg)["excluded_presets"][0]["reason"],
            "pilot_step_bound",
        )

    def test_continuation_cost_includes_reservations_and_horizon_is_bounded(self):
        row = observation(10, 0.9)
        result = self.decide([row], budget={"spent_usd": 0.87, "reserved_usd": 0.02})
        self.assertEqual(result["reason"], "next_interval_exceeds_budget")
        result = self.decide([row], budget={"steps_used": 997})
        self.assertEqual(result["next_steps"], 3)

    def test_invalid_policy_and_migration_requests_are_rejected(self):
        for changed in (
            {"patience": True},
            {"min_delta": float("nan")},
            {"next_steps": 0},
            {"domains": ["short", "short"]},
            {"presets": [preset(optimizer_mode="resume")]},
            {"presets": [preset(), preset()]},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    self.decide([observation(0)], config=policy(**changed))
        with self.assertRaises(ValueError):
            observation_key(artifact(True))
        with self.assertRaises(ValueError):
            self.decide([observation(0)], budget={"spent_usd": float("nan")})


class RegisteredPhaseTests(unittest.TestCase):
    def setUp(self):
        self.config = policy(presets=[preset()])
        self.decision = decide(
            self.config,
            [observation(i) for i in range(3)],
            current=artifact(2),
            budget={},
        )
        self.calls = []

    def executor(self, plan):
        self.calls.append(plan)
        return {
            "initialization": "weights_only",
            "optimizer_migrated": False,
            "steps": plan["max_steps"],
            "cost_usd": plan["max_cost_usd"],
        }

    def test_handler_actually_invokes_registered_new_phase_executor(self):
        result = execute_registered_phase(
            self.config,
            self.decision,
            budget={},
            artifact_sha256=artifact(2)["artifact_sha256"],
            executor=self.executor,
        )
        self.assertTrue(result["executed"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["optimizer_mode"], "weights_only")
        self.assertEqual(self.calls[0]["source_artifact"], artifact(2))
        self.assertTrue(self.calls[0]["new_phase_required"])

    def test_real_cpu_weights_only_pilot_has_fresh_optimizer_and_preserves_source(self):
        # Actual CPU phase callback, not a serving/GPU correctness claim.
        import torch

        torch.manual_seed(123)
        source = torch.nn.Linear(2, 1)
        original = {k: v.detach().clone() for k, v in source.state_dict().items()}
        x, y = torch.ones(3, 2), torch.zeros(3, 1)
        trained = []

        def run_phase(plan):
            candidate = torch.nn.Linear(2, 1)
            candidate.load_state_dict(source.state_dict())
            optimizer = torch.optim.Adam(candidate.parameters(), lr=0.01)
            self.assertEqual(len(optimizer.state), 0)
            for _ in range(plan["max_steps"]):
                optimizer.zero_grad()
                loss = (candidate(x) - y).square().mean()
                loss.backward()
                optimizer.step()
            trained.append(candidate)
            return {
                "initialization": "weights_only",
                "optimizer_migrated": False,
                "steps": plan["max_steps"],
                "cost_usd": 0.0,
            }

        result = execute_registered_phase(
            self.config,
            self.decision,
            budget={},
            artifact_sha256=artifact(2)["artifact_sha256"],
            executor=run_phase,
        )
        self.assertTrue(result["executed"])
        self.assertTrue(
            any(
                not torch.equal(v, trained[0].state_dict()[k])
                for k, v in original.items()
            )
        )
        self.assertTrue(
            all(torch.equal(v, source.state_dict()[k]) for k, v in original.items())
        )

    def test_handler_rechecks_hash_registration_and_live_budget_before_invocation(self):
        for change in ("hash", "budget", "registration", "tried"):
            with self.subTest(change=change):
                cfg = deepcopy(self.config)
                if change == "registration":
                    cfg["presets"][0]["settings"]["head_only"] = False
                with self.assertRaises(ValueError):
                    execute_registered_phase(
                        cfg,
                        self.decision,
                        budget={"spent_usd": 0.99} if change == "budget" else {},
                        artifact_sha256=(
                            "f" * 64
                            if change == "hash"
                            else artifact(2)["artifact_sha256"]
                        ),
                        executor=self.executor,
                        tried_presets=("head-only",) if change == "tried" else (),
                    )
                self.assertEqual(self.calls, [])

    def test_handler_does_not_claim_success_if_executor_migrates_or_overruns(self):
        for wrong in ({"optimizer_migrated": True}, {"steps": 21}, {"cost_usd": 0.051}):
            with self.subTest(wrong=wrong):

                def bad(plan):
                    return {**self.executor(plan), **wrong}

                with self.assertRaisesRegex(ValueError, "violated"):
                    execute_registered_phase(
                        self.config,
                        self.decision,
                        budget={},
                        artifact_sha256=artifact(2)["artifact_sha256"],
                        executor=bad,
                    )


if __name__ == "__main__":
    unittest.main()
