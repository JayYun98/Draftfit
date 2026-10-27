"""Real, CPU-only optional Optuna/SQLite controller and CLI checks."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from draftfit.training.search import (
    SearchError,
    SearchStudy,
    digest,
    load_json,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def contract(family="dflash"):
    return {
        "schema_version": 1,
        "study_name": "bounded-test",
        "identity": {
            "target": {
                "model_id": "fixture/target",
                "revision": "f" * 40,
                "weights_sha256": sha("target"),
            },
            "draft": {
                "config_sha256": sha("draft-config"),
                "checkpoint_sha256": sha("base-checkpoint"),
            },
            "data": {
                "train_sha256": sha("train"),
                "validation_sha256": sha("heldout"),
                "split_sha256": sha("split-lineage"),
            },
            "objective": {
                "family": family,
                "name": "heldout_objective_loss",
                "config_sha256": sha(family),
                "validation_protocol_sha256": sha("full-heldout-fixed-seeds"),
            },
            "base_config_sha256": sha("base-config"),
            "code_revision": "a" * 40,
        },
        "space": {
            "training.learning_rate": {
                "kind": "float",
                "low": 1e-5,
                "high": 1e-3,
                "log": True,
            },
            "training.lr_scheduler": {
                "kind": "categorical",
                "choices": ["constant", "cosine"],
            },
            "training.objective_chunk_blocks": {
                "kind": "int",
                "low": 8,
                "high": 16,
                "strategies": ["dflash", "dflash2"],
            },
            "training.optimizer_cpu_offload": {
                "kind": "categorical",
                "choices": [True],
            },
        },
        "budget": {
            "max_trials": 4,
            "max_total_updates": 16,
            "max_total_seconds": None,
            "trial_max_updates": 4,
            "trial_max_seconds": 3600,
        },
        "memory": {"max_trial_mib": 6000, "reserve_mib": 1024},
        "sampler": {"seed": 42, "startup_trials": 1},
        "pruner": {
            "startup_trials": 1,
            "warmup_updates": 0,
            "interval_updates": 1,
            "min_trials": 1,
        },
        "report_updates": [2, 4],
    }


def receipt(c, proposal, updates=2, loss=1.0, parent=None):
    return {
        "contract_digest": digest(c),
        "config_digest": proposal["config_digest"],
        "checkpoint_sha256": sha(f"trial-{proposal['trial']}-update-{updates}"),
        "parent_checkpoint_sha256": parent
        or c["identity"]["draft"]["checkpoint_sha256"],
        "updates": updates,
        "elapsed_seconds": float(updates),
        "validation_loss": loss,
        "peak_memory_mib": 4000,
        "free_memory_mib": 2048,
    }


class TrainingSearchContractTests(unittest.TestCase):
    def test_strict_json_and_immutable_safe_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "{"):
                path.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(SearchError):
                    load_json(path)
        for name in (
            "training.loss_type",
            "training.batch_size",
            "training.resume_from",
            "model.path",
            "__import__('os').system('false')",
        ):
            c = contract()
            c["space"][name] = {"kind": "categorical", "choices": ["unsafe"]}
            with self.subTest(name=name), self.assertRaises(SearchError):
                validate_contract(c)
        c = contract()
        validated = validate_contract(c)
        c["memory"]["max_trial_mib"] = 1
        self.assertEqual(validated["memory"]["max_trial_mib"], 6000)

    def test_contract_bounds_types_and_finite_values(self):
        mutations = [
            lambda c: c.update(schema_version=True),
            lambda c: c["budget"].update(max_total_updates=None),
            lambda c: c["budget"].update(max_trials=True),
            lambda c: c["budget"].update(trial_max_seconds=float("inf")),
            lambda c: c["memory"].update(reserve_mib=0),
            lambda c: c["space"]["training.learning_rate"].update(low=float("nan")),
            lambda c: c["space"]["training.learning_rate"].update(log="true"),
            lambda c: c["space"]["training.objective_chunk_blocks"].update(low=True),
            lambda c: c["space"]["training.objective_chunk_blocks"].update(
                strategies=["shell"]
            ),
            lambda c: c["space"]["training.optimizer_cpu_offload"].update(choices=[1]),
            lambda c: c.update(report_updates=[4, 2]),
            lambda c: c["identity"]["data"].update(
                validation_sha256=c["identity"]["data"]["train_sha256"]
            ),
            lambda c: c["identity"].update(code_revision="a" * 41),
        ]
        for mutate in mutations:
            c = contract()
            mutate(c)
            with self.subTest(contract=c), self.assertRaises(SearchError):
                validate_contract(c)

    def test_import_help_and_contract_validation_without_optional_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract()))
            base = [sys.executable, "-S", "-m", "draftfit.training.search"]
            for args in (["--help"], ["validate", "--contract", str(path)]):
                result = subprocess.run(
                    base + args, cwd=ROOT, text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                base
                + [
                    "status",
                    "--contract",
                    str(path),
                    "--storage",
                    str(Path(directory) / "study.sqlite"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "Optional dependency missing", json.loads(result.stderr)["error"]
            )
            self.assertFalse((Path(directory) / "study.sqlite").exists())
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; import draftfit.training.search; assert 'optuna' not in sys.modules; assert 'torch' not in sys.modules",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(
    importlib.util.find_spec("optuna"),
    "Install the optional search extra for real SQLite tests",
)
class TrainingSearchSQLiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "study.sqlite"
        self.c = contract()

    def controller(self):
        return SearchStudy(self.path, self.c)

    def ask(self):
        return self.controller().ask(free_memory_mib=7577.6, estimated_memory_mib=5000)

    def finish(self, proposal, loss=1.0):
        r1 = receipt(self.c, proposal, loss=loss)
        self.controller().report(proposal["trial"], r1)
        r2 = receipt(self.c, proposal, 4, loss, r1["checkpoint_sha256"])
        self.controller().report(proposal["trial"], r2)
        return self.controller().tell(proposal["trial"], state="complete")

    def frozen(self):
        import optuna

        storage = optuna.storages.RDBStorage(f"sqlite:///{self.path}")
        try:
            return optuna.load_study(
                storage=storage, study_name=self.c["study_name"]
            ).trials
        finally:
            storage.remove_session()
            storage.engine.dispose()

    def cli(self, command, *args, success=True):
        cp = self.directory / "contract.json"
        cp.write_text(json.dumps(self.c))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "draftfit.training.search",
                command,
                "--contract",
                str(cp),
                "--storage",
                str(self.path),
                *map(str, args),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0 if success else 2, result.stderr)
        return json.loads(result.stdout if success else result.stderr)

    def test_cli_across_real_process_restarts_and_optuna_final_value(self):
        proposal = self.cli(
            "ask", "--free-memory-mib", 7577.6, "--estimated-memory-mib", 5000
        )
        self.assertEqual(proposal["metric_role"], "screening_only")
        self.assertEqual(proposal["state"], "running")
        r1 = receipt(self.c, proposal, loss=0.5)
        r2 = receipt(self.c, proposal, 4, 0.75, r1["checkpoint_sha256"])
        for r in (r1, r2):
            rp = self.directory / "report.json"
            rp.write_text(json.dumps(r))
            result = self.cli("report", "--trial", proposal["trial"], "--receipt", rp)
            self.assertEqual(result["state"], "running")
        self.assertEqual(
            self.cli("tell", "--trial", 0, "--state", "complete")["state"], "complete"
        )
        status = self.cli("status")
        self.assertEqual(status["best_screening_trial"], 0)
        self.assertEqual(
            status["reserved"], {"trials": 1, "updates": 4, "seconds": 3600}
        )
        frozen = self.frozen()[0]
        self.assertEqual(frozen.state.name, "COMPLETE")
        self.assertEqual(frozen.value, 0.75)  # Fixed final checkpoint, not best loss.
        self.assertEqual(frozen.intermediate_values, {1: 0.5, 3: 0.75})
        self.assertEqual(frozen.params, proposal["params"])
        self.assertFalse(
            any(k in status for k in ("throughput", "acceptance", "serving_passed"))
        )

    def test_same_family_median_pruning_is_durable(self):
        self.finish(self.ask(), loss=1)
        second = self.ask()
        result = self.controller().report(1, receipt(self.c, second, loss=10))
        self.assertEqual(result["state"], "pruned")
        self.assertEqual(self.cli("status")["trials"][1]["state"], "pruned")
        self.assertEqual(self.frozen()[1].state.name, "PRUNED")
        self.assertEqual(self.controller().status()["reserved"]["updates"], 8)
        with self.assertRaises(SearchError):
            self.controller().tell(1, state="complete")

    def test_conditional_parameters_and_restart_seed_policy(self):
        for family in ("dspark", "dflash", "dflash2"):
            with self.subTest(family=family):
                self.path = self.directory / f"{family}.sqlite"
                self.c = contract(family)
                first = self.ask()
                self.assertEqual(
                    "training.objective_chunk_blocks" in first["params"],
                    family != "dspark",
                )
                self.assertIs(first["params"]["training.optimizer_cpu_offload"], True)
                self.controller().tell(
                    0, state="interrupted", reason="Operator stopped at boundary"
                )
                second = self.ask()
                self.assertNotEqual(
                    first["params"]["training.learning_rate"],
                    second["params"]["training.learning_rate"],
                )
                self.assertNotEqual(first["config_digest"], second["config_digest"])

    def test_fixed_seed_config_uses_real_enqueue_and_normal_receipt_protocol(self):
        params = {
            "training.learning_rate": 0.0002,
            "training.lr_scheduler": "constant",
            "training.objective_chunk_blocks": 12,
            "training.optimizer_cpu_offload": True,
        }
        path = self.directory / "seed-params.json"
        path.write_text(json.dumps(params))
        proposal = self.cli(
            "ask",
            "--free-memory-mib",
            7577.6,
            "--estimated-memory-mib",
            5000,
            "--params",
            path,
        )
        self.assertEqual(proposal["params"], params)
        self.assertEqual(proposal["limits"]["max_updates"], 4)
        self.assertLessEqual(
            proposal["limits"]["deadline"], proposal["started_at"] + 3600
        )
        self.finish(proposal)
        self.assertEqual(self.frozen()[0].params, params)
        self.assertEqual(self.frozen()[0].value, 1.0)

    def test_fixed_seed_invalid_or_inactive_parameters_never_reserve(self):
        params = {
            "training.learning_rate": 0.0002,
            "training.lr_scheduler": "constant",
            "training.objective_chunk_blocks": 12,
            "training.optimizer_cpu_offload": True,
        }
        invalid = [
            dict(params, **{"training.learning_rate": float("inf")}),
            dict(params, **{"training.learning_rate": 0.1}),
            dict(params, **{"training.lr_scheduler": "__import__('os')"}),
            dict(params, **{"training.optimizer_cpu_offload": 1}),
            dict(params, **{"training.objective_chunk_blocks": 12.5}),
            {k: v for k, v in params.items() if k != "training.learning_rate"},
            dict(params, **{"training.loss_type": "dpace"}),
        ]
        for bad in invalid:
            with self.subTest(params=bad), self.assertRaises(SearchError):
                self.controller().ask(
                    free_memory_mib=7577.6, estimated_memory_mib=5000, params=bad
                )
            self.assertFalse(self.path.exists())
        self.c = contract("dspark")
        with self.assertRaises(SearchError):
            self.controller().ask(
                free_memory_mib=7577.6, estimated_memory_mib=5000, params=params
            )

    def test_seed_queue_crash_is_failed_and_charged_on_restart(self):
        import optuna

        params = {
            "training.learning_rate": 0.0002,
            "training.lr_scheduler": "constant",
            "training.objective_chunk_blocks": 12,
            "training.optimizer_cpu_offload": True,
        }
        with patch.object(
            optuna.study.Study,
            "ask",
            side_effect=RuntimeError("Injected queued seed crash"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Injected queued"):
                self.controller().ask(
                    free_memory_mib=7577.6, estimated_memory_mib=5000, params=params
                )
        self.assertEqual(self.frozen()[0].state.name, "WAITING")
        status = self.cli("status")
        self.assertEqual(status["trials"][0]["state"], "interrupted")
        self.assertEqual(status["reserved"]["trials"], 1)
        self.assertEqual(self.frozen()[0].state.name, "FAIL")
        self.assertEqual(self.ask()["trial"], 1)

    def test_reports_reject_invalid_values_lineage_and_conflicting_duplicates(self):
        proposal = self.ask()
        good = receipt(self.c, proposal)
        invalid = [
            ("validation_loss", float("nan")),
            ("validation_loss", float("inf")),
            ("validation_loss", True),
            ("elapsed_seconds", -1),
            ("updates", True),
            ("updates", 1),
            ("config_digest", sha("other-config")),
            ("contract_digest", sha("other-contract")),
            ("parent_checkpoint_sha256", sha("other-parent")),
            ("checkpoint_sha256", good["parent_checkpoint_sha256"]),
        ]
        for key, value in invalid:
            bad = dict(good, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(SearchError):
                self.controller().report(0, bad)
            self.assertEqual(self.controller().status()["trials"][0]["reports"], [])
        self.controller().report(0, good)
        self.assertEqual(len(self.controller().report(0, good)["reports"]), 1)
        with self.assertRaisesRegex(SearchError, "Conflicting duplicate"):
            self.controller().report(0, dict(good, validation_loss=2))
        with self.assertRaises(SearchError):
            self.controller().report(0, dict(good, throughput=100))
        r2 = receipt(self.c, proposal, 4, parent=good["checkpoint_sha256"])
        with self.assertRaisesRegex(SearchError, "lineage"):
            self.controller().report(
                0,
                dict(
                    r2,
                    checkpoint_sha256=self.c["identity"]["draft"]["checkpoint_sha256"],
                ),
            )
        for key in ("elapsed_seconds", "peak_memory_mib"):
            with self.subTest(key=key), self.assertRaises(SearchError):
                self.controller().report(0, dict(r2, **{key: 0}))
        self.controller().report(0, r2)
        with self.assertRaises(SearchError):
            self.controller().report(0, dict(r2, updates=3))

    def test_invalid_cli_report_leaves_trial_running(self):
        proposal = self.ask()
        rp = self.directory / "bad-report.json"
        rp.write_text(
            json.dumps(dict(receipt(self.c, proposal), validation_loss=float("nan")))
        )
        result = self.cli("report", "--trial", 0, "--receipt", rp, success=False)
        self.assertIn("Non-finite", result["error"])
        self.assertEqual(self.cli("status")["trials"][0]["state"], "running")

    def test_no_implicit_retry_and_terminal_states_are_durable(self):
        for state in ("interrupted", "failed", "rejected"):
            proposal = self.ask()
            with self.assertRaisesRegex(SearchError, "still running"):
                self.ask()
            self.controller().tell(
                proposal["trial"], state=state, reason="Explicit operator decision"
            )
            self.assertEqual(self.cli("status")["trials"][-1]["state"], state)
            self.assertEqual(self.frozen()[-1].state.name, "FAIL")
            with self.assertRaises(SearchError):
                self.controller().tell(proposal["trial"], state="complete")
        self.assertEqual(self.controller().status()["reserved"]["updates"], 12)

    def test_memory_rejection_consumes_trial_and_budget_across_restart(self):
        self.c["budget"]["max_trials"] = 1
        result = self.controller().ask(free_memory_mib=6500, estimated_memory_mib=6000)
        self.assertEqual(result["state"], "rejected")
        self.assertEqual(self.frozen()[0].state.name, "FAIL")
        with self.assertRaisesRegex(SearchError, "count budget"):
            self.ask()
        self.assertEqual(self.cli("status")["reserved"]["trials"], 1)

    def test_update_and_elapsed_reservation_caps_never_refund_unused_budget(self):
        for total, amount in (("max_total_updates", 4), ("max_total_seconds", 3600)):
            with self.subTest(total=total):
                self.path = self.directory / f"{total}.sqlite"
                self.c = contract()
                self.c["budget"].update(max_total_updates=None, max_total_seconds=None)
                self.c["budget"][total] = amount
                proposal = self.ask()
                self.controller().tell(
                    proposal["trial"],
                    state="interrupted",
                    reason="Stopped before any update",
                )
                with self.assertRaisesRegex(SearchError, total):
                    self.ask()
                self.assertEqual(self.controller().status()["reserved"]["updates"], 4)

    def test_valid_but_over_budget_or_memory_reports_reject_durably(self):
        cases = [
            ("updates", 5),
            ("elapsed_seconds", 3601),
            ("peak_memory_mib", 5001),
            ("free_memory_mib", 1023),
        ]
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key):
                self.path = self.directory / f"overrun-{index}.sqlite"
                proposal = self.ask()
                bad = dict(receipt(self.c, proposal), **{key: value})
                result = self.controller().report(0, bad)
                self.assertEqual(result["state"], "rejected")
                self.assertEqual(result["rejected_receipt"], bad)
                self.assertEqual(self.cli("status")["trials"][0]["state"], "rejected")
                self.assertEqual(self.frozen()[0].state.name, "FAIL")

    def test_rejected_receipt_cli_retry_is_idempotent_after_process_restart(self):
        cases = [
            ("updates", 5),
            ("elapsed_seconds", 3601),
            ("peak_memory_mib", 5001),
            ("free_memory_mib", 1023),
        ]
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key):
                self.path = self.directory / f"rejected-retry-{index}.sqlite"
                proposal = self.ask()
                rejected = dict(receipt(self.c, proposal), **{key: value})
                rp = self.directory / "rejected-receipt.json"
                rp.write_text(json.dumps(rejected))
                first = self.cli("report", "--trial", 0, "--receipt", rp)
                self.assertEqual(first["state"], "rejected")
                before = self.cli("status")

                retried = self.cli("report", "--trial", 0, "--receipt", rp)
                self.assertEqual(retried, first)
                self.assertEqual(self.cli("status"), before)
                self.assertEqual(before["trials"][0]["rejected_receipt"], rejected)
                self.assertEqual(self.frozen()[0].state.name, "FAIL")
                self.assertEqual(self.frozen()[0].intermediate_values, {})

                rp.write_text(json.dumps(dict(rejected, validation_loss=2)))
                conflict = self.cli(
                    "report", "--trial", 0, "--receipt", rp, success=False
                )
                self.assertEqual(conflict["error"], "Conflicting duplicate report")
                self.assertEqual(self.cli("status"), before)

    def test_reported_cost_overrun_is_never_hidden_by_smaller_reservation(self):
        for field, amount, total in (
            ("updates", 17, "max_total_updates"),
            ("elapsed_seconds", 7201, "max_total_seconds"),
        ):
            with self.subTest(field=field):
                self.path = self.directory / f"cost-{field}.sqlite"
                self.c = contract()
                if field == "elapsed_seconds":
                    self.c["budget"].update(max_total_seconds=7200)
                proposal = self.ask()
                result = self.controller().report(
                    0, dict(receipt(self.c, proposal), **{field: amount})
                )
                self.assertEqual(result["state"], "rejected")
                status = self.cli("status")
                self.assertEqual(
                    status["charged"]["updates" if field == "updates" else "seconds"],
                    amount,
                )
                with self.assertRaisesRegex(SearchError, total):
                    self.ask()
                self.assertEqual(self.controller().status()["reserved"]["trials"], 1)

    def test_contract_reuse_rejects_identity_objective_space_and_budget_changes(self):
        self.ask()
        original = copy.deepcopy(self.c)
        mutations = [
            lambda c: c["identity"]["target"].update(weights_sha256=sha("target2")),
            lambda c: c["identity"]["draft"].update(checkpoint_sha256=sha("base2")),
            lambda c: c["identity"]["data"].update(split_sha256=sha("split2")),
            lambda c: c["identity"]["objective"].update(family="dspark"),
            lambda c: c["identity"]["objective"].update(
                config_sha256=sha("new-objective")
            ),
            lambda c: c["budget"].update(max_trials=99),
            lambda c: c["space"]["training.learning_rate"].update(high=0.01),
            lambda c: c["pruner"].update(min_trials=2),
        ]
        for mutate in mutations:
            self.c = copy.deepcopy(original)
            mutate(self.c)
            with (
                self.subTest(contract=self.c),
                self.assertRaisesRegex(SearchError, "contract digest mismatch"),
            ):
                self.controller().status()
        self.c = original
        self.assertEqual(self.controller().status()["reserved"]["trials"], 1)

    def test_completion_requires_fixed_endpoint_and_no_score_override(self):
        proposal = self.ask()
        with self.assertRaises(SearchError):
            self.controller().tell(0, state="complete")
        with self.assertRaises(SearchError):
            self.controller().tell(0, state="pruned", reason="No evaluation yet")
        self.controller().report(0, receipt(self.c, proposal))
        with self.assertRaises(SearchError):
            self.controller().tell(0, state="complete")
        self.assertEqual(self.controller().status()["trials"][0]["state"], "running")

    def test_deadline_survives_restart(self):
        self.c["budget"].update(max_total_seconds=7200)
        proposal = self.ask()
        started = proposal["started_at"]
        with patch("draftfit.training.search.time.time", return_value=started + 3601):
            result = self.controller().report(0, receipt(self.c, proposal))
        self.assertEqual(result["state"], "rejected")
        with patch("draftfit.training.search.time.time", return_value=started + 7201):
            with self.assertRaisesRegex(SearchError, "Study elapsed deadline"):
                self.ask()
        self.assertEqual(self.controller().status()["reserved"]["trials"], 1)

    def test_allocation_crash_is_charged_and_not_retried(self):
        import optuna

        actual = optuna.study.Study.ask

        def crash_after_ask(study, *args, **kwargs):
            actual(study, *args, **kwargs)
            raise RuntimeError("Injected crash after durable Optuna allocation")

        with patch.object(optuna.study.Study, "ask", crash_after_ask):
            with self.assertRaisesRegex(RuntimeError, "Injected crash"):
                self.ask()
        status = self.cli("status")
        self.assertEqual(status["reserved"]["trials"], 1)
        self.assertEqual(status["trials"][0]["state"], "interrupted")
        self.assertEqual(self.frozen()[0].state.name, "FAIL")
        self.assertEqual(self.ask()["trial"], 1)

    def test_report_and_tell_intents_reconcile_after_storage_interruption(self):
        import optuna

        proposal = self.ask()
        r1 = receipt(self.c, proposal)
        with patch.object(
            optuna.trial.Trial,
            "report",
            side_effect=RuntimeError("Injected report failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Injected report"):
                self.controller().report(0, r1)
        self.assertEqual(self.cli("status")["trials"][0]["reports"], [r1])
        self.assertEqual(self.frozen()[0].intermediate_values, {1: 1.0})
        self.controller().report(
            0, receipt(self.c, proposal, 4, parent=r1["checkpoint_sha256"])
        )
        with patch.object(
            optuna.study.Study,
            "tell",
            side_effect=RuntimeError("Injected tell failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Injected tell"):
                self.controller().tell(0, state="complete")
        self.assertEqual(self.cli("status")["trials"][0]["state"], "complete")
        self.assertEqual(self.frozen()[0].state.name, "COMPLETE")

    def test_unmanaged_study_is_not_adopted(self):
        import optuna

        storage = optuna.storages.RDBStorage(f"sqlite:///{self.path}")
        try:
            optuna.create_study(storage=storage, study_name=self.c["study_name"])
        finally:
            storage.remove_session()
            storage.engine.dispose()
        with self.assertRaisesRegex(SearchError, "unmanaged"):
            self.controller().status()

    def test_concurrent_cli_asks_cannot_double_reserve(self):
        cp = self.directory / "contract.json"
        cp.write_text(json.dumps(self.c))
        command = [
            sys.executable,
            "-m",
            "draftfit.training.search",
            "ask",
            "--contract",
            str(cp),
            "--storage",
            str(self.path),
            "--free-memory-mib",
            "7577.6",
            "--estimated-memory-mib",
            "5000",
        ]
        processes = [
            subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        outputs = [p.communicate(timeout=30) for p in processes]
        self.assertEqual(sorted(p.returncode for p in processes), [0, 2], outputs)
        self.assertEqual(self.controller().status()["reserved"]["trials"], 1)


if __name__ == "__main__":
    unittest.main()
