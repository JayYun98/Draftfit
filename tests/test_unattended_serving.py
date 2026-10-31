"""Actual tiny CPU training + real executable MOCK serving transport; no CUDA claim."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from draftfit.training import serving_dispatch as dispatcher
from draftfit.training import sglang_producer as producer
from draftfit.unattended import Controller, load_plan, read, run, write
from tests import test_unattended_controller as controller_tests

MOCK_COMMAND = """
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[-1]);r=json.loads((p/'request.json').read_text());e=r['expected'];b=e['bindings']
f=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
c=f(b)
def arm(name,artifact=None):
 rows=[]
 for trial in range(b['config']['trials']):
  for prompt in b['prompts']:
   n=b['config']['max_new_tokens']
   rows.append(dict(prompt,trial=trial,token_ids=list(range(n)),generated_tokens=n,seconds=.1,accepted_tokens=0,proposed_tokens=0 if artifact is None else n*3))
 a=dict(name=name,status='complete',comparison_fingerprint=c,runs=rows)
 if artifact is not None:a['artifact']=artifact
 return a
report=dict(schema='draftfit-production-serving-v1',status='complete',split='validation',production_engine_speed=True,
 speed_decisions_allowed=False,timing_scope='end_to_end_generation',acceptance_scope='committed_draft_suffix_excludes_target_anchor',
 bindings=b,comparison_fingerprint=c,baseline=arm('target-only'),candidates=[arm(n,a) for n,a in e['artifacts'].items()])
(p/'report.json').write_text(json.dumps(report))
"""


class UnattendedServingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = controller_tests.UnattendedControllerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.path = self.fixture.root, self.fixture.path
        cfg = read(self.path)
        cfg["serving"] = dict(
            max_seconds=20, cost_usd=0.001, config={"max_new_tokens": 32}
        )
        write(self.path, cfg)
        runtime = dict(
            engine="sglang",
            version="0.5.19",
            build_sha256="a" * 64,
            device="MOCK CPU test transport",
            dtype="bfloat16",
        )
        patch = mock.patch.object(producer, "runtime_identity", return_value=runtime)
        patch.start()
        self.addCleanup(patch.stop)
        original = dispatcher.dispatch
        self.calls = []
        self.fail_transport = False
        self.bad_parity = False

        def invoke(*args, **kwargs):
            self.calls.append(kwargs["admission"]["reservation_id"])
            code = MOCK_COMMAND
            if self.bad_parity:
                code += '\nreport["candidates"][0]["runs"][0]["token_ids"][0]+=1;(p/"report.json").write_text(json.dumps(report))'
            kwargs["command"] = [
                sys.executable,
                "-c",
                "raise SystemExit(9)" if self.fail_transport else code,
            ]
            return original(*args, **kwargs)

        patch = mock.patch.object(dispatcher, "dispatch", side_effect=invoke)
        patch.start()
        self.addCleanup(patch.stop)

    def test_real_controller_optional_hook_strict_policy_and_completed_resume(self):
        state = run(self.path)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["total_steps"], 2)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(
            set(state["production_history"]["phase0"]), {"dspark", "ce", "selector"}
        )
        for rows in state["production_history"]["phase0"].values():
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["serving"]["production_engine_speed"] for r in rows))
            self.assertTrue(
                all(not r["serving"]["speed_decisions_allowed"] for r in rows)
            )
        for decision in state["phases"][0]["decisions"].values():
            self.assertTrue(decision["current_serving_validated"])
            self.assertNotIn("speed", decision["best"])
        self.assertFalse(state["phases"][0]["production_report"]["target_resident"])
        again = run(self.path, resume=True)
        self.assertEqual(again["production_history"], state["production_history"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(again["best"], state["best"])

    def test_completed_resume_rejects_other_checkpoint_verified_job_substitution(self):
        import shutil

        state = run(self.path)
        best = state["best"]["dspark"]
        first = (
            self.root
            / "unattended/serving"
            / (best["phase"] + "-" + best["checkpoint_sha256"])
            / "job"
        )
        second = Path(state["phases"][0]["production_report"]["job"])
        self.assertNotEqual(first, second)
        shutil.rmtree(first)
        shutil.copytree(second, first)
        self.assertNotEqual(
            read(first / "request.json")["expected"]["artifacts"]["dspark"][
                "checkpoint_sha256"
            ],
            best["checkpoint_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "matching verified|binding"):
            run(self.path, resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_target_mutation_after_native_cannot_commit_production(self):
        original = producer.runtime_identity

        def mutate(deadline):
            path = self.root / "target/config.json"
            cfg = read(path)
            cfg["rms_norm_eps"] = 0.005
            write(path, cfg)
            return original(deadline)

        with mock.patch.object(producer, "runtime_identity", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                run(self.path)
        state = read(self.root / "unattended/state.json")
        self.assertEqual(state["history"], {})
        self.assertEqual(state.get("production_history", {}), {})
        self.assertEqual(state["best"], {})
        self.assertEqual(state["publication_jobs"], [])
        self.assertEqual(len(self.calls), 0)

    def test_source_mutation_after_production_policy_cannot_commit(self):
        original = dispatcher.policy_update

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.root / "target/config.json"
            cfg = read(path)
            cfg["rms_norm_eps"] = 0.005
            write(path, cfg)
            return result

        with mock.patch.object(dispatcher, "policy_update", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                run(self.path)
        state = read(self.root / "unattended/state.json")
        self.assertEqual(state["history"], {})
        self.assertEqual(state.get("production_history", {}), {})
        self.assertEqual(state["best"], {})
        self.assertEqual(state["publication_jobs"], [])
        self.assertEqual(len(self.calls), 1)

    def test_completed_resume_rejects_changed_best_export(self):
        state = run(self.path)
        best = next(iter(state["best"].values()))
        path = Path(best["export"]) / "config.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "immutable export changed"):
            run(self.path, resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_completed_resume_rejects_changed_committed_report(self):
        state = run(self.path)
        job = Path(state["phases"][0]["production_report"]["job"])
        path = job / read(job / "receipt.json")["attempt"] / "report.json"
        report = read(path)
        report["candidates"][0]["runs"][0]["seconds"] = 0.2
        write(path, report)
        with self.assertRaisesRegex(ValueError, "serving report changed"):
            run(self.path, resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_one_token_native_parity_uses_production_acceptance(self):
        base = read(self.root / "base.json")
        base["training"]["joint_verification"]["max_new_tokens"] = 1
        write(self.root / "base.json", base)
        state = run(self.path)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(self.calls), 2)
        reports = list(
            (self.root / "unattended").glob("phase*/attempt*/verification-step*.json")
        )
        self.assertTrue(reports)
        self.assertTrue(all(read(p)["tokenparity"] for p in reports))
        self.assertTrue(
            all(
                c["proposed_tokens"] == 0
                for p in reports
                for c in read(p)["candidates"]
            )
        )
        self.assertTrue(
            all(
                d["current_serving_validated"]
                for d in state["phases"][0]["decisions"].values()
            )
        )

    def test_failure_after_verified_receipt_replays_without_gpu_or_admission_duplication(
        self,
    ):
        original = dispatcher.policy_update
        failed = [False]

        def fail(*args, **kwargs):
            result = original(*args, **kwargs)
            if not failed[0]:
                failed[0] = True
                raise RuntimeError("after verified receipt")
            return result

        with mock.patch.object(dispatcher, "policy_update", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "after verified"):
                run(self.path)
        state = read(self.root / "unattended/state.json")
        self.assertEqual(state["history"], {})
        self.assertEqual(state.get("production_history", {}), {})
        self.assertEqual(state["best"], {})
        admissions = list((self.root / "unattended/serving").glob("*/admission-*.json"))
        self.assertEqual(len(admissions), 1)
        config, stages, binding = load_plan(self.path)
        controller = Controller(config, stages, binding, resume=True)
        try:
            self.assertGreaterEqual(
                controller.budget()["spent_usd"], read(admissions[0])["cost_floor"]
            )
        finally:
            controller.publisher.close()
            controller.lock.close()
        state = run(self.path, resume=True)
        self.assertEqual(state["total_steps"], 2)
        self.assertEqual(
            len(self.calls), 2
        )  # first receipt reused; only next milestone runs
        self.assertEqual(
            len(list((self.root / "unattended/serving").glob("*/admission-*.json"))), 2
        )
        for rows in state["production_history"]["phase0"].values():
            self.assertEqual(len(rows), 2)

    def test_actual_external_parity_failure_rolls_back_policy_and_keeps_checkpoint(
        self,
    ):
        self.bad_parity = True
        with self.assertRaisesRegex(ValueError, "parity"):
            run(self.path)
        state = read(self.root / "unattended/state.json")
        self.assertEqual(state["history"], {})
        self.assertEqual(state.get("production_history", {}), {})
        self.assertEqual(state["best"], {})
        self.assertEqual(state["publication_jobs"], [])
        self.assertTrue(
            (Path(state["phases"][0]["attempts"][0]) / "joint-step1.pt").is_file()
        )
        self.assertEqual(len(self.calls), 1)

    def test_failed_worker_retries_with_fresh_admission_without_duplicate_history(self):
        self.fail_transport = True
        with self.assertRaisesRegex(RuntimeError, "exited 9"):
            run(self.path)
        state = read(self.root / "unattended/state.json")
        self.assertEqual(state["history"], {})
        self.assertEqual(state["best"], {})
        self.fail_transport = False
        result = run(self.path, resume=True)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(set(self.calls)), 3)
        self.assertEqual(
            len(list((self.root / "unattended/serving").glob("*/admission-*.json"))), 3
        )
        self.assertTrue(
            all(len(h) == 2 for h in result["production_history"]["phase0"].values())
        )

    def test_real_cpu_next_updates_optimizer_rng_cursor_match_native(self):
        import torch

        cfg = read(self.path)
        serving = cfg.pop("serving")
        cfg["output_dir"] = "native"
        write(self.path, cfg)
        native = run(self.path)
        cfg["serving"] = serving
        cfg["output_dir"] = "external"
        write(self.path, cfg)
        external = run(self.path)
        left = torch.load(native["phases"][0]["resume_checkpoint"], weights_only=True)
        right = torch.load(
            external["phases"][0]["resume_checkpoint"], weights_only=True
        )

        def same(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                for key in a:
                    same(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    same(x, y)
            else:
                self.assertEqual(a, b)

        for key in ("drafts", "optimizer", "rng", "epoch", "next_batch", "step"):
            self.assertIn(key, left)
            same(left[key], right[key])
        self.assertEqual(left["step"], 2)

    def test_unavailable_handoff_and_bad_config_cannot_silently_use_native_acceptance(
        self,
    ):
        cfg = read(self.path)
        cfg["serving"]["cost_usd"] = -1
        write(self.path, cfg)
        with self.assertRaises(ValueError):
            load_plan(self.path)
        self.assertFalse((self.root / "unattended").exists())
        cfg["serving"]["cost_usd"] = 0.001
        cfg["selection"] = "loss"
        write(self.path, cfg)
        with self.assertRaisesRegex(ValueError, "acceptance"):
            load_plan(self.path)


if __name__ == "__main__":
    unittest.main()
