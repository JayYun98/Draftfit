"""Actual tiny target/joint updates; only remote publication transport is injected."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.unattended import Controller, load_plan, read, run, write
from tests.test_joint_online_cpu import online_fixture
from tests.test_joint_policy import policy


class UnattendedControllerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        env = mock.patch.dict(
            os.environ,
            {
                "SPECFORGE_DEVICE": "cpu",
                "OMP_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        torch.set_num_threads(1)
        online_fixture(self.root, False)
        base = read(self.root / "base.json")
        base["training"].update(
            max_steps=4,
            total_steps=4,
            num_epochs=4,
            joint_feature_queue_bytes=32768,
            joint_feature_queue_batches=4,
            max_checkpoints=1,
            joint_verification=dict(
                interval=1, max_prompts=2, max_new_tokens=4, draft_tokens=2, trials=1
            ),
        )
        write(self.root / "base.json", base)
        rows = [
            json.loads(x) for x in (self.root / "eval.jsonl").read_text().splitlines()
        ]
        for r in rows:
            r["domain"] = "general"
        (self.root / "eval.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        write(self.root / "budget.json", dict(timestamp_unix=time.time(), spent_usd=0))
        self.config = dict(
            schema="draftfit-unattended-v1",
            manifest="family.json",
            output_dir="unattended",
            max_steps=8,
            deadline_unix=time.time() + 100,
            budget_path="budget.json",
            selection="acceptance",
            policy=policy(
                domains=["general"],
                patience=1,
                min_delta=1e6,
                min_requests=2,
                next_steps=1,
            ),
        )
        self.config["policy"]["limits"]["max_steps"] = 8
        self.path = self.root / "controller.json"
        write(self.path, self.config)

    def test_real_joint_early_stop_best_export_and_idempotent_resume(self):
        result = run(self.path)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["best"]), {"dspark", "ce", "selector"})
        for best in result["best"].values():
            self.assertTrue(Path(best["checkpoint"]).is_file())
            self.assertTrue((Path(best["export"]) / "model.safetensors").is_file())
        resumed = run(self.path, resume=True)
        self.assertEqual(len(resumed["phases"][0]["attempts"]), 1)
        self.assertEqual(resumed["best"], result["best"])

    def test_failed_observer_resume_replays_before_next_update(self):
        original = Controller.observe
        failed = [False]

        def fail(controller, **kwargs):
            if kwargs["step"] == 1 and not failed[0]:
                failed[0] = True
                raise RuntimeError("injected observer failure")
            return original(controller, **kwargs)

        with mock.patch.object(Controller, "observe", fail):
            with self.assertRaisesRegex(RuntimeError, "observer failure"):
                run(self.path)
        result = run(self.path, resume=True)
        history = result["history"]["phase0:dspark"]
        self.assertEqual([r["artifact"]["step"] for r in history], [1, 2])
        self.assertEqual(result["total_steps"], 2)

    def test_latest_export_tracks_current_weights_without_replacing_best(self):
        from safetensors.torch import load_file

        result = run(self.path)
        phase = result["phases"][0]
        self.assertEqual(result["total_steps"], 2)
        self.assertTrue(all(row["step"] == 1 for row in result["best"].values()))
        latest = phase["latest_export"]
        self.assertEqual(latest["checkpoint"], phase["checkpoint"])
        self.assertEqual(latest["checkpoint_sha256"], phase["checkpoint_sha256"])
        state = torch.load(phase["resume_checkpoint"], weights_only=True)
        for name, row in latest["arms"].items():
            self.assertNotEqual(row["export"], result["best"][name]["export"])
            weights = load_file(str(Path(row["export"]) / "model.safetensors"))
            for key, value in state["drafts"][name].items():
                self.assertTrue(torch.equal(weights[key], value), (name, key))
            Controller.verify_export(result["best"][name])
        resumed = run(self.path, resume=True)
        self.assertEqual(resumed["phases"][0]["latest_export"], latest)
        self.assertEqual(resumed["best"], result["best"])
        self.assertEqual(len(resumed["phases"][0]["attempts"]), 1)
        # A retained latest export has the same immutable-byte contract as best.
        folder = Path(latest["arms"]["dspark"]["export"])
        cfg = read(folder / "config.json")
        cfg["hidden_size"] += 1
        write(folder / "config.json", cfg)
        with self.assertRaisesRegex(ValueError, "immutable export changed"):
            run(self.path, resume=True)

    def test_cli_rejects_unaligned_admission_before_creating_outputs(self):
        import subprocess
        import sys

        base = read(self.root / "base.json")
        base["training"]["joint_verification"]["interval"] = 2
        write(self.root / "base.json", base)
        self.config["policy"].update(next_steps=3, patience=50)
        write(self.path, self.config)
        for flags in (["--plan"], []):
            with self.subTest(flags=flags):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "draftfit.cli",
                        "unattended",
                        "--config",
                        str(self.path),
                        *flags,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("multiple of verification interval", completed.stderr)
                self.assertFalse((self.root / "unattended").exists())

    def test_speed_thresholds_and_triggers_require_unsupported_benchmark_quality(self):
        from copy import deepcopy

        for serving in (False, True):
            for setting in (
                "min_speedup",
                "min_domain_speedup",
                "below_serving_threshold",
                "acceptance_improving_speed_plateau",
            ):
                with self.subTest(serving=serving, setting=setting):
                    config = deepcopy(self.config)
                    if serving:
                        config["serving"] = dict(max_seconds=20, cost_usd=0.001)
                    if setting.startswith("min_"):
                        config["policy"][setting] = 1.1
                    else:
                        config["presets"] = [
                            dict(
                                id="speed-pilot",
                                when=[setting],
                                reason="speed policy",
                                overrides=["training.learning_rate=0.00001"],
                                pilot_steps=2,
                                pilot_cost_usd=0.1,
                            )
                        ]
                    write(self.path, config)
                    with self.assertRaisesRegex(ValueError, "benchmark-quality speed"):
                        load_plan(self.path)
                    self.assertFalse((self.root / "unattended").exists())
            # Zero thresholds and the default relative-gain tuning remain valid.
            config = deepcopy(self.config)
            config["policy"].update(min_speedup=0, min_domain_speedup=0)
            if serving:
                config["serving"] = dict(max_seconds=20, cost_usd=0.001)
            write(self.path, config)
            load_plan(self.path)

    def test_aligned_admission_allows_final_partial_phase_and_step_budget(self):
        from draftfit.training import joint_family

        for phase_steps, next_steps, external_steps, expected in (
            (3, 2, 0, 3),
            (4, 2, 5, 3),
            (5, 4, 0, 5),
        ):
            with self.subTest(phase_steps=phase_steps, external_steps=external_steps):
                base = read(self.root / "base.json")
                base["training"].update(max_steps=phase_steps, total_steps=phase_steps)
                base["training"]["joint_verification"]["interval"] = 2
                write(self.root / "base.json", base)
                self.config["output_dir"] = f"partial-{phase_steps}"
                self.config["policy"].update(next_steps=next_steps, patience=50)
                write(self.path, self.config)
                write(
                    self.root / "budget.json",
                    dict(
                        timestamp_unix=time.time(),
                        spent_usd=0,
                        steps_used=external_steps,
                    ),
                )
                original = Controller.observe
                observed = []

                def observe(controller, **kwargs):
                    observed.append(kwargs["step"])
                    self.assertLessEqual(
                        kwargs["step"], controller.state["phases"][-1]["admitted_until"]
                    )
                    return original(controller, **kwargs)

                with (
                    mock.patch.object(Controller, "observe", observe),
                    mock.patch.object(
                        joint_family,
                        "_backward_window",
                        wraps=joint_family._backward_window,
                    ) as backward,
                ):
                    result = run(self.path)
                self.assertEqual(backward.call_count, expected)
                self.assertEqual(result["total_steps"], expected)
                self.assertEqual(observed, [*range(2, expected, 2), expected])
                phase = result["phases"][0]
                self.assertEqual(
                    phase["latest_export"]["checkpoint"], phase["resume_checkpoint"]
                )
                checkpoint = torch.load(phase["resume_checkpoint"], weights_only=True)
                self.assertEqual(checkpoint["step"], expected)
                self.assertEqual(checkpoint["horizon"], phase_steps)
                resumed = run(self.path, resume=True)
                self.assertEqual(resumed["total_steps"], expected)
                self.assertEqual(len(resumed["phases"][0]["attempts"]), 1)

    def test_deadline_between_milestones_retains_latest_verified_export(self):
        from draftfit.training import joint_family

        self.config["policy"].update(next_steps=2, patience=50)
        write(self.path, self.config)
        real_backward = joint_family._backward_window
        now = [time.time()]
        calls = []
        clock = mock.Mock(wraps=time)
        clock.time.side_effect = lambda: now[0]

        def backward(*args, **kwargs):
            result = real_backward(*args, **kwargs)
            calls.append(True)
            if len(calls) == 3:
                now[0] = self.config["deadline_unix"] + 1
            return result

        # Only the clock is simulated; updates, checkpoints and exports are real.
        with (
            mock.patch("draftfit.unattended.time", clock),
            mock.patch.object(joint_family, "time", clock),
            mock.patch.object(joint_family, "_backward_window", backward),
        ):
            result = run(self.path)
        self.assertEqual(result["total_steps"], 3)
        phase = result["phases"][0]
        self.assertEqual(phase["result"]["reason"], "deadline")
        self.assertEqual(phase["last_observation"]["step"], 2)
        self.assertEqual(phase["latest_export"]["checkpoint"], phase["checkpoint"])
        self.assertEqual(
            phase["latest_export"]["checkpoint_sha256"], phase["checkpoint_sha256"]
        )
        self.assertNotEqual(phase["resume_checkpoint"], phase["checkpoint"])
        self.assertEqual(
            torch.load(phase["resume_checkpoint"], weights_only=True)["step"], 3
        )
        self.assertLessEqual(result["total_steps"], phase["admitted_until"])
        resumed = run(self.path, resume=True)
        self.assertEqual(resumed["total_steps"], 3)
        self.assertEqual(len(resumed["phases"][0]["attempts"]), 1)

    def test_unobserved_resume_keeps_partial_step_budget_and_exact_state(self):
        base = read(self.root / "base.json")
        base["training"]["joint_verification"]["interval"] = 2
        write(self.root / "base.json", base)
        self.config["policy"].update(next_steps=2, patience=50)
        write(self.path, self.config)
        write(
            self.root / "budget.json",
            dict(timestamp_unix=time.time(), spent_usd=0, steps_used=5),
        )
        reference = run(self.path)
        self.config["output_dir"] = "interrupted-partial"
        write(self.path, self.config)
        with mock.patch.object(
            Controller, "observe", side_effect=RuntimeError("before observation")
        ):
            with self.assertRaisesRegex(RuntimeError, "before observation"):
                run(self.path)
        failed = read(self.root / "interrupted-partial/state.json")
        self.assertEqual(failed["total_steps"], 0)
        resumed = run(self.path, resume=True)
        self.assertEqual(resumed["total_steps"], 3)
        for history in resumed["history"].values():
            self.assertEqual([row["artifact"]["step"] for row in history], [2, 3])
        expected = torch.load(
            reference["phases"][0]["resume_checkpoint"], weights_only=True
        )
        actual = torch.load(
            resumed["phases"][0]["resume_checkpoint"], weights_only=True
        )

        def assert_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_equal(a, b)
            else:
                self.assertEqual(left, right)

        for key in (
            "drafts",
            "optimizer",
            "rng",
            "step",
            "epoch",
            "next_batch",
            "consumed_samples",
        ):
            with self.subTest(field=key):
                assert_equal(expected[key], actual[key])

    def test_registered_weights_only_phase_runs_automatically(self):
        base = read(self.root / "base.json")
        base["training"]["max_checkpoints"] = 10
        write(self.root / "base.json", base)
        self.config["presets"] = [
            dict(
                id="lower-lr",
                when=["loss_plateau"],
                reason="bounded pilot",
                overrides=["training.learning_rate=0.00001"],
                pilot_steps=2,
                pilot_cost_usd=0.1,
            )
        ]
        limits = self.config["policy"]["limits"]
        limits.update(
            max_pilots=1, max_pilot_steps=2, max_pilot_usd=0.2, max_exploration_usd=0.2
        )
        write(self.path, self.config)
        result = run(self.path)
        self.assertEqual(
            [p["preset"] for p in result["phases"]], ["initial", "lower-lr"]
        )
        self.assertEqual(result["total_steps"], 4)
        second = result["phases"][1]
        initial = torch.load(
            Path(second["attempts"][0]) / "joint-step0.pt", weights_only=True
        )
        source = torch.load(result["phases"][0]["checkpoint"], weights_only=True)
        for arm, weights in source["drafts"].items():
            for key, value in weights.items():
                self.assertTrue(torch.equal(value, initial["drafts"][arm][key]))
        self.assertEqual(initial["step"], 0)
        self.assertEqual(initial["optimizer"]["optimizer_state_dict"]["state"], {})

    def test_publication_retry_does_not_retrain_completed_phase(self):
        self.config["publication"] = dict(
            private=True, repos={n: "tester/" + n for n in ["dspark", "ce", "selector"]}
        )
        write(self.path, self.config)
        config, stages, binding = load_plan(self.path)
        import sys

        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            Controller(
                config,
                stages,
                binding,
                publication_command=[sys.executable, "-c", "raise SystemExit(1)"],
            ).execute()
        success = "import json,sys;from pathlib import Path;p=Path(sys.argv[-1]);(p/'receipt.json').write_text(json.dumps({'status':'verified'}))"
        result = Controller(
            config,
            stages,
            binding,
            resume=True,
            publication_command=[sys.executable, "-c", success],
        ).execute()
        self.assertEqual(result["total_steps"], 2)
        self.assertTrue(result["publication_complete"])
        self.assertTrue(
            all(
                read(Path(p) / "receipt.json")["status"] == "verified"
                for p in result["publication_jobs"]
            )
        )

    def test_invalid_preset_is_rejected_before_training_or_outputs(self):
        self.config["presets"] = [
            dict(
                id="bad",
                when=["loss_plateau"],
                reason="test",
                overrides=["training.batch_size=1000"],
                pilot_steps=2,
                pilot_cost_usd=0.1,
            )
        ]
        write(self.path, self.config)
        with self.assertRaises(ValueError):
            load_plan(self.path)
        self.assertFalse((self.root / "unattended").exists())

    def test_changed_holdout_cannot_resume_even_between_phases(self):
        run(self.path)
        rows = [
            json.loads(line)
            for line in (self.root / "eval.jsonl").read_text().splitlines()
        ]
        rows[0]["input_ids"][0] = 45
        (self.root / "eval.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        with self.assertRaisesRegex(ValueError, "changed"):
            run(self.path, resume=True)

    def test_invalid_budget_is_rejected_on_resume_before_more_updates(self):
        original = Controller.observe

        def fail(controller, **kwargs):
            raise RuntimeError("interrupt")

        with mock.patch.object(Controller, "observe", fail):
            with self.assertRaises(RuntimeError):
                run(self.path)
        write(
            self.root / "budget.json",
            dict(timestamp_unix=time.time() - 121, spent_usd=0),
        )
        with self.assertRaisesRegex(ValueError, "fresh external budget"):
            run(self.path, resume=True)
        self.assertEqual(
            len(read(self.root / "unattended/state.json")["phases"][0]["attempts"]), 1
        )

    def test_export_checks_config_bytes_and_uses_authoritative_joint(self):
        result = run(self.path)
        config, stages, binding = load_plan(self.path)
        controller = Controller(config, stages, binding, resume=True)
        self.addCleanup(controller.lock.close)
        self.addCleanup(controller.publisher.close)
        best = result["best"]["dspark"]
        # The per-arm convenience snapshot is deliberately corrupted. Export
        # must still be identical to the authoritative shared joint checkpoint.
        checkpoint = Path(best["checkpoint"])
        step = int(checkpoint.stem.removeprefix("joint-step"))
        for stage in stages:
            path = (
                checkpoint.parent
                / stage.name
                / f"{stage.name}-step{step}"
                / "training_state.pt"
            )
            if path.exists():
                path.write_bytes(b"not a checkpoint")
        exported = controller.export(
            checkpoint, stages, self.root / "unattended/artifacts/reexport"
        )
        from safetensors.torch import load_file

        state = torch.load(checkpoint, weights_only=True)
        for stage in stages:
            weights = load_file(
                str(Path(exported["arms"][stage.name]["export"]) / "model.safetensors")
            )
            for key, value in state["drafts"][stage.name].items():
                self.assertTrue(torch.equal(value, weights[key]), (stage.name, key))
        folder = Path(exported["arms"]["dspark"]["export"])
        cfg = read(folder / "config.json")
        cfg["hidden_size"] += 1
        write(folder / "config.json", cfg)
        with self.assertRaisesRegex(ValueError, "immutable export changed"):
            controller.export(checkpoint, stages, folder.parent)

    def test_real_cli_plan_does_not_create_training_output(self):
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "draftfit.cli",
                "unattended",
                "--config",
                str(self.path),
                "--plan",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout)["arms"], ["dspark", "ce", "selector"]
        )
        self.assertFalse((self.root / "unattended").exists())

    def test_resource_advice_runs_registered_phase_with_real_updates_mock_telemetry(
        self,
    ):
        # GPU telemetry is explicitly simulated; all target/draft updates are real CPU.
        self.config["policy"].update(patience=50, min_delta=0)
        self.config["presets"] = [
            dict(
                id="batch-up",
                when=["loss_plateau"],
                reason="pilot",
                overrides=["training.batch_size=2"],
                pilot_steps=2,
                pilot_cost_usd=0.1,
            )
        ]
        self.config["policy"]["limits"].update(
            max_pilots=1, max_pilot_steps=2, max_pilot_usd=0.1, max_exploration_usd=0.1
        )
        self.config["resources"] = dict(
            policy=dict(window_size=2, cooldown_steps=0),
            pilot_seconds={"initial": 2, "batch-up": 2},
        )
        write(self.path, self.config)

        class Sampler:
            def __init__(self, **kw):
                pass

            def start(self):
                return self

            def stop(self):
                return self

            def summarize(self, phase, preset, step, throughput, loss):
                return dict(
                    status="available",
                    observation=dict(
                        phase_id=phase,
                        preset=preset,
                        step=step,
                        throughput=10,
                        loss=1,
                        peak_vram_bytes=20,
                        capacity_bytes=100,
                        utilization=0.2,
                    ),
                )

        with mock.patch("draftfit.training.resource_monitor.ResourceSampler", Sampler):
            result = run(self.path)
        self.assertEqual(
            [p["preset"] for p in result["phases"]], ["initial", "batch-up"]
        )
        self.assertEqual(result["total_steps"], 4)
        self.assertEqual(result["resource_state"]["transitions"], 1)
        self.assertAlmostEqual(result["exploration_floor"], 0.1)
        self.assertEqual(
            result["phases"][0]["resource_decision"]["reason"], "underutilized_window"
        )

    def test_cpu_resources_stay_unavailable_and_never_invent_gpu_advice(self):
        self.config["resources"] = dict(policy={}, pilot_seconds={"initial": 2})
        write(self.path, self.config)
        result = run(self.path)
        self.assertTrue(result["resource_reports"])
        self.assertTrue(
            all(
                r["status"] == "measurement_unavailable"
                for r in result["resource_reports"]
            )
        )
        self.assertEqual(result["resource_state"]["transitions"], 0)

    def test_hung_publication_obeys_controller_deadline_and_keeps_jobs(self):
        import sys

        self.config["publication"] = dict(
            private=True, repos={n: "tester/" + n for n in ["dspark", "ce", "selector"]}
        )
        self.config["deadline_unix"] = time.time() + 3
        write(self.path, self.config)
        config, stages, binding = load_plan(self.path)
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "publication failed or timed out"):
            Controller(
                config,
                stages,
                binding,
                publication_command=[
                    sys.executable,
                    "-c",
                    "import time;time.sleep(100)",
                ],
            ).execute()
        self.assertLess(time.monotonic() - start, 6)
        state = read(self.root / "unattended/state.json")
        self.assertTrue(state["publication_jobs"])
        self.assertTrue(
            all((Path(p) / "payload").is_dir() for p in state["publication_jobs"])
        )
        self.assertTrue(state["best"])

    def test_too_small_or_wrong_domain_workload_rejected_before_training(self):
        for override in ({"min_requests": 3}, {"domains": ["absent"]}):
            with self.subTest(override=override):
                config = {
                    **self.config,
                    "policy": {**self.config["policy"], **override},
                }
                write(self.path, config)
                with self.assertRaisesRegex(ValueError, "workload"):
                    load_plan(self.path)
                self.assertFalse((self.root / "unattended").exists())

    def test_corrupted_unobserved_checkpoint_is_rejected_before_resume(self):
        with mock.patch.object(
            Controller, "observe", side_effect=RuntimeError("interrupt")
        ):
            with self.assertRaises(RuntimeError):
                run(self.path)
        path = self.root / "unattended/phase0/attempt0/joint-step1.pt"
        state = torch.load(path, weights_only=True)
        next(iter(state["drafts"]["dspark"].values())).add_(1)
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "integrity receipt"):
            run(self.path, resume=True)

    def test_process_exit_after_committed_observation_resumes_exactly(self):
        import subprocess
        import sys

        reference = run(self.path)
        reference_checkpoint = reference["phases"][0]["resume_checkpoint"]
        self.config["output_dir"] = "killed"
        write(self.path, self.config)
        script = """import os,sys
from draftfit.unattended import Controller,run
original=Controller.observe
def observe(self,**kwargs):
 result=original(self,**kwargs)
 if kwargs['step']==1: os._exit(23)
 return result
Controller.observe=observe
run(sys.argv[1])
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 23, completed.stderr)
        self.assertEqual(read(self.root / "killed/state.json")["total_steps"], 1)
        resumed = run(self.path, resume=True)
        actual = torch.load(
            resumed["phases"][0]["resume_checkpoint"], weights_only=True
        )
        expected = torch.load(reference_checkpoint, weights_only=True)
        self.assertEqual(
            [o["artifact"]["step"] for o in resumed["history"]["phase0:dspark"]], [1, 2]
        )
        for arm, weights in expected["drafts"].items():
            for key, value in weights.items():
                self.assertTrue(
                    torch.equal(value, actual["drafts"][arm][key]), (arm, key)
                )
        self.assertEqual(actual["consumed_samples"], expected["consumed_samples"])
        self.assertAlmostEqual(resumed["cost_floor"], reference["cost_floor"])

    def test_existing_drafts_are_compared_but_not_selected(self):
        reference = run(self.path)
        base = read(self.root / "base.json")
        base["training"]["joint_verification"]["existing_drafts"] = {
            name: row["export"] for name, row in reference["best"].items()
        }
        write(self.root / "base.json", base)
        self.config["output_dir"] = "compare-existing"
        write(self.path, self.config)
        result = run(self.path)
        last = Path(result["phases"][0]["attempts"][0]) / "verification-step2.json"
        report = read(last)
        self.assertEqual(
            {r["name"] for r in report["existing_drafts"]}, set(reference["best"])
        )
        self.assertTrue(report["tokenparity"])
        self.assertTrue(
            all(not row["eligible_for_selection"] for row in report["existing_drafts"])
        )
        self.assertEqual(set(result["best"]), set(reference["best"]))
        self.assertTrue(
            all(
                Path(b["checkpoint"]).is_relative_to(
                    (self.root / "compare-existing").resolve()
                )
                for b in result["best"].values()
            )
        )
        # Completed resume must still detect changes to the comparison reference.
        folder = Path(reference["best"]["dspark"]["export"])
        cfg = read(folder / "config.json")
        cfg["hidden_size"] += 1
        write(folder / "config.json", cfg)
        with self.assertRaisesRegex(ValueError, "changed"):
            run(self.path, resume=True)
