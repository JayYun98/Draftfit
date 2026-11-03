"""Real tiny CPU teacher, asynchronous features, updates and exact resume."""

import copy
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError

from draftfit.config import Config
from tests import test_joint_online_cpu as online_tests


class JointAsyncCaptureTests(unittest.TestCase):
    cli = online_tests.JointOnlineCPU.cli
    execution_device = "cpu"

    def configure(self, root, asynchronous):
        manifest = online_tests.online_fixture(root, seeded=False)
        path = root / "base.json"
        config = json.loads(path.read_text())
        config["training"].update(
            joint_feature_queue_bytes=32768,
            joint_feature_queue_batches=3,
            joint_async_capture=asynchronous,
        )
        path.write_text(json.dumps(config))
        return manifest

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_cli_updates_match_sync_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(123)
            manifest = self.configure(root, False)
            original = (root / "target/model.safetensors").read_bytes()
            original_features = {
                path: path.read_bytes() for path in root.rglob("*.ckpt")
            }
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            baseline = torch.load(root / "outputs/joint-step2.pt", weights_only=True)
            path = root / "base.json"
            config = json.loads(path.read_text())
            config["training"]["joint_async_capture"] = True
            path.write_text(json.dumps(config))
            manifest["output_dir"] = "async"
            (root / "family.json").write_text(json.dumps(manifest))
            self.cli(root, "family", "--manifest", "family.json", "--joint")
            candidate = torch.load(root / "async/joint-step2.pt", weights_only=True)
            self.assert_nested_equal(baseline["drafts"], candidate["drafts"])
            self.assert_nested_equal(baseline["optimizer"], candidate["optimizer"])
            self.assert_nested_equal(baseline["rng"], candidate["rng"])
            self.assertEqual(original, (root / "target/model.safetensors").read_bytes())
            manifest["output_dir"] = "resumed"
            (root / "family.json").write_text(json.dumps(manifest))
            self.cli(
                root,
                "family",
                "--manifest",
                "family.json",
                "--joint",
                "--resume-from",
                str(root / "async/joint-step1.pt"),
            )
            resumed = torch.load(root / "resumed/joint-step2.pt", weights_only=True)
            self.assert_nested_equal(candidate["drafts"], resumed["drafts"])
            self.assert_nested_equal(candidate["optimizer"], resumed["optimizer"])
            self.assert_nested_equal(candidate["rng"], resumed["rng"])
            curves = [
                json.loads(line)
                for line in (root / "async/learning-curves.jsonl")
                .read_text()
                .splitlines()
            ]
            closed = [row for row in curves if row["kind"] == "queue_closed"][-1]
            self.assertEqual(closed["consumed"], candidate["consumed_samples"])
            self.assertEqual(closed["resident_bytes"], 0)
            self.assertLessEqual(closed["peak_bytes"], 32768)
            self.assertEqual(
                original_features,
                {path: path.read_bytes() for path in root.rglob("*.ckpt")},
                "feature payload reached disk",
            )

    def test_source_prefetches_during_consumer_work_and_closes(self):
        from draftfit.family import build_family_plan
        from draftfit.training.joint_online import JointOnlineSource

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.configure(root, True)
            # Use the real family preparation path to resolve dimensions.
            import os
            from unittest.mock import patch

            with patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
                _, stages = build_family_plan(str(root / "family.json"))
                source = JointOnlineSource(stages[0], needs_dspark=True)
                self.addCleanup(source.close)
                next_capture = threading.Event()
                capture = source._capture_batch

                def observed(rows, indices, evaluation):
                    result = capture(rows, indices, evaluation)
                    if indices == [1]:
                        next_capture.set()
                    return result

                source._capture_batch = observed
                loader, _ = source.loader(stages[0], 0, False, True)
                batches = iter(loader)
                first = next(batches)
                self.assertTrue(
                    next_capture.wait(5), "no capture overlapped held consumer batch"
                )
                with source.pause_capture():
                    count = source.capture_calls
                    self.assertGreaterEqual(count, 2)
                source.consume([first])
                batches.close()
                source.close()
                self.assertEqual(source.queue_metrics()["resident_bytes"], 0)
                self.assertFalse(source._queue.metrics()["producer_alive"])

    def run_queue_regression(self, name):
        from unittest.mock import patch

        from tests import test_shared_target_queue as queue_tests

        with patch.object(
            queue_tests,
            "online_fixture",
            side_effect=lambda root, seeded: self.configure(root, True),
        ):
            getattr(queue_tests.SharedTargetQueueCPU, name)(self)

    def test_async_epoch_boundary_retains_accumulation_credit(self):
        self.run_queue_regression("test_epoch_boundary_keeps_inflight_credit")

    def test_async_backward_failure_joins_producer_without_acknowledging(self):
        self.run_queue_regression(
            "test_failed_backward_releases_queue_without_acknowledging"
        )

    def test_async_verification_policy_stop_and_resume(self):
        self.run_queue_regression("test_control_stops_and_preserves_resume_patience")

    def test_async_observer_handoff_restores_and_continues_training(self):
        from unittest.mock import patch

        from tests import test_joint_serving_hook as serving_tests

        original_setup = serving_tests.fixtures.UnattendedControllerTest.setUp

        def setup(fixture):
            original_setup(fixture)
            path = fixture.root / "base.json"
            config = json.loads(path.read_text())
            config["training"]["joint_async_capture"] = True
            path.write_text(json.dumps(config))

        with patch.object(
            serving_tests.fixtures.UnattendedControllerTest, "setUp", setup
        ):
            serving_tests.JointServingHookTest.test_milestone_handoff_preserves_next_update_and_committed_cursor(
                self
            )

    def test_configuration_requires_bounded_builtin_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.configure(root, True)
            payload = json.loads((root / "base.json").read_text())
            self.assertTrue(Config.model_validate(payload).training.joint_async_capture)
            for section, key, value in (
                ("training", "joint_feature_queue_bytes", 0),
                ("training", "joint_online_capture", False),
                ("model", "trust_remote_code", True),
            ):
                changed = copy.deepcopy(payload)
                changed[section][key] = value
                with self.subTest(key=key), self.assertRaises(ValidationError):
                    Config.model_validate(changed)


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("DRAFTFIT_RUN_QUEUE_CUDA") == "1",
    "explicit same-GPU async capture gate",
)
class JointAsyncCaptureCUDA(unittest.TestCase):
    execution_device = "cuda"
    cli = online_tests.JointOnlineCPU.cli
    configure = JointAsyncCaptureTests.configure
    assert_nested_equal = JointAsyncCaptureTests.assert_nested_equal
    test_real_cli_updates_match_sync_and_resume_exactly = (
        JointAsyncCaptureTests.test_real_cli_updates_match_sync_and_resume_exactly
    )
