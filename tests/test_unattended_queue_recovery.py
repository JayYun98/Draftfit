"""Real CPU capture/update recovery, including failures before consumer ACK.

Only failure boundaries are injected; Qwen capture, three family losses,
backward, AdamW, checkpoint serialization and continuation execute normally.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from draftfit.optimizer import BF16Optimizer
from draftfit.training import joint_family
from draftfit.training.joint_online import JointOnlineSource
from tests.test_joint_deadline import stages_for
from tests.test_joint_online_cpu import online_fixture


class UnattendedQueueRecoveryTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        env = mock.patch.dict(
            os.environ,
            {
                "SPECFORGE_DEVICE": "cpu",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        directory = tempfile.TemporaryDirectory(prefix="unattended-queue-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        online_fixture(self.root, seeded=False)
        # Three microbatches per epoch and accumulation=2 force an update
        # to own leases from both epochs. One additional item is lookahead.
        path = self.root / "train.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:3]) + "\n")
        path = self.root / "base.json"
        base = json.loads(path.read_text())
        base["training"].update(
            num_epochs=2,
            max_steps=3,
            total_steps=3,
            joint_feature_queue_bytes=5184,
            joint_feature_queue_batches=3,
        )
        path.write_text(json.dumps(base))

    def prepare(self, name):
        output = self.root / name
        stages = stages_for(self.root, output)
        for stage in stages:
            stage.config.training.save_interval = 1
        source = JointOnlineSource(stages[0], needs_dspark=True)
        self.assertEqual(source._batch_bytes(source._rows[False], [0]), 1728)
        return output, stages, source

    def run_training(self, prepared, **kwargs):
        output, stages, source = prepared
        consumed = []
        original = source.consume

        def consume(batches):
            original(batches)
            consumed.extend(sample for batch in batches for sample in batch.sample_ids)

        with (
            mock.patch(
                "draftfit.training.joint_online.JointOnlineSource", return_value=source
            ),
            mock.patch.object(source, "consume", side_effect=consume),
        ):
            result = joint_family.run_joint_family(output, stages, **kwargs)
        self.assertEqual({r["status"] for r in result["runs"]}, {"complete"})
        self.assert_closed(source)
        return consumed

    def assert_closed(self, source):
        metrics = source.queue_metrics()
        self.assertEqual((metrics["resident_bytes"], metrics["resident_items"]), (0, 0))
        self.assertEqual(
            metrics["produced"], metrics["consumed"] + metrics["cancelled"]
        )
        self.assertLessEqual(metrics["peak_bytes"], source._queue_bytes)
        self.assertEqual(source._leases, {})

    def assert_exact(self, expected, actual, path="state"):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertTrue(torch.equal(expected, actual), path)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys(), path)
            for key in expected:
                self.assert_exact(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_exact(left, right, f"{path}[{index}]")
        else:
            self.assertEqual(expected, actual, path)

    def load(self, output, step):
        return torch.load(
            output / f"joint-step{step}.pt", weights_only=True, map_location="cpu"
        )

    def test_prefetch_stop_and_cross_epoch_resume_restore_full_optimizer(self):
        full = self.prepare("full")
        expected_samples = self.run_training(full)
        expected = self.load(full[0], 3)
        self.assertEqual(len(expected_samples), 6)
        self.assertEqual(full[2].queue_metrics()["consumed"], 6)
        initial = self.load(full[0], 0)
        key = "layers.0.self_attn.q_proj.weight"
        for arm in expected["drafts"]:
            self.assertFalse(
                torch.equal(initial["drafts"][arm][key], expected["drafts"][arm][key])
            )
            self.assertTrue(
                torch.equal(
                    expected["drafts"]["dspark"][key], expected["drafts"][arm][key]
                )
            )

        for stop in (1, 2):
            with self.subTest(stop=stop):
                partial = self.prepare(f"partial-{stop}")
                prefix = self.run_training(partial, stop_at_step=stop)
                saved = self.load(partial[0], stop)
                self.assertEqual(
                    (saved["epoch"], saved["next_batch"]), divmod(stop * 2, 3)
                )
                self.assertEqual(saved["consumed_samples"], stop * 2)
                self.assertGreater(partial[2].queue_metrics()["cancelled"], 0)
                resumed = self.prepare(f"resumed-{stop}")
                suffix = self.run_training(
                    resumed, resume_from=partial[0] / f"joint-step{stop}.pt"
                )
                self.assertEqual(prefix + suffix, expected_samples)
                self.assertEqual(resumed[2].queue_metrics()["produced"], len(suffix))
                # Includes FP32 master parameters, Adam moments, scheduler,
                # RNG, shared/private weights, and the consumed data cursor.
                self.assert_exact(expected, self.load(resumed[0], 3))

    def test_capture_backward_and_optimizer_failure_replay_last_durable_cursor(self):
        full = self.prepare("full")
        expected_samples = self.run_training(full)
        expected = self.load(full[0], 3)
        for boundary in ("capture", "backward", "optimizer"):
            with self.subTest(boundary=boundary):
                failed = self.prepare(f"failed-{boundary}")
                output, stages, source = failed
                calls = 0
                if boundary == "capture":
                    original = source._capture_batch

                    def operation(rows, indices, evaluation):
                        nonlocal calls
                        if not evaluation:
                            calls += 1
                            if calls == 4:
                                # First epoch's last item is still leased.
                                self.assertGreater(
                                    source.queue_metrics()["resident_bytes"], 0
                                )
                                raise RuntimeError("injected capture failure")
                        return original(rows, indices, evaluation)

                    failure = mock.patch.object(
                        source, "_capture_batch", side_effect=operation
                    )
                else:
                    owner, name = (
                        (joint_family, "_backward_window")
                        if boundary == "backward"
                        else (BF16Optimizer, "step")
                    )
                    original = getattr(owner, name)

                    def operation(*args, **kwargs):
                        nonlocal calls
                        result = original(*args, **kwargs)
                        calls += 1
                        if calls == 2:
                            # Even a mutated in-memory optimizer must not ACK
                            # its window when the call fails before returning.
                            raise RuntimeError(f"injected {boundary} failure")
                        return result

                    failure = mock.patch.object(owner, name, operation)
                with (
                    mock.patch(
                        "draftfit.training.joint_online.JointOnlineSource",
                        return_value=source,
                    ),
                    failure,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"injected {boundary} failure"
                    ):
                        joint_family.run_joint_family(output, stages)
                self.assert_closed(source)
                self.assertEqual(source.consumed_samples, 2)
                self.assertEqual(source.queue_metrics()["consumed"], 2)
                self.assertGreater(source.queue_metrics()["cancelled"], 0)
                self.assertFalse((output / "joint-step2.pt").exists())
                self.assert_exact(self.load(full[0], 1), self.load(output, 1))
                resumed = self.prepare(f"recovered-{boundary}")
                suffix = self.run_training(
                    resumed, resume_from=output / "joint-step1.pt"
                )
                self.assertEqual(suffix, expected_samples[2:])
                self.assert_exact(expected, self.load(resumed[0], 3))

    def test_insufficient_byte_budget_cancels_unacknowledged_window(self):
        output, stages, _ = self.prepare("too-small")
        for stage in stages:
            stage.config.training.joint_feature_queue_bytes = 1728
        source = JointOnlineSource(stages[0], needs_dspark=True)
        with mock.patch(
            "draftfit.training.joint_online.JointOnlineSource", return_value=source
        ):
            with self.assertRaisesRegex(ValueError, "cannot hold accumulation window"):
                joint_family.run_joint_family(output, stages)
        self.assert_closed(source)
        self.assertEqual(source.queue_metrics()["consumed"], 0)
        self.assertEqual(source.queue_metrics()["cancelled"], 1)
        self.assertGreater(source.queue_metrics()["backpressure"], 0)
        self.assertEqual(self.load(output, 0)["consumed_samples"], 0)
        self.assertFalse((output / "joint-step1.pt").exists())


if __name__ == "__main__":
    unittest.main()
