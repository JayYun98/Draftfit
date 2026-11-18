"""Live CPU feature storage when a joint consumer returns capture credit."""

import json
import os
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from draftfit.family import build_family_plan
from draftfit.research.experiments.async_joint_capture.queue import AsyncFeatureQueue
from draftfit.training.feature_queue import BoundedFeatureQueue, tensor_bytes
from draftfit.training.joint_family import run_joint_family
from tests.test_joint_online_cpu import online_fixture


class AsyncCaptureCreditTests(unittest.TestCase):
    def check_live_payload(self, *, probe_steps=None):
        budget = 1728  # Space for one tiny Qwen feature batch and its reservation.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}),
        ):
            root = Path(directory)
            online_fixture(root, seeded=False)
            path = root / "base.json"
            config = json.loads(path.read_text())
            config["training"].update(
                joint_async_capture=True,
                joint_feature_queue_bytes=budget,
                joint_feature_queue_batches=1,
                accumulation_steps=1,
            )
            path.write_text(json.dumps(config))
            _, stages = build_family_plan(str(root / "family.json"))
            for stage in stages:
                path = Path(stage.config.model.draft_model_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                stage.draft_config.to_json_file(str(path))

            first_payload = []
            live_bytes = []
            published = threading.Event()
            publish = BoundedFeatureQueue.publish
            release = AsyncFeatureQueue.release

            def observe_publish(queue, value, size):
                sequence = publish(queue, value, size)
                if sequence == 0:
                    first_payload.extend(weakref.ref(t) for t in value.tensors.values())
                elif sequence == 1:
                    live_bytes.append(
                        tensor_bytes([value.tensors, [ref() for ref in first_payload]])
                    )
                    published.set()
                return sequence

            def release_before_consumer_continues(queue, lease, *, consumed):
                release(queue, lease, consumed=consumed)
                if lease.sequence == 0:
                    self.assertTrue(published.wait(5), "producer did not reuse credit")

            with (
                patch.object(BoundedFeatureQueue, "publish", observe_publish),
                patch.object(
                    AsyncFeatureQueue, "release", release_before_consumer_continues
                ),
            ):
                run_joint_family(root / "outputs", stages, probe_steps=probe_steps)
            self.assertEqual(len(live_bytes), 1)
            self.assertLessEqual(live_bytes[0], budget)

    def test_trainer_releases_live_payload_before_credit(self):
        self.check_live_payload()

    def test_calibration_releases_live_payload_before_credit(self):
        self.check_live_payload(probe_steps=2)
