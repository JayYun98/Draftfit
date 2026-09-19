"""Owned teacher HTTP/sink boundary checks; fake Mooncake is not RDMA validation."""

import copy
import http.client
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from draftfit.inference.capture_sink import CaptureSink
from draftfit.inference.teacher_server import TeacherService, make_server
from draftfit.offline_capture.sglang import OfflineCaptureBatch
from draftfit.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore


class NativeTeacherBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.raw = _FakeMooncakeStore()
        self.transport = MooncakeFeatureStore(store=self.raw, store_id="run")
        self.sink = CaptureSink(self.transport, aux_layer_ids=[0])
        self.spec = {
            "store_id": "run",
            "sample_id": "run:task",
            "gen": 1,
            "features": {"aux": "hidden_states", "last_hidden": "target"},
            "passthrough": [
                {"name": "input_ids", "data": [1, 2], "shape": [1, 2], "dtype": "int64"}
            ],
        }
        self.teacher = Mock(config=SimpleNamespace(hidden_size=2), capture_layers=[0])
        self.teacher.capture.side_effect = lambda **kw: OfflineCaptureBatch(
            torch.ones(1, 2, 2), torch.ones(1, 2, 2), **kw
        )
        self.service = TeacherService(
            teacher=self.teacher,
            sink=self.sink,
            max_model_len=8,
            backend="transformers",
            target_model="fixture",
            target_revision="local",
        )
        self.body = {
            "input_ids": [[1, 2]],
            "spec_capture": [self.spec],
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        }

    def test_actual_http_limits_and_metadata(self):
        server = make_server(self.service, port=0, max_body_bytes=1024)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)

        def request(method, path, body=None, headers=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            try:
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        code, health = request("GET", "/health")
        self.assertEqual(code, 200)
        self.assertEqual(health["target_model"], "fixture")
        code, rows = request(
            "POST",
            "/generate",
            json.dumps(self.body),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(code, 200)
        self.assertEqual(
            rows[0]["meta_info"]["spec_capture"]["backend"], "transformers"
        )
        self.assertEqual(len(self.raw._d), 3)
        self.assertEqual(
            request(
                "POST", "/generate", "x" * 1025, {"Content-Type": "application/json"}
            )[0],
            413,
        )
        self.assertEqual(
            request(
                "POST",
                "/generate",
                "{}",
                {"Content-Type": "application/json", "Origin": "https://example.com"},
            )[0],
            400,
        )
        self.assertEqual(
            request("POST", "/generate", "{}", {"Content-Type": "text/plain"})[0], 400
        )
        self.teacher.capture.assert_called_once()

    def test_namespace_schema_and_all_rows_validated_before_work(self):
        for key, value in (
            ("store_id", "other"),
            ("sample_id", "../escape"),
            ("gen", True),
            ("gen", 2),
        ):
            spec = copy.deepcopy(self.spec)
            spec[key] = value
            with self.assertRaises(ValueError):
                self.sink.validate(spec, 2)
        spec = copy.deepcopy(self.spec)
        spec["passthrough"][0]["data"] = [1.5, 2]
        with self.assertRaises(ValueError):
            self.sink.validate(spec, 2)
        bad = copy.deepcopy(self.body)
        bad["input_ids"].append([])
        bad["spec_capture"].append(self.spec)
        with self.assertRaises(ValueError):
            self.service.generate(bad)
        self.teacher.capture.assert_not_called()
        self.assertFalse(self.raw._d)
        with self.assertRaises(ValueError):
            make_server(self.service, host="0.0.0.0", port=0)

    def test_partial_write_failure_cleanup_and_retry_same_generation(self):
        original = self.raw.put_from

        def fail_second(*args):
            if self.raw.put_calls == 1:
                return -1
            return original(*args)

        self.raw.put_from = fail_second
        with self.assertRaises(RuntimeError):
            self.sink.put_sample(
                self.spec, aux=torch.ones(2, 2), last_hidden=torch.ones(2, 2)
            )
        self.assertFalse(self.raw._d)
        self.raw.put_from = original
        self.sink.put_sample(
            self.spec, aux=torch.ones(2, 2), last_hidden=torch.ones(2, 2)
        )
        keys = set(self.raw._d)
        retry = {**self.spec, "replace": True}
        self.sink.put_sample(
            retry, aux=torch.zeros(2, 2), last_hidden=torch.zeros(2, 2)
        )
        self.assertEqual(keys, set(self.raw._d))
        self.assertTrue(all("/g1/" in key for key in keys))
        self.raw.fail_remove = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.sink.put_sample(
                retry, aux=torch.ones(2, 2), last_hidden=torch.ones(2, 2)
            )

    def test_hard_pin_and_model_shape_enforced(self):
        self.transport._put_config.with_hard_pin = False
        with self.assertRaisesRegex(ValueError, "hard pin"):
            CaptureSink(self.transport, aux_layer_ids=[0])
        self.transport._put_config.with_hard_pin = True
        self.teacher.capture.side_effect = None
        self.teacher.capture.return_value = OfflineCaptureBatch(
            torch.ones(1, 1, 2),
            torch.ones(1, 2, 2),
            torch.tensor([[1, 2]]),
            torch.ones(1, 2),
            torch.ones(1, 2),
        )
        with self.assertLogs("draftfit.inference.teacher_server", level="ERROR"):
            rows = self.service.generate(self.body)
        self.assertIn("error", rows[0]["meta_info"]["spec_capture"])
        self.assertFalse(self.raw._d)

    def test_retry_cleanup_does_not_refresh_mooncake_read_leases(self):
        class LeaseStore(_FakeMooncakeStore):
            def __init__(self):
                super().__init__()
                self.leases = {}
                self.probes = 0

            def is_exist(self, key):
                self.probes += 1
                if key in self._d:
                    self.leases[key] = 1
                return super().is_exist(key)

            def remove(self, key):
                # Lease persists across requests until the test expires it.
                if self.leases.get(key, 0):
                    return -706
                if key not in self._d:
                    return -704
                return super().remove(key)

        raw = LeaseStore()
        sink = CaptureSink(
            MooncakeFeatureStore(store=raw, store_id="run"), aux_layer_ids=[0]
        )
        sink.put_sample(self.spec, aux=torch.ones(2, 2), last_hidden=torch.ones(2, 2))
        keys = set(raw._d)
        for key in keys:
            raw.is_exist(key)
        probes = raw.probes
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                sink.put_sample(
                    {**self.spec, "replace": True},
                    aux=torch.zeros(2, 2),
                    last_hidden=torch.zeros(2, 2),
                )
            self.assertEqual(raw.probes, probes)
            self.assertEqual(set(raw._d), keys)
        raw.leases.clear()
        sink.put_sample(
            {**self.spec, "replace": True},
            aux=torch.zeros(2, 2),
            last_hidden=torch.zeros(2, 2),
        )
        self.assertEqual(raw.probes, probes)
        self.assertEqual(set(raw._d), keys)
        sink._remove([*keys, "run/absent/g1/hidden_states"])
        self.assertFalse(raw._d)
        self.assertEqual(raw.probes, probes)  # No existence probe even for absence.

    def test_remove_missing_ok_accepts_only_documented_absence(self):
        for status in (None, 0, -704, -706, -1, -9999):
            with self.subTest(status=status):
                self.raw.remove = Mock(return_value=status)
                self.assertEqual(
                    self.transport._store_remove("key"), status in (None, 0)
                )
                self.assertEqual(
                    self.transport._store_remove("key", missing_ok=True),
                    status in (None, 0, -704),
                )
        self.raw.remove = Mock(side_effect=OSError("transport failed"))
        self.assertFalse(self.transport._store_remove("key", missing_ok=True))
        self.raw.is_exist = Mock(side_effect=AssertionError("must not probe"))
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.sink._remove(["key"])
        self.raw.is_exist.assert_not_called()

    def test_interrupt_during_put_removes_partial_attempt(self):
        original = self.raw.put_from

        def interrupt_second(*args):
            if self.raw.put_calls == 1:
                raise KeyboardInterrupt("interrupted store write")
            return original(*args)

        self.raw.put_from = interrupt_second
        with self.assertRaises(KeyboardInterrupt):
            self.sink.put_sample(
                self.spec, aux=torch.ones(2, 2), last_hidden=torch.ones(2, 2)
            )
        self.assertFalse(self.raw._d)


if __name__ == "__main__":
    unittest.main()
