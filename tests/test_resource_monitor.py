"""Hardware fakes exercise actual sampler threads, subprocess parsing and policy IO."""

import json
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from draftfit.training.resource_monitor import ResourceSampler, _CudaProbe, _Unavailable
from draftfit.training.resource_policy import decide
from tests.test_resource_policy import admission, config, current

GPU_A = "GPU-11111111-2222-3333-4444-555555555555"
GPU_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def hardware(
    *, uuid=GPU_A, visible=None, output=None, peak=50, available=True, index=0
):
    cuda = SimpleNamespace(
        is_available=Mock(return_value=available),
        current_device=Mock(return_value=index),
        device_count=Mock(return_value=2),
        get_device_properties=Mock(
            return_value=SimpleNamespace(uuid=uuid, total_memory=100)
        ),
        max_memory_allocated=Mock(return_value=peak),
    )

    def parse_device(device):
        parts = str(device).split(":")
        return SimpleNamespace(
            type=parts[0], index=int(parts[1]) if len(parts) > 1 else None
        )

    torch = SimpleNamespace(cuda=cuda, device=parse_device)
    run = Mock(
        return_value=SimpleNamespace(
            stdout=output if output is not None else f"{uuid}, 40\n"
        )
    )
    env = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
    return torch, run, env


def probe_fixture(**kwargs):
    torch, run, env = hardware(**kwargs)
    probe = _CudaProbe(torch_module=torch, run_command=run, environ=env)
    return probe, torch.cuda, run


def summarize(sampler, **extra):
    return sampler.summarize(
        **(
            dict(phase_id="phase-0", preset="base", step=12, throughput=20.0, loss=1.0)
            | extra
        )
    )


def wait_samples(sampler, count):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with sampler._lock:
            if sampler._count >= count:
                return
            error = sampler._error
        if error:
            raise AssertionError(error)
        time.sleep(0.001)
    raise AssertionError("sampler did not produce expected samples")


class CudaProbeTests(unittest.TestCase):
    def test_uuid_overrides_logical_ordinal_and_smi_is_bounded(self):
        probe, cuda, run = probe_fixture(visible="7,3", index=1)
        row = probe.sample()
        self.assertEqual(row["cuda_index"], 1)
        self.assertEqual(row["gpu_uuid"], GPU_A)
        self.assertEqual(row["utilization"], 0.4)
        self.assertEqual(row["capacity_bytes"], 100)
        cuda.get_device_properties.assert_called_once_with(1)
        cuda.max_memory_allocated.assert_called_once_with(1)
        args, kwargs = run.call_args
        self.assertIn(f"--id={GPU_A}", args[0])
        self.assertEqual(kwargs["timeout"], 1.0)
        self.assertTrue(kwargs["check"])
        self.assertNotIn("shell", kwargs)

    def test_full_visible_uuid_resolves_reordered_cuda_index(self):
        probe, cuda, run = probe_fixture(
            uuid=None, visible=f"{GPU_B},{GPU_A}", index=1, output=f"{GPU_A}, 60"
        )
        self.assertEqual(probe.sample()["gpu_uuid"], GPU_A)
        self.assertEqual(cuda.get_device_properties.call_args.args, (1,))

    def test_torch_uuid_object_and_abbreviated_visible_uuid(self):
        class UUID:
            def __str__(self):
                return GPU_A

        probe, _, _ = probe_fixture(uuid=UUID(), visible="GPU-11111111")
        self.assertEqual(probe.sample()["gpu_uuid"], GPU_A)

    def test_real_torch_cuuuid_without_nvidia_prefix(self):
        # Recorded Torch 2.13/cu130 A100 UUID shape: _CUuuid.__str__ omits GPU-.
        bare = "23d0e211-350d-ab34-5a61-eb7ec4d9372e"

        class CUuuid:
            def __str__(self):
                return bare

        for visible in (None, "0", "GPU-23d0e211"):
            with self.subTest(visible=visible):
                probe, _, run = probe_fixture(
                    uuid=CUuuid(), visible=visible, output=f"GPU-{bare}, 0"
                )
                self.assertEqual(probe.sample()["gpu_uuid"], "GPU-" + bare)
                self.assertIn("--id=GPU-" + bare, run.call_args.args[0])
        with self.assertRaisesRegex(_Unavailable, "conflict"):
            probe_fixture(uuid=CUuuid(), visible=GPU_B)
        with self.assertRaisesRegex(_Unavailable, "mig"):
            probe_fixture(uuid=CUuuid(), visible="MIG-" + bare)
        with self.assertRaisesRegex(_Unavailable, "mismatch"):
            probe_fixture(uuid=CUuuid(), output=f"{GPU_B}, 0")[0].sample()

    def test_bare_uuid_normalization_requires_exact_complete_uuid(self):
        for value in (
            "23d0e211",
            "23d0e211350dab345a61eb7ec4d9372e",
            " 23d0e211-350d-ab34-5a61-eb7ec4d9372e",
            "23d0e211-350d-ab34-5a61-eb7ec4d9372e\n",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(_Unavailable, "identity_unavailable"),
            ):
                probe_fixture(uuid=value)

    def test_unknown_numeric_empty_and_mig_identity_fail_closed(self):
        for visible in (None, "3,7", "", "GPU-11111111", "MIG-GPU-111/1/2"):
            with self.subTest(visible=visible), self.assertRaises(_Unavailable):
                probe_fixture(uuid=None, visible=visible)
        with self.assertRaisesRegex(_Unavailable, "conflict"):
            probe_fixture(visible=GPU_B)

    def test_uuid_mismatch_multiple_rows_and_invalid_utilization(self):
        for output in (
            f"{GPU_B}, 40",
            f"{GPU_A}, 40\n{GPU_A}, 50",
            "",
            "N/A",
            f"{GPU_A}, N/A",
            f"{GPU_A}, nan",
            f"{GPU_A}, inf",
            f"{GPU_A}, -1",
            f"{GPU_A}, 101",
        ):
            with (
                self.subTest(output=output),
                self.assertRaises((ValueError, _Unavailable)),
            ):
                probe_fixture(output=output)[0].sample()

    def test_non_cuda_and_unavailable_do_not_query_smi(self):
        for device, available in (("cpu", True), (None, False)):
            torch, run, env = hardware(available=available)
            with self.subTest(device=device), self.assertRaises(_Unavailable):
                _CudaProbe(device, torch_module=torch, run_command=run, environ=env)
            run.assert_not_called()
            torch.cuda.get_device_properties.assert_not_called()

    def test_invalid_capacity_peak_and_device_index_rejected(self):
        for peak in (-1, float("nan"), float("inf"), 101):
            with self.subTest(peak=peak), self.assertRaises(_Unavailable):
                probe_fixture(peak=peak)[0].sample()
        for index in (-1, 2):
            with self.subTest(index=index), self.assertRaises(_Unavailable):
                probe_fixture(index=index)
        torch, run, env = hardware()
        torch.cuda.get_device_properties.return_value.total_memory = 0
        with self.assertRaises(_Unavailable):
            _CudaProbe(torch_module=torch, run_command=run, environ=env)


class ResourceSamplerTests(unittest.TestCase):
    def make_sampler(self, probe=None, **kwargs):
        probe = probe or probe_fixture()[0]
        sampler = ResourceSampler(
            interval_seconds=0.003, _probe_factory=lambda device: probe, **kwargs
        )
        self.addCleanup(sampler.stop)
        return sampler

    def assert_unavailable(self, result, reason=None):
        self.assertEqual(result["status"], "measurement_unavailable")
        self.assertIsNone(result["observation"])
        self.assertTrue(result["reason"])
        if reason is not None:
            self.assertEqual(result["reason"], reason)
        json.dumps(result, allow_nan=False)

    def test_thread_ring_summary_scopes_and_policy_consumption(self):
        probe, cuda, run = probe_fixture()
        sampler = self.make_sampler(probe, max_samples=3).start()
        wait_samples(sampler, 6)
        sampler.stop()
        result = summarize(sampler)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["samples"], 3)
        self.assertGreaterEqual(result["samples_dropped"], 3)
        self.assertFalse(sampler._thread.is_alive())
        observation = result["observation"]
        self.assertEqual(observation["memory_scope"], "process_torch_allocator")
        self.assertEqual(observation["utilization_scope"], "device")
        self.assertEqual(
            observation["peak_window"], "since_last_external_allocator_reset"
        )
        self.assertEqual(observation["peak_vram_bytes"], 50)
        self.assertEqual(observation["utilization"], 0.4)
        self.assertLessEqual(
            observation["sample_start_monotonic_seconds"],
            observation["sample_end_monotonic_seconds"],
        )
        observations = [observation | dict(step=step) for step in (10, 11, 12)]
        decision = decide(
            config(),
            observations,
            current=current(),
            admission=admission(),
            state=dict(transitions=0, last_transition_step=None),
        )
        self.assertEqual(decision["next_preset"], "large")
        json.dumps(result, allow_nan=False)

    def test_context_manager_cleans_thread_even_when_body_raises(self):
        sampler = self.make_sampler()
        with self.assertRaisesRegex(RuntimeError, "training failure"):
            with sampler:
                wait_samples(sampler, 2)
                raise RuntimeError("training failure")
        self.assertFalse(sampler._thread.is_alive())
        self.assertEqual(summarize(sampler)["status"], "available")

    def test_stop_reads_final_allocator_peak_instead_of_returning_old_sample(self):
        probe, cuda, _ = probe_fixture()
        sampler = ResourceSampler(
            interval_seconds=60, min_samples=1, _probe_factory=lambda device: probe
        ).start()
        self.addCleanup(sampler.stop)
        wait_samples(sampler, 1)
        cuda.max_memory_allocated.return_value = 90
        sampler.stop()
        self.assertFalse(sampler._thread.is_alive())
        self.assertEqual(summarize(sampler)["observation"]["peak_vram_bytes"], 90)

    def test_caller_device_is_resolved_before_starting_worker(self):
        torch, run, env = hardware(index=1)
        owner = threading.get_ident()

        def current_device():
            self.assertEqual(threading.get_ident(), owner)
            return 1

        torch.cuda.current_device.side_effect = current_device
        sampler = ResourceSampler(
            interval_seconds=0.003,
            _probe_factory=lambda device: _CudaProbe(
                device, torch_module=torch, run_command=run, environ=env
            ),
        )
        self.addCleanup(sampler.stop)
        with sampler:
            wait_samples(sampler, 2)
        self.assertEqual(summarize(sampler)["observation"]["cuda_index"], 1)

    def test_sampling_errors_poison_prior_good_samples(self):
        for error in (
            FileNotFoundError(),
            subprocess.TimeoutExpired("nvidia-smi", 1),
            subprocess.CalledProcessError(1, "nvidia-smi"),
            RuntimeError("driver"),
        ):
            with self.subTest(error=type(error).__name__):
                probe, _, run = probe_fixture()
                run.side_effect = [SimpleNamespace(stdout=f"{GPU_A}, 40"), error]
                sampler = self.make_sampler(probe, min_samples=1).start()
                self.assertTrue(sampler._stop.wait(1))
                sampler.stop()
                self.assert_unavailable(summarize(sampler), type(error).__name__)
                self.assertFalse(sampler._thread.is_alive())

    def test_start_failure_is_explicit_and_stop_is_idempotent(self):
        sampler = ResourceSampler(
            _probe_factory=Mock(side_effect=_Unavailable("cuda_unavailable"))
        )
        sampler.start().stop().stop()
        self.assert_unavailable(summarize(sampler), "cuda_unavailable")
        with self.assertRaises(RuntimeError):
            sampler.start()

    def test_native_hang_bounded_stop_and_late_result_discarded(self):
        entered, release = threading.Event(), threading.Event()
        probe, _, run = probe_fixture()

        def blocked(*args, **kwargs):
            entered.set()
            release.wait(5)
            return SimpleNamespace(stdout=f"{GPU_A}, 40")

        run.side_effect = blocked
        sampler = self.make_sampler(probe, min_samples=1).start()
        self.assertTrue(entered.wait(1))
        try:
            started = time.monotonic()
            sampler.stop()
            self.assertLess(time.monotonic() - started, 2.0)
            self.assert_unavailable(summarize(sampler), "worker_stop_timeout")
        finally:
            release.set()
            sampler.stop()
        self.assertFalse(sampler._thread.is_alive())
        self.assertEqual(summarize(sampler)["samples"], 0)

    def test_lifecycle_insufficient_samples_and_rebinding(self):
        sampler = self.make_sampler()
        self.assert_unavailable(summarize(sampler), "sampler_not_started")
        sampler.start()
        self.assert_unavailable(summarize(sampler), "sampler_not_stopped")
        wait_samples(sampler, 2)
        sampler.stop()
        first = summarize(sampler)
        self.assertEqual(first, summarize(sampler))
        self.assert_unavailable(
            summarize(sampler, phase_id="next"), "sampler_already_bound"
        )
        self.assert_unavailable(summarize(sampler, step=13), "sampler_already_bound")
        empty = self.make_sampler(min_samples=100, max_samples=100).start()
        empty.stop()
        self.assert_unavailable(summarize(empty), "insufficient_samples")

    def test_old_interval_start_allowed_but_stale_last_sample_rejected(self):
        sampler = self.make_sampler(max_age_seconds=1).start()
        wait_samples(sampler, 2)
        sampler.stop()
        with sampler._lock:
            sampler._rows[0]["monotonic_seconds"] -= 100
        self.assertEqual(summarize(sampler)["status"], "available")
        with patch(
            "draftfit.training.resource_monitor.time.monotonic",
            return_value=time.monotonic() + 10,
        ):
            self.assert_unavailable(summarize(sampler), "stale_samples")

    def test_zero_real_util_or_allocator_peak_remains_available(self):
        for options in (dict(output=f"{GPU_A}, 0"), dict(peak=0)):
            sampler = self.make_sampler(probe_fixture(**options)[0]).start()
            wait_samples(sampler, 2)
            sampler.stop()
            result = summarize(sampler)
            self.assertEqual(result["status"], "available")
            key = "utilization" if "output" in options else "peak_vram_bytes"
            self.assertEqual(result["observation"][key], 0)

    def test_zero_observations_reach_policy_through_real_sampler_threads(self):
        for throughput in (20.0, 0.0):
            with self.subTest(throughput=throughput):
                observations = []
                for step in (10, 11, 12):
                    sampler = self.make_sampler(
                        probe_fixture(output=f"{GPU_A}, 0", peak=0)[0]
                    ).start()
                    wait_samples(sampler, 2)
                    sampler.stop()
                    result = summarize(
                        sampler, step=step, loss=0.0, throughput=throughput
                    )
                    self.assertEqual(result["status"], "available")
                    observations.append(result["observation"])
                decision = decide(
                    config(),
                    observations,
                    current=current(),
                    admission=admission(),
                    state=dict(transitions=0, last_transition_step=None),
                )
                self.assertEqual(decision["rejected"], [])
                self.assertEqual(
                    decision["reason"],
                    "underutilized_window" if throughput else "zero_throughput",
                )
                self.assertEqual(
                    decision["next_preset"], "large" if throughput else None
                )
                json.dumps(observations, allow_nan=False)

    def test_caller_invalid_loss_throughput_and_binding(self):
        sampler = self.make_sampler().start()
        wait_samples(sampler, 2)
        sampler.stop()
        for key in ("loss", "throughput"):
            for value in (float("nan"), float("inf"), -1, True):
                with self.subTest(key=key, value=value):
                    self.assert_unavailable(
                        summarize(sampler, **{key: value}), "invalid_metric"
                    )
        for change in (
            dict(step=0),
            dict(step=True),
            dict(phase_id=""),
            dict(preset=None),
        ):
            self.assert_unavailable(
                summarize(sampler, **change), "invalid_milestone_binding"
            )

    def test_real_cpu_explicit_unavailable(self):
        # Actual installed torch if present; CPU is explicitly selected even on CUDA hosts.
        sampler = ResourceSampler(device="cpu").start().stop()
        result = summarize(sampler)
        self.assert_unavailable(result)
        self.assertIn(result["reason"], ("non_cuda_device", "ModuleNotFoundError"))
        self.assertIsNone(sampler._thread)


if __name__ == "__main__":
    unittest.main()
