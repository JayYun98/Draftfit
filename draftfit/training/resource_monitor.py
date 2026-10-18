"""Read-only CUDA telemetry for one milestone interval; never executes advice.

Usage: sampler = ResourceSampler(device="cuda:0").start(); do training;
sampler.stop(); sampler.summarize(phase_id=..., preset=..., step=...,
throughput=..., loss=...). Always stop in finally, or use the context manager.
Use a NEW sampler for each milestone/phase. Summaries require a stopped sampler,
enough recent samples, positive capacity, and finite nonnegative policy metrics.
Measured zero utilization, allocator peak, throughput or loss remains available;
resource_policy decides whether no progress permits tuning. CPU/missing tools/failures
return status=measurement_unavailable with observation=None, never guessed data.

The observation is directly consumable by resource_policy. Utilization is a
sample mean of DEVICE-wide GPU busy percentages, not process utilization. Memory
is the current process's torch allocator peak, not total device VRAM usage; CUDA
context/non-torch/other-process allocations are excluded. Peak counters are NOT
reset here: peak_window is since_last_external_allocator_reset, which can precede
this interval. Only an owner resetting those counters before start can claim an
interval allocator peak (and must not reset again during collection). The bounded ring's timestamps describe utilization
coverage, and samples_dropped reports truncated coverage.

GPU identity uses torch properties' UUID (with or without NVIDIA's GPU- prefix),
or a full UUID in CUDA_VISIBLE_DEVICES.
Unknown/numeric-only fallback mappings and MIG are unavailable, never silently
mapped to another physical GPU. nvidia-smi verifies the returned UUID. Its calls
time out after one second. stop joins for at most 1.25 seconds and reports timeout;
Python cannot cancel a stuck native CUDA call, so the worker is daemonized and
late results are discarded. No torch import or hardware access occurs on import.
"""
from __future__ import annotations

from collections import deque
import importlib
import math
import os
import re
import subprocess
import threading
import time

_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class _Unavailable(RuntimeError):
    pass


def _finite(value, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0):
        raise _Unavailable("invalid_metric")
    return value


class _CudaProbe:
    def __init__(self, device=None, *, torch_module=None, run_command=None, environ=None):
        torch = torch_module if torch_module is not None else importlib.import_module("torch")
        self.cuda = torch.cuda
        self.run = run_command or subprocess.run
        env = os.environ if environ is None else environ
        if device is not None and type(device) is not int:
            parsed = torch.device(device)
            if parsed.type != "cuda":
                raise _Unavailable("non_cuda_device")
            device = parsed.index
        if not self.cuda.is_available():
            raise _Unavailable("cuda_unavailable")
        # Resolve current_device in the CALLER thread; CUDA device state is thread-local.
        self.index = self.cuda.current_device() if device is None else device
        if type(self.index) is not int or not 0 <= self.index < self.cuda.device_count():
            raise _Unavailable("invalid_cuda_index")
        props = self.cuda.get_device_properties(self.index)
        self.capacity = _finite(props.total_memory, positive=True)
        visible = env.get("CUDA_VISIBLE_DEVICES")
        tokens = visible.split(",") if visible is not None else []
        token = tokens[self.index].strip() if self.index < len(tokens) else None
        uuid = str(getattr(props, "uuid", ""))
        if "MIG-" in uuid or (token is not None and token.startswith("MIG-")):
            raise _Unavailable("mig_utilization_unsupported")
        # Torch _CUuuid.__str__ returns the canonical UUID without NVIDIA's prefix.
        # Normalize only a complete UUID; never infer a device from an ordinal.
        if _GPU_UUID.fullmatch("GPU-" + uuid):
            uuid = "GPU-" + uuid
        if not _GPU_UUID.fullmatch(uuid):
            uuid = token or ""
        if not _GPU_UUID.fullmatch(uuid):
            raise _Unavailable("gpu_identity_unavailable")
        if token and token.startswith("GPU-") and not uuid.lower().startswith(token.lower()):
            raise _Unavailable("gpu_identity_conflict")
        self.uuid = uuid

    def sample(self):
        completed = self.run(
            ["nvidia-smi", f"--id={self.uuid}", "--query-gpu=uuid,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=1.0,
        )
        lines = completed.stdout.strip().splitlines()
        if len(lines) != 1:
            raise _Unavailable("ambiguous_gpu_response")
        fields = [field.strip() for field in lines[0].split(",")]
        if len(fields) != 2 or fields[0].lower() != self.uuid.lower():
            raise _Unavailable("gpu_identity_mismatch")
        utilization = _finite(float(fields[1])) / 100.
        if utilization > 1:
            raise _Unavailable("invalid_utilization")
        peak = _finite(self.cuda.max_memory_allocated(self.index))
        if peak > self.capacity:
            raise _Unavailable("invalid_allocator_peak")
        return dict(gpu_uuid=self.uuid, cuda_index=self.index,
                    capacity_bytes=self.capacity, peak_vram_bytes=peak,
                    utilization=utilization, monotonic_seconds=time.monotonic())


class ResourceSampler:
    """Single-use sampler. Control start/stop/summarize from one owner thread.

    ``_probe_factory`` is a hardware test seam; production uses _CudaProbe.
    No partial observation is returned after any sampling failure. No automatic
    backend installation, allocator reset, synchronization, or device mutation.
    """

    def __init__(self, device=None, *, interval_seconds=.25, max_samples=120,
                 min_samples=2, max_age_seconds=3., _probe_factory=None):
        for value in (interval_seconds, max_age_seconds):
            _finite(value, positive=True)
        if (type(max_samples) is not int or type(min_samples) is not int
                or not 1 <= min_samples <= max_samples <= 10000):
            raise ValueError("require 1 <= min_samples <= max_samples <= 10000")
        self.device = device
        self.interval = interval_seconds
        self.min_samples = min_samples
        self.max_age = max_age_seconds
        self._factory = _probe_factory or _CudaProbe
        self._rows = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._stopped = False
        self._error = None
        self._count = 0
        self._binding = None

    def _fail(self, error):
        # Do not echo subprocess output, exception text or environment variables.
        reason = str(error) if isinstance(error, _Unavailable) else type(error).__name__
        with self._lock:
            self._error = self._error or reason
        self._stop.set()

    def start(self):
        if self._started or self._stopped:
            raise RuntimeError("sampler is single-use")
        self._started = True
        try:
            probe = self._factory(self.device)
            self._thread = threading.Thread(target=self._collect, args=(probe,),
                                            name="draftfit-resource-monitor", daemon=True)
            self._thread.start()
        except Exception as exc:
            self._fail(exc)
        return self

    def _collect(self, probe):
        try:
            while True:
                row = probe.sample()
                with self._lock:
                    if self._error is not None:
                        break
                    self._rows.append(row)
                    self._count += 1
                if self._stop.is_set():
                    break
                # A stop during sleep wakes one final sample; a stop during an
                # in-flight sample retains that sample. Neither starts two calls.
                self._stop.wait(self.interval)
        except Exception as exc:
            self._fail(exc)

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=1.25)
            if self._thread.is_alive():
                self._fail(_Unavailable("worker_stop_timeout"))
        self._stopped = True
        return self

    def summarize(self, phase_id, preset, step, throughput, loss):
        """Return status, observation (or None), reason, and sample coverage."""
        with self._lock:
            rows, error, count = list(self._rows), self._error, self._count
        result = dict(status="measurement_unavailable", reason=None, observation=None,
                      samples=len(rows), samples_dropped=count - len(rows))

        def unavailable(reason):
            result["reason"] = reason
            return result

        if error:
            return unavailable(error)
        if not self._started or not self._stopped:
            return unavailable("sampler_not_stopped" if self._started else "sampler_not_started")
        try:
            if (not isinstance(phase_id, str) or not phase_id or not isinstance(preset, str)
                    or not preset or type(step) is not int or step <= 0):
                raise _Unavailable("invalid_milestone_binding")
            _finite(throughput)
            _finite(loss)
            binding = (phase_id, preset, step, throughput, loss)
            if self._binding is not None and binding != self._binding:
                raise _Unavailable("sampler_already_bound")
            if len(rows) < self.min_samples:
                raise _Unavailable("insufficient_samples")
            now = time.monotonic()
            if not 0 <= now - rows[-1]["monotonic_seconds"] <= self.max_age:
                raise _Unavailable("stale_samples")
            if len({(r["gpu_uuid"], r["cuda_index"], r["capacity_bytes"]) for r in rows}) != 1:
                raise _Unavailable("device_changed")
            capacity = _finite(rows[0]["capacity_bytes"], positive=True)
            peak = _finite(max(r["peak_vram_bytes"] for r in rows))
            utilization = _finite(sum(r["utilization"] / len(rows) for r in rows))
            if peak > capacity or utilization > 1:
                raise _Unavailable("invalid_resource_ratio")
        except (KeyError, TypeError, ValueError, _Unavailable) as exc:
            return unavailable(str(exc) if isinstance(exc, _Unavailable) else "invalid_sample")
        self._binding = binding
        result.update(status="available", observation=dict(
            phase_id=phase_id, preset=preset, step=step, throughput=throughput, loss=loss,
            peak_vram_bytes=peak, capacity_bytes=capacity, utilization=utilization,
            gpu_uuid=rows[0]["gpu_uuid"], cuda_index=rows[0]["cuda_index"],
            memory_scope="process_torch_allocator", capacity_scope="device",
            peak_window="since_last_external_allocator_reset", utilization_scope="device",
            utilization_aggregation="sample_mean", utilization_source="nvidia-smi",
            sample_start_monotonic_seconds=rows[0]["monotonic_seconds"],
            sample_end_monotonic_seconds=rows[-1]["monotonic_seconds"],
        ))
        return result

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
        return False
