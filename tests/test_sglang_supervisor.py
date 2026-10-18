"""Real POSIX CPU process lifetime regressions; no engine or GPU is loaded."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import psutil

from draftfit.training import sglang_producer as producer


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def run_producer(root, behavior="sleep", seconds=12):
    from types import SimpleNamespace

    # Exercise the production supervisor with a harmless stage entry point.
    producer.__spec__ = SimpleNamespace(name=__name__)
    root = Path(root)
    (root / "producer.pid").write_text(str(os.getpid()))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, producer._terminate_signal)
    producer._supervise(
        ["--_stage", behavior, str(root)], time.time() + seconds, root / "stage.log"
    )


def run_dispatch(root):
    from draftfit.training.serving_dispatch import _run

    root = Path(root)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _run(
                [
                    sys.executable,
                    "-c",
                    "from tests.test_sglang_supervisor import run_producer;"
                    f"run_producer({str(root)!r})",
                ],
                root,
                time.time() + 12,
                lock.fileno(),
            )
        finally:
            # This is the same boundary at which memory_handoff restores training.
            pids = json.loads((root / "pids.json").read_text())
            (root / "restored.json").write_text(
                json.dumps([pid for pid in pids if alive(pid)])
            )


@unittest.skipUnless(os.name == "posix", "POSIX process groups and flock required")
class SGLangSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix="sglang-supervisor-",
            dir="/private/tmp" if sys.platform == "darwin" else None,
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1]),
            PYTHONDONTWRITEBYTECODE="1",
        )
        self.processes = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        pids = []
        for name in ("pids.json",):
            if (self.root / name).exists():
                pids.extend(json.loads((self.root / name).read_text()))
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_for(self, predicate, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("bounded subprocess condition was not reached")

    def launch(self, dispatch=False, behavior="sleep", seconds=12):
        lock = (self.root / ".lock").open("a")
        if not dispatch:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = (
            "from tests.test_sglang_supervisor import run_dispatch;"
            f"run_dispatch({str(self.root)!r})"
            if dispatch
            else "from tests.test_sglang_supervisor import run_producer;"
            f"run_producer({str(self.root)!r}, {behavior!r}, {seconds!r})"
        )
        env = dict(self.env)
        if not dispatch:
            env["DRAFTFIT_SERVING_LOCK_FD"] = str(lock.fileno())
        with (self.root / "outer.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                pass_fds=() if dispatch else (lock.fileno(),),
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        lock.close()
        self.processes.append(process)
        self.wait_for(lambda: (self.root / "pids.json").exists())
        return process, json.loads((self.root / "pids.json").read_text())

    def lock_available(self):
        with (self.root / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    def test_sigkill_producer_keeps_lock_until_stage_and_grandchild_exit(self):
        process, pids = self.launch()
        self.assertTrue(all(alive(pid) for pid in pids))
        process.kill()
        process.wait(timeout=3)
        end = time.monotonic() + 5
        while any(alive(pid) for pid in pids) and time.monotonic() < end:
            if self.lock_available():
                self.assertFalse(
                    any(alive(pid) for pid in pids),
                    "job lock reacquired while serving descendants are alive",
                )
            time.sleep(0.01)
        self.assertFalse(
            any(alive(pid) for pid in pids),
            "orphan serving processes survived producer SIGKILL",
        )
        self.wait_for(self.lock_available)

    def test_dispatch_waits_for_descendants_before_restore_after_producer_sigkill(self):
        process, pids = self.launch(dispatch=True)
        os.kill(int((self.root / "producer.pid").read_text()), signal.SIGKILL)
        process.wait(timeout=6)
        self.assertEqual(
            json.loads((self.root / "restored.json").read_text()),
            [],
            "training restored while serving descendants are alive",
        )
        self.assertFalse(any(alive(pid) for pid in pids))

    def test_dispatcher_sigkill_does_not_release_job_lock_while_serving(self):
        process, pids = self.launch(dispatch=True)
        process.kill()
        process.wait(timeout=3)
        self.assertFalse(self.lock_available())
        os.kill(int((self.root / "producer.pid").read_text()), signal.SIGKILL)
        self.wait_for(lambda: not any(alive(pid) for pid in pids))
        self.wait_for(self.lock_available)

    def test_restore_waits_for_stopped_guardian_and_lock_remains_owned(self):
        process, pids = self.launch(dispatch=True)
        guardian = int((self.root / "guardian.pid").read_text())
        os.kill(guardian, signal.SIGSTOP)
        try:
            os.kill(int((self.root / "producer.pid").read_text()), signal.SIGKILL)
            time.sleep(0.15)
            self.assertTrue(all(alive(pid) for pid in pids))
            self.assertFalse(self.lock_available())
            self.assertIsNone(process.poll())
            self.assertFalse((self.root / "restored.json").exists())
        finally:
            os.kill(guardian, signal.SIGCONT)
        process.wait(timeout=6)
        self.assertEqual(json.loads((self.root / "restored.json").read_text()), [])
        self.assertTrue(self.lock_available())

    def test_killed_guardian_fail_stops_worker_without_restoring_training(self):
        process, pids = self.launch(dispatch=True)
        os.kill(int((self.root / "guardian.pid").read_text()), signal.SIGKILL)
        process.wait(timeout=6)
        self.assertEqual(process.returncode, 70, (self.root / "outer.log").read_text())
        self.assertFalse((self.root / "restored.json").exists())
        self.assertTrue(all(alive(pid) for pid in pids))
        self.assertFalse(self.lock_available())

    def test_producer_sigterm_cleans_stage_and_grandchild(self):
        process, pids = self.launch()
        process.terminate()
        process.wait(timeout=6)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM)
        self.assertFalse(any(alive(pid) for pid in pids))
        self.assertTrue(self.lock_available())

    def test_stage_deadline_cleans_stage_and_grandchild(self):
        start = time.monotonic()
        process, pids = self.launch(seconds=3.5)
        process.wait(timeout=6)
        self.assertNotEqual(process.returncode, 0)
        self.assertLess(time.monotonic() - start, 4)
        self.assertFalse(any(alive(pid) for pid in pids))
        self.assertTrue(self.lock_available())

    def test_successful_stage_exit_still_cleans_ignoring_grandchild(self):
        process, pids = self.launch(behavior="exit")
        process.wait(timeout=6)
        self.assertEqual(process.returncode, 0, (self.root / "outer.log").read_text())
        self.assertFalse(any(alive(pid) for pid in pids))
        self.assertTrue(self.lock_available())


if __name__ == "__main__":
    if "--_stage" in sys.argv:
        root = Path(sys.argv[-1])
        leaf = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)",
            ]
        )
        (root / "guardian.pid").write_text(str(os.getppid()))
        (root / "pids.json").write_text(json.dumps([os.getpid(), leaf.pid]))
        if sys.argv[2] == "sleep":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
    else:
        unittest.main()
