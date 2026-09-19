"""Local arguments and real process cleanup, never GPU validation evidence."""

import contextlib
import io
import os
import select
import signal
import subprocess
import sys
import threading
import unittest

from scripts.gates.run_owned_teacher_gpu import (
    arguments,
    stop_owned_group,
    training_inputs,
)


class OwnedTeacherGPUArgumentsTest(unittest.TestCase):
    def test_bounded_real_runtime_arguments(self):
        args = arguments(
            [
                "--backend",
                "vllm",
                "--revision",
                "a" * 40,
                "--output",
                "/tmp/new-gpu-evidence",
            ]
        )
        self.assertEqual(args.steps, 20)
        self.assertEqual(args.algorithms, ["dspark", "dflash2"])
        self.assertFalse(hasattr(args, "mock"))

    def test_mutable_revision_and_unbounded_steps_rejected(self):
        for revision, steps in (("main", "20"), ("a" * 40, "1"), ("a" * 40, "101")):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                arguments(
                    [
                        "--backend",
                        "transformers",
                        "--revision",
                        revision,
                        "--steps",
                        steps,
                        "--output",
                        "/tmp/new-gpu-evidence",
                    ]
                )

    def test_dspark_receives_final_teacher_states(self):
        import torch

        tensors = {
            name: torch.ones(1, 2, 2)
            for name in ("input_ids", "hidden_states", "loss_mask", "target")
        }
        inputs = training_inputs(tensors, "dspark", "cpu")
        self.assertIs(inputs["target_last_hidden_states"], tensors["target"])
        self.assertNotIn(
            "target_last_hidden_states", training_inputs(tensors, "dflash2", "cpu")
        )

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_term_reaps_actual_parent_and_child_group(self):
        # The leader reaps its child on TERM, matching a cooperative supervisor.
        program = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
def terminate(_signum, _frame):
    child.wait(timeout=2)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
print(child.pid, flush=True)
time.sleep(30)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", program],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        errors = []
        worker = None
        try:
            self.assertTrue(
                select.select([process.stdout], [], [], 3)[0], "child startup timed out"
            )
            child_pid = int(process.stdout.readline())
            self.assertEqual(os.getpgid(process.pid), process.pid)
            self.assertEqual(os.getpgid(child_pid), process.pid)

            def stop():
                try:
                    stop_owned_group(process)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=stop, daemon=True)
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "TERM cleanup exceeded five seconds")
            self.assertFalse(errors, errors)
            self.assertEqual(process.returncode, 0)
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            # Exact session ID from our own Popen, never an ambient process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
            process.stdout.close()
            if worker is not None:
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
