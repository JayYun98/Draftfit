"""Real local subprocess lifecycle tests; no Hugging Face or GPU operations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from draftfit.export.hf_publish import prepare_publication
from draftfit.export.publication_worker import PublicationWorker


SCRIPT = r'''
import json, os, pathlib, signal, subprocess, sys, time
mode = sys.argv[1]
job = pathlib.Path(sys.argv[sys.argv.index('--job-dir') + 1])
if mode == 'success':
    (job / 'completed').touch()
    sys.exit(0)
if mode == 'fail':
    sys.exit(7)
if mode == 'env':
    secret = os.environ['HF_TOKEN']
    (job / 'env-ok').write_text(str(bool(secret)))
    (job / 'argv.json').write_text(json.dumps(sys.argv))
    print(secret, flush=True)
    print(secret, file=sys.stderr, flush=True)
    sys.exit(0)
if mode == 'retry':
    if (job / 'attempted').exists():
        sys.exit(0)
    (job / 'attempted').touch()
if mode in ('forkhang', 'forkexit'):
    subprocess.Popen([sys.executable, __file__, 'grandchild', '--job-dir', str(job)])
    while not (job / 'grandchild.pid').exists():
        time.sleep(.005)
    if mode == 'forkexit':
        sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode == 'grandchild':
    (job / 'grandchild.pid').write_text(str(os.getpid()))
else:
    (job / 'ready').touch()
while True:
    if mode == 'grandchild':
        (job / 'heartbeat').write_text(str(time.monotonic()))
    time.sleep(.01)
'''


def wait_file(path, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(.005)
    raise AssertionError("child readiness file absent")


class PublicationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / 'export'
        source.mkdir()
        (source / 'config.json').write_text('{}')
        (source / 'model.safetensors').write_bytes(b'fixture-weights')
        self.job = prepare_publication(source, 'test/fixture', self.root / 'job', 'local-only')
        self.script = self.root / 'child.py'
        self.script.write_text(SCRIPT)

    def worker(self, mode='success', **kwargs):
        worker = PublicationWorker(command=[sys.executable, str(self.script), mode],
                                   terminate_timeout=.03, kill_timeout=.5, **kwargs)
        self.addCleanup(worker.close)
        return worker

    def test_success_real_child_and_durable_status(self):
        worker = self.worker()
        state = worker.start(self.job)
        self.assertEqual(state['status'], 'running')
        result = worker.wait(time.time() + 3)
        self.assertEqual((result['status'], result['returncode'], result['retryable']), ('succeeded', 0, False))
        self.assertTrue((self.job / 'completed').exists())
        self.assertEqual(json.loads((self.job / 'publication-worker.json').read_text()), result)
        self.assertEqual(worker.poll(), result)
        self.assertEqual(worker.close(), result)

    def test_nonzero_exit_is_retryable_without_raising_raw_error(self):
        worker = self.worker('fail')
        worker.start(self.job)
        result = worker.wait(time.time() + 3)
        self.assertEqual((result['status'], result['returncode'], result['retryable']), ('failed', 7, True))
        self.assertTrue((self.job / 'payload' / 'model.safetensors').exists())

    def test_deadline_kills_ignoring_term_then_same_job_retries(self):
        worker = self.worker('retry')
        publication = (self.job / 'publication.json').read_bytes()
        receipt = self.job / 'receipt.json'
        receipt.write_text('{"status":"pending","commit":"known-commit"}')
        saved_receipt = receipt.read_bytes()
        worker.start(self.job)
        wait_file(self.job / 'ready')
        started = time.monotonic()
        result = worker.wait(time.time() + .05)
        self.assertLess(time.monotonic() - started, 1.)
        self.assertEqual(result['status'], 'timed_out')
        self.assertTrue(result['retryable'])
        self.assertLess(result['returncode'], 0)
        self.assertEqual(receipt.read_bytes(), saved_receipt)
        self.assertEqual((self.job / 'publication.json').read_bytes(), publication)
        worker.start(self.job)
        self.assertEqual(worker.wait(time.time() + 3)['status'], 'succeeded')
        self.assertEqual(receipt.read_bytes(), saved_receipt)

    def test_at_most_one_child_and_same_job_flock(self):
        owner, other = self.worker('hang'), self.worker()
        owner.start(self.job)
        wait_file(self.job / 'ready')
        with self.assertRaisesRegex(RuntimeError, 'already active'):
            owner.start(self.job)
        with self.assertRaisesRegex(RuntimeError, 'already owned'):
            other.start(self.job)
        owner.close()
        other.start(self.job)
        self.assertEqual(other.wait(time.time() + 3)['status'], 'succeeded')

    def test_poll_does_not_wait_for_network_and_close_is_bounded(self):
        worker = self.worker('hang')
        worker.start(self.job)
        wait_file(self.job / 'ready')
        started = time.monotonic()
        self.assertEqual(worker.poll()['status'], 'running')
        self.assertLess(time.monotonic() - started, .2)
        result = worker.close()
        self.assertLess(time.monotonic() - started, 1.)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(worker.close(), result)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            worker.start(self.job)

    def test_process_group_descendants_stopped_on_timeout_and_parent_success(self):
        for mode in ('forkhang', 'forkexit'):
            with self.subTest(mode=mode):
                for name in ('grandchild.pid', 'heartbeat', 'ready'):
                    (self.job / name).unlink(missing_ok=True)
                worker = self.worker(mode)
                worker.start(self.job)
                wait_file(self.job / 'grandchild.pid')
                wait_file(self.job / 'heartbeat')
                result = worker.wait(time.time() + (.1 if mode == 'forkhang' else 3))
                self.assertEqual(result['status'], 'timed_out' if mode == 'forkhang' else 'succeeded')
                # A reparented zombie may still exist in the process table. It
                # cannot execute; a heartbeat detects a surviving live child.
                time.sleep(.05)
                heartbeat = (self.job / 'heartbeat').read_bytes()
                time.sleep(.08)
                self.assertEqual((self.job / 'heartbeat').read_bytes(), heartbeat)

    def test_credentials_inherited_never_written_in_journal_or_argv(self):
        sentinel = 'test-only-credential-do-not-log'
        worker = self.worker('env')
        with patch.dict(os.environ, {'HF_TOKEN': sentinel}):
            worker.start(self.job)
            result = worker.wait(time.time() + 3)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual((self.job / 'env-ok').read_text(), 'True')
        for path in self.job.rglob('*'):
            if path.is_file():
                self.assertNotIn(sentinel.encode(), path.read_bytes())
        self.assertNotIn(sentinel, repr(result))

    def test_default_module_command_executes_local_job_validation_without_hf(self):
        # Invalid immutable snapshot fails before credential/network setup.
        (self.job / 'payload' / 'config.json').write_text('{"tampered":true}')
        worker = PublicationWorker()
        self.addCleanup(worker.close)
        worker.start(self.job)
        self.assertEqual(worker._process.args, [sys.executable, '-m', 'draftfit.export.hf_publish',
                                                '--job-dir', str(self.job.resolve())])
        self.assertEqual(worker.wait(time.time() + 3)['status'], 'failed')

    def test_missing_command_and_unprepared_job(self):
        worker = PublicationWorker(command=[str(self.root / 'nonexistent')])
        self.addCleanup(worker.close)
        with self.assertRaisesRegex(ValueError, 'prepared durable'):
            worker.start(self.root)
        result = worker.start(self.job)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_type'], 'FileNotFoundError')
        self.assertNotIn(str(self.root), repr(result))
        retry = self.worker()
        retry.start(self.job)
        self.assertEqual(retry.wait(time.time() + 3)['status'], 'succeeded')

    def test_expired_and_invalid_deadlines(self):
        worker = self.worker('hang')
        worker.start(self.job)
        wait_file(self.job / 'ready')
        for deadline in (float('nan'), float('inf'), True, -1):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                worker.wait(deadline)
        self.assertEqual(worker.wait(0)['status'], 'timed_out')

    def test_context_exception_still_terminates_child(self):
        worker = self.worker('hang')
        with self.assertRaisesRegex(RuntimeError, 'controller failure'):
            with worker:
                worker.start(self.job)
                wait_file(self.job / 'ready')
                raise RuntimeError('controller failure')
        self.assertEqual(worker.poll()['status'], 'cancelled')
        self.assertIsNone(worker._process)

    def test_interpreter_exit_cleanup_has_no_non_daemon_network_thread(self):
        controller = self.root / 'controller.py'
        controller.write_text('''
import pathlib, sys, time
from draftfit.export.publication_worker import PublicationWorker
job = pathlib.Path(sys.argv[1])
worker = PublicationWorker(command=[sys.executable, sys.argv[2], 'hang'], terminate_timeout=.03)
worker.start(job)
while not (job / 'ready').exists():
    time.sleep(.005)
# Deliberately omit close: bounded atexit handler owns cleanup.
''')
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        result = subprocess.run([sys.executable, str(controller), str(self.job), str(self.script)],
                                env=env, capture_output=True, timeout=4)
        self.assertEqual(result.returncode, 0)
        receipt = json.loads((self.job / 'publication-worker.json').read_text())
        self.assertEqual(receipt['status'], 'cancelled')
        retry = self.worker()
        retry.start(self.job)
        self.assertEqual(retry.wait(time.time() + 3)['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
