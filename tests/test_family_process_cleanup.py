"""Real process-group regressions for workers that leave live descendants."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from draftfit.config import Config
from draftfit.family import FamilyStage, _run_concurrent


GRANDCHILD = r'''
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)
'''

WORKER = r'''
import json, subprocess, sys, time
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
pid_file = Path(config['output_dir']) / 'grandchild.pid'
subprocess.Popen([sys.executable, '-u', '-c', sys.argv[2], str(pid_file)])
deadline = time.monotonic() + 2
while not pid_file.exists():
    if time.monotonic() > deadline: sys.exit(8)
    time.sleep(.01)
if config['run_id'] == 'fail': sys.exit(7)
time.sleep(30)  # Default SIGTERM action; the grandchild ignores it.
'''

CONTROLLER = r'''
import signal, subprocess, sys
from pathlib import Path
from unittest.mock import patch
from draftfit.config import Config
from draftfit.family import FamilyStage, _run_concurrent
from draftfit.cli import _worker_signal_unwind, _WorkerTermination
class Draft:
    def to_json_file(self, path): Path(path).write_text('{}')
root = Path(sys.argv[1])
config = Config.model_validate({
    'model': {'target_model_path': 'target'},
    'data': {'hidden_states_path': '/train'},
    'training': {'strategy': 'dflash'},
    'run_id': 'cancel', 'output_dir': str(root / 'cancel'),
})
real_popen = subprocess.Popen
def start(command, **kwargs):
    return real_popen([sys.executable, '-u', '-c', sys.argv[2], command[-1], sys.argv[3]], **kwargs)
with patch('draftfit.family.subprocess.Popen', side_effect=start):
    try:
        with _worker_signal_unwind():
            _run_concurrent(root, [FamilyStage('cancel', config, Draft(), None)], 1)
    except _WorkerTermination as exc:
        sys.exit(128 + exc.signum)
'''


class Draft:
    def to_json_file(self, path):
        Path(path).write_text('{}')


def _stage(root, name):
    config = Config.model_validate({
        'model': {'target_model_path': 'target'},
        'data': {'hidden_states_path': '/train'},
        'training': {'strategy': 'dflash'},
        'run_id': name, 'output_dir': str(root / name),
    })
    return FamilyStage(name, config, Draft(), None)


def _ps_state(pid):
    result = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                            capture_output=True, text=True, timeout=1)
    return result.stdout.strip()


def _assert_gone_or_zombie(test, pid):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        state = _ps_state(pid)
        if not state or state.startswith('Z'):
            return
        time.sleep(.02)
    test.fail(f'grandchild {pid} remained live (ps stat={_ps_state(pid)!r})')


def _cleanup(root):
    results = root / 'results.json'
    if results.exists():
        for row in json.loads(results.read_text()):
            if row.get('pid'):
                try:
                    os.killpg(row['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
    for pid_file in root.glob('*/grandchild.pid'):
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


@unittest.skipUnless(os.name == 'posix', 'requires POSIX process groups')
class FamilyProcessCleanupTest(unittest.TestCase):
    def test_reaped_failed_parent_kills_live_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_popen = subprocess.Popen

            def start(command, **kwargs):
                return real_popen([sys.executable, '-u', '-c', WORKER,
                                   command[-1], GRANDCHILD], **kwargs)

            try:
                with mock.patch('draftfit.family.subprocess.Popen', side_effect=start):
                    result = _run_concurrent(root, [_stage(root, 'fail')], 1)
                row = result['runs'][0]
                self.assertEqual((row['status'], row['exit_code']), ('failed', 7))
                self.assertTrue((root / 'fail' / 'grandchild.pid').exists())
                _assert_gone_or_zombie(self, int((root / 'fail' / 'grandchild.pid').read_text()))
            finally:
                _cleanup(root)

    def test_sigterm_cancellation_kills_ignoring_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = subprocess.Popen([sys.executable, '-u', '-c', CONTROLLER,
                                           str(root), WORKER, GRANDCHILD])
            try:
                deadline = time.monotonic() + 3
                pid_file = root / 'cancel' / 'grandchild.pid'
                while not pid_file.exists():
                    self.assertIsNone(controller.poll(), 'controller exited before child was ready')
                    if time.monotonic() > deadline:
                        self.fail('grandchild did not become ready')
                    time.sleep(.01)
                controller.send_signal(signal.SIGTERM)
                self.assertEqual(controller.wait(timeout=6), 143)
                row = json.loads((root / 'results.json').read_text())[0]
                self.assertEqual(row['status'], 'cancelled')
                _assert_gone_or_zombie(self, int(pid_file.read_text()))
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=1)
                _cleanup(root)


if __name__ == '__main__':
    unittest.main()
