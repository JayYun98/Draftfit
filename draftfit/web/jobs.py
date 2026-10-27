from __future__ import annotations

import json
import ast
import math
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from .store import Store

_ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def read_metric(line):
    clean = _ANSI.sub('', line)
    try:
        item = json.loads(clean)
        if isinstance(item, dict):
            loss = item.get('loss', item.get('train/loss'))
            step = item.get('step', item.get('global_step'))
            if loss is not None and step is not None:
                return {'step': int(step), 'loss': float(loss)}
    except (ValueError, TypeError):
        pass
    step = re.search(r'(?:step|Step|global_step)\s*[=: ]\s*(\d+)', clean)
    if step and '{' in clean:
        try:
            metrics = ast.literal_eval(clean[clean.index('{'):])
            value = metrics.get('loss', metrics.get('train/loss'))
            if value is not None and math.isfinite(float(value)):
                return {'step': int(step.group(1)), 'loss': float(value)}
        except (ValueError, SyntaxError, TypeError, AttributeError):
            pass
    loss = re.search(r'(?:loss|Loss)\s*[=: ]\s*([\d.eE+\-]+)', clean)
    if step and loss:
        try:
            value = float(loss.group(1))
            if math.isfinite(value):
                return {'step': int(step.group(1)), 'loss': value}
        except ValueError:
            pass
    return None


class Jobs:
    def __init__(self, store: Store, *, env=None):
        self.env = env or {}
        self.store = store
        self.processes = {}
        self.cancellations = set()
        self.lock = threading.RLock()
        self.threads = []
        self.stopping = set()
        self.closing = False

    def start(self, name, commands, cwd, *, metadata=None, timeout=3600):
        with self.lock:
            if self.closing:
                raise ValueError('The workspace is shutting down.')
            record = self.store.create('job', name=name, status='queued', commands=commands,
                                       metadata=metadata or {}, metrics=[], exit_code=None)
            directory = self.store.path('jobs/' + record['id'])
            directory.mkdir(parents=True)
            record['log_path'] = str(directory.relative_to(self.store.root) / 'output.log')
            self.store.put('job', record)
            worker = threading.Thread(target=self._run, args=(record, commands, Path(cwd), timeout), daemon=True)
            self.threads.append(worker)
            worker.start()
        return record

    def _run(self, record, commands, cwd, timeout):
        ident = record['id']
        deadline = time.monotonic() + timeout
        with self.lock:
            record.update(status='cancelling' if ident in self.cancellations else 'running',
                          started_at=time.time())
            self.store.put('job', record)
        try:
            with self.store.path(record['log_path']).open('a', buffering=1) as log:
                for index, argv in enumerate(commands):
                    if ident in self.cancellations:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Execution timed out.')
                    record['stage'] = index
                    self.store.put('job', record)
                    log.write(f'\n[stage {index + 1}/{len(commands)}]\n')
                    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding='utf-8', errors='replace',
                                            env={**os.environ, **self.env, 'PYTHONUNBUFFERED': '1'}, start_new_session=True)
                    with self.lock:
                        self.processes[ident] = proc
                    if ident in self.cancellations:
                        self._signal(proc)
                    timed_out = threading.Event()
                    def expire():
                        timed_out.set()
                        self._signal(proc)
                    timer = threading.Timer(remaining, expire)
                    timer.daemon = True
                    timer.start()
                    try:
                        for line in iter(proc.stdout.readline, ''):
                            clean = _ANSI.sub('', line)
                            log.write(clean)
                            metric = read_metric(clean)
                            if metric and math.isfinite(metric['loss']):
                                record['metrics'] = [*record['metrics'][-4999:], metric]
                                self.store.put('job', record)
                        code = proc.wait()
                        record['exit_code'] = code
                        if timed_out.is_set():
                            raise TimeoutError('Execution timed out. Check the engine checkpoint before resuming.')
                        if code != 0:
                            break
                    finally:
                        timer.cancel()
                        if proc.poll() is None:
                            self._signal(proc)
                            proc.wait()
                        proc.stdout.close()
                record['status'] = ('cancelled' if ident in self.cancellations else
                                    'succeeded' if record['exit_code'] == 0 else 'failed')
        except Exception as exc:
            record.update(status='failed', error=str(exc))
        finally:
            with self.lock:
                if ident in self.cancellations:
                    record['status'] = 'cancelled'
                record['finished_at'] = time.time()
                self.store.put('job', record)
                self.stopping.discard(self.processes.pop(ident, None))

    def _signal(self, proc):
        # The engine owns detached worker/service sessions and their per-phase
        # teardown deadlines. Signal its owner once; killing it first would
        # orphan those sessions and leave inherited stdout open indefinitely.
        with self.lock:
            if proc in self.stopping or proc.poll() is not None:
                return
            self.stopping.add(proc)
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def cancel(self, ident):
        with self.lock:
            record = self.store.get('job', ident)
            if record['status'] not in ('queued', 'running', 'cancelling'):
                raise ValueError('Only an active job can be stopped.')
            self.cancellations.add(ident)
            record['status'] = 'cancelling'
            self.store.put('job', record)
            proc = self.processes.get(ident)
            if proc:
                self._signal(proc)
        return record

    def close(self):
        """Stop only children launched by this UI; the engine owns recovery."""
        with self.lock:
            self.closing = True
            for job in self.store.list('job'):
                if job['status'] in ('queued', 'running', 'cancelling'):
                    self.cancel(job['id'])
            workers = tuple(self.threads)
        # Do not close SQLite while a reader is still waiting for engine cleanup.
        for worker in workers:
            worker.join()
