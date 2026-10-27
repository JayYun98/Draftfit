from __future__ import annotations

import json
import fcntl
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class Store:
    """Small durable record store. No process handles or credentials are persisted."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner = (self.root / '.workspace.lock').open('a')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.owner.close()
            raise ValueError('Another workspace server is using this state directory.') from exc
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'workspace.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT PRIMARY KEY, body TEXT)')
        self.db.commit()
        for job in self.list('job'):
            if job['status'] in ('queued', 'running', 'cancelling'):
                job.update(status='interrupted', error='The workspace server stopped. Check engine state before resuming.')
                self.put('job', job)

    def create(self, kind: str, **body):
        body = {'id': uuid.uuid4().hex[:16], 'created_at': time.time(), **body}
        self.put(kind, body)
        return body

    def put(self, kind: str, body: dict):
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?)',
                            (kind, body['id'], json.dumps(body, ensure_ascii=False, allow_nan=False)))
            self.db.commit()
        return body

    def get(self, kind: str, ident: str):
        with self.lock:
            row = self.db.execute('SELECT body FROM records WHERE kind=? AND id=?', (kind, ident)).fetchone()
        if not row:
            raise ValueError('Record not found.')
        return json.loads(row[0])

    def list(self, kind: str):
        with self.lock:
            rows = self.db.execute('SELECT body FROM records WHERE kind=? ORDER BY rowid DESC', (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def path(self, relative: str):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('The path is outside the workspace.')
        return path

    def close(self):
        self.db.close()
        self.owner.close()
