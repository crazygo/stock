"""Shared, isolated helpers. No writes to legacy experiments or R2."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / '.cache/stock_review_pair_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一项双页面更新正在运行') from exc
        yield


def worker(arguments, timeout, host='127.0.0.1', port=11111, capture=False):
    script = PKG / 'members_worker.py' if capture else ROOT / 'analysis/stock_traits_daily_v1/opend_worker.py'
    command = [sys.executable, str(script), *arguments, '--host', host, '--port', str(port)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=max(.1, timeout))
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error_type': 'timeout'}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith('@@STDV1_RESULT@@ '):
            return json.loads(line[len('@@STDV1_RESULT@@ '):])
    return {'ok': False, 'error_type': 'worker_failed', 'exit_code': result.returncode}
