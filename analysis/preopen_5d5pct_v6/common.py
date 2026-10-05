"""Isolated experiment utilities; no writes to v5 or broker trade context."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'analysis/preopen_ranked_policy_v5'
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar
SESSIONS = {r['session_date']: r for r in calendar('2024-08-01', '2026-12-31')['sessions']}
DATES = sorted(SESSIONS)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, Path): return str(value)
    return value

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def save(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing.parquet')
    frame.to_parquet(tmp, index=False, compression='zstd', compression_level=7)
    tmp.replace(path)

def journal(kind, payload):
    path = OUT / 'cache/audit.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(clean(dict(at=now(), kind=kind, **payload)), ensure_ascii=False) + '\n')
