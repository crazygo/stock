"""Budgeted current snapshot, separate from immutable evaluation inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

import core


def refresh(base, budget=30):
    base = Path(base)
    stamp = datetime.now(timezone.utc)
    folder = core.ROOT / '.cache/soxl_soxs_14d5pct_v2/live' / stamp.strftime('%Y%m%dT%H%M%S%fZ')
    (folder / 'bars').mkdir(parents=True)
    (folder / 'calendar.json').write_bytes((base / 'calendar.json').read_bytes())
    for symbol in core.SYMBOLS:
        frame = pd.read_parquet(base / 'bars' / f'{symbol}.parquet')
        if 'received_at' not in frame:
            frame['received_at'] = stamp.isoformat()
        else:
            frame['received_at'] = frame.received_at.fillna(stamp.isoformat())
        frame.to_parquet(folder / 'bars' / f'{symbol}.parquet', index=False, compression='zstd', compression_level=7)
    today = stamp.astimezone(__import__('zoneinfo').ZoneInfo(core.ET)).date().isoformat()
    log = folder / 'refresh.log'
    command = [sys.executable, str(Path(__file__).with_name('acquire.py')), '--output', str(folder), '--end', today]
    result = {'requested_at': stamp.isoformat(), 'requested_end_et': today, 'budget_seconds': budget,
              'base_snapshot': str(base), 'network_data_automatic_upload': False, 'shared_files_modified': False}
    with log.open('w') as output:
        try:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=budget)
            result['worker_exit_code'] = completed.returncode
            result['status'] = 'snapshot_ready' if completed.returncode == 0 else 'partial_snapshot'
        except subprocess.TimeoutExpired:
            result['status'] = 'budget_timeout_partial_snapshot_preserved'
    core.dump(folder / 'refresh_status.json', result)
    return folder, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--budget', type=float, default=30)
    args = parser.parse_args()
    folder, result = refresh(args.base, args.budget)
    print(json.dumps({'folder': str(folder), **result}, indent=2))
