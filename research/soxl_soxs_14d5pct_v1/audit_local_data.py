"""Inspect existing local five-minute coverage without fetching or mixing prices.

Optional audit dependency: pandas + pyarrow (already in the project's Python).
This audit does not feed local data into the frozen IBKR model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from paths import IMPORT_ROOT, PROJECT_ROOT


def audit():
    calendar = json.loads((IMPORT_ROOT / 'raw' / 'calendar.json').read_text())
    report = {'as_of': '2026-10-12', 'data_source': 'existing local Futu files',
              'scope': 'timestamp/metadata coverage only; no OHLC, split or fill validation',
              'raw_data_changed': False, 'network_requests': 0, 'symbols': []}
    for symbol in ['SOXL', 'SOXS']:
        path = PROJECT_ROOT / 'market_data' / 'us_5m' / symbol / '2026.parquet'
        if not path.exists():
            report['symbols'].append({'symbol': symbol, 'status': 'missing'})
            continue
        frame = pd.read_parquet(path)
        starts = pd.to_datetime(frame.start_at, utc=True)
        ends = pd.to_datetime(frame.end_at, utc=True)
        valid = (starts.notna() & ends.notna() & frame.price_basis.eq('NONE')
                 & frame.session_type.eq('regular')
                 & (ends - starts).eq(pd.Timedelta(minutes=5)))
        usable = pd.DatetimeIndex(ends[valid])
        months = []
        for month in ['2026-05', '2026-07', '2026-09']:
            days = []
            for session in calendar['sessions']:
                if not session['session_date'].startswith(month):
                    continue
                expected = pd.date_range(session['open_at'], session['close_at'],
                                         freq='5min', inclusive='right')
                missing = expected[~expected.isin(usable)]
                days.append({'date': session['session_date'], 'expected_bars': len(expected),
                             'present_bars': len(expected) - len(missing),
                             'missing_bars': len(missing)})
            months.append({'month': month, 'expected_sessions': len(days),
                           'full_sessions': sum(d['missing_bars'] == 0 for d in days),
                           'missing_bars': sum(d['missing_bars'] for d in days),
                           'incomplete_sessions': [d for d in days if d['missing_bars']]})
        report['symbols'].append({
            'symbol': symbol, 'path': str(path.relative_to(PROJECT_ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(frame),
            'first_time_key': str(frame.time_key.min()), 'last_time_key': str(frame.time_key.max()),
            'last_valid_regular_end_et': str(usable.max().tz_convert('America/New_York')),
            'missing_time_or_basis_or_session_metadata_rows': int((
                starts.isna() | ends.isna() | frame.price_basis.isna()
                | frame.session_type.isna()).sum()),
            'duplicate_usable_end_timestamps': int(usable.duplicated().sum()), 'months': months,
        })
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
