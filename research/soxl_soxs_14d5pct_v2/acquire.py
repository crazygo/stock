"""Versioned Local -> R2 -> OpenD acquisition; never modifies shared archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar
from scripts.r2_client import R2Client

DEFAULT_DATA = ROOT / '.cache/soxl_soxs_14d5pct_v2/data'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(frame, symbol, source, received_at=None):
    """The documented Futu time_key is the ET bar end; retain receipt separately."""
    if frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    ends = pd.to_datetime(f.time_key).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    starts = ends - pd.Timedelta(minutes=5)
    f['symbol'] = symbol
    f['start_at'] = starts
    f['end_at'] = ends
    f['available_at'] = ends + pd.Timedelta(seconds=1)
    f['availability_basis'] = 'historical_assumption_bar_end_plus_1s'
    f['received_at'] = received_at
    f['source'] = source
    f['price_basis'] = 'NONE'
    local = ends.dt.tz_convert('America/New_York')
    minutes = local.dt.hour * 60 + local.dt.minute
    f['session_type'] = np.select(
        [(minutes > 570) & (minutes <= 960), (minutes > 240) & (minutes <= 570),
         (minutes > 960) & (minutes <= 1200)],
        ['regular', 'pre_market', 'post_market'], default='overnight')
    # Sunday's true evening overnight bars belong to the following Monday.
    trading = local.dt.tz_localize(None).dt.normalize()
    trading = trading + pd.to_timedelta((minutes > 1200).astype(int), unit='D')
    f['session_date'] = trading.dt.strftime('%Y-%m-%d')
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        f[col] = pd.to_numeric(f[col], errors='raise')
    bad = ((f[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
           | (f.high < f[['open', 'close', 'low']].max(axis=1) - 1e-7)
           | (f.low > f[['open', 'close', 'high']].min(axis=1) + 1e-7)
           | (f.volume < 0) | f[required].isna().any(axis=1))
    if bad.any():
        raise ValueError(f'{symbol}: {int(bad.sum())} invalid raw minute bars')
    columns = ['symbol', 'time_key', 'start_at', 'end_at', 'available_at',
               'availability_basis', 'received_at', 'source', 'price_basis',
               'session_type', 'session_date'] + required
    if 'turnover' in f:
        columns.append('turnover')
    return f[columns].drop_duplicates('end_at', keep='last').sort_values('end_at').reset_index(drop=True)


def missing_sessions(frame, sessions):
    ends = pd.DatetimeIndex(frame.end_at) if len(frame) else pd.DatetimeIndex([], tz='UTC')
    missing = []
    for session in sessions:
        expected = pd.date_range(session['open_at'], session['close_at'], freq='5min', inclusive='right')
        count = int(expected.isin(ends).sum())
        if count != len(expected):
            missing.append({'date': session['session_date'], 'missing_regular_bars': len(expected) - count})
    return missing


def ranges(days):
    if not days:
        return []
    out = []; start = previous = pd.Timestamp(days[0]['date'])
    for item in days[1:]:
        day = pd.Timestamp(item['date'])
        if (day - previous).days > 7 or day.year != start.year:
            out.append((str(start.date()), str(previous.date()))); start = day
        previous = day
    out.append((str(start.date()), str(previous.date())))
    return out


def acquire(output, symbols, start, end, host, port, offline=False):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    cal = calendar(start, '2026-12-31')
    (output / 'calendar.json').write_text(json.dumps(cal, indent=2))
    requested = [s for s in cal['sessions'] if s['session_date'] <= end]
    log = []; report = []; quote = None

    def record(item):
        log.append(item)
        with (output / 'acquisition.jsonl').open('a') as file:
            file.write(json.dumps(item, default=str) + '\n')

    try:
        for symbol in symbols:
            target = output / 'bars' / f'{symbol}.parquet'
            parts = []
            if target.exists():
                parts.append(pd.read_parquet(target))
                record({'symbol': symbol, 'source': 'local_snapshot', 'path': str(target), 'sha256': sha(target)})
            else:
                for year in range(int(start[:4]), int(end[:4]) + 1):
                    shared = ROOT / 'market_data/us_5m' / symbol / f'{year}.parquet'
                    if shared.exists():
                        original = pd.read_parquet(shared)
                        valid = original.price_basis.eq('NONE') if 'price_basis' in original else pd.Series(False, index=original.index)
                        captured = output / 'captured/local' / symbol / shared.name
                        captured.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(shared, captured)
                        parts.append(normalize(original[valid], symbol, 'local_Futu_archive'))
                        record({'symbol': symbol, 'source': 'local', 'path': str(shared.relative_to(ROOT)),
                                'sha256': sha(shared), 'excluded_unknown_basis_rows': int((~valid).sum())})
            frame = pd.concat(parts, ignore_index=True).drop_duplicates('end_at', keep='last').sort_values('end_at') if parts else pd.DataFrame()
            gaps = missing_sessions(frame, requested)
            if gaps and not offline:
                for year in sorted({int(x['date'][:4]) for x in gaps}):
                    key = f'us_5m/{symbol}/{year}.parquet'
                    remote = output / 'captured/r2' / symbol / f'{year}.parquet'
                    try:
                        client = R2Client(timeout=5, deadline=time.monotonic() + 12)
                        metadata = client.head_object(key)
                        if metadata:
                            client.get_object(key, remote)
                            raw = pd.read_parquet(remote)
                            valid = raw.price_basis.eq('NONE') if 'price_basis' in raw else pd.Series(False, index=raw.index)
                            fresh = normalize(raw[valid], symbol, 'R2_Futu_archive')
                            frame = pd.concat([frame, fresh], ignore_index=True).drop_duplicates('end_at', keep='first').sort_values('end_at')
                        record({'symbol': symbol, 'source': 'r2', 'key': key, 'found': metadata is not None,
                                'sha256': sha(remote) if metadata else None})
                    except Exception as exc:
                        record({'symbol': symbol, 'source': 'r2', 'key': key, 'error_type': type(exc).__name__, 'error': str(exc)[:250]})
                gaps = missing_sessions(frame, requested)
            if gaps and not offline:
                import futu as ft
                if quote is None:
                    ft.SysConfig.enable_proto_encrypt(False)
                    quote = ft.OpenQuoteContext(host=host, port=port)
                for begin, finish in ranges(gaps):
                    page_key = None; page = 0; seen_keys = set()
                    time.sleep(1.2)
                    while True:
                        requested_at = datetime.now(timezone.utc).isoformat()
                        ret, raw, next_key = quote.request_history_kline(
                            code='US.' + symbol, start=begin, end=finish,
                            ktype=ft.KLType.K_5M, autype=ft.AuType.NONE,
                            max_count=1000, page_req_key=page_key,
                            extended_time=True, session=ft.Session.ALL)
                        received = datetime.now(timezone.utc).isoformat()
                        if ret != ft.RET_OK:
                            record({'symbol': symbol, 'source': 'OpenD', 'start': begin, 'end': finish,
                                    'page': page, 'error': str(raw)[:400]})
                            break
                        part = output / 'captured/OpenD' / symbol / f'{begin}_{finish}_{page:04d}.parquet'
                        part.parent.mkdir(parents=True, exist_ok=True)
                        raw.to_parquet(part, index=False, compression='zstd', compression_level=7)
                        norm = normalize(raw, symbol, 'OpenD_history_NONE_ALL', received)
                        frame = pd.concat([frame, norm], ignore_index=True).drop_duplicates('end_at', keep='last').sort_values('end_at')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_parquet(target, index=False, compression='zstd', compression_level=7)
                        record({'symbol': symbol, 'source': 'OpenD', 'start': begin, 'end': finish,
                                'page': page, 'rows': len(raw), 'requested_at': requested_at,
                                'received_at': received, 'receipt_basis': 'observed_after_provider_response', 'sha256': sha(part),
                                'first': str(raw.time_key.min()) if len(raw) else None,
                                'last': str(raw.time_key.max()) if len(raw) else None})
                        if page % 10 == 0:
                            print(f'{symbol} {begin}..{finish} page={page} total_rows={len(frame)}', flush=True)
                        if next_key is None:
                            break
                        key_id = repr(next_key)
                        if key_id in seen_keys:
                            raise ValueError(f'{symbol}: repeating pagination key')
                        seen_keys.add(key_id); page_key = next_key; page += 1
                        time.sleep(0.6)
            if len(frame):
                target.parent.mkdir(parents=True, exist_ok=True)
                frame = frame[(frame.session_date >= start) & (frame.session_date <= end)]
                frame.to_parquet(target, index=False, compression='zstd', compression_level=7)
            missing = missing_sessions(frame, requested)
            item = {'symbol': symbol, 'requested_start': start, 'requested_end': end,
                    'rows': len(frame), 'first_bar_end': str(frame.end_at.min()) if len(frame) else None,
                    'last_bar_end': str(frame.end_at.max()) if len(frame) else None,
                    'regular_missing_sessions': missing, 'regular_complete': not missing,
                    'path': str(target.relative_to(output)), 'sha256': sha(target) if target.exists() else None}
            report.append(item)
            (output / 'coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({k: v for k, v in item.items() if k != 'regular_missing_sessions'}, ensure_ascii=False), flush=True)
    finally:
        if quote is not None:
            quote.close()
    summary = {'start': start, 'end': end, 'symbols': report,
               'complete': len(report) == len(symbols) and all(x['regular_complete'] for x in report),
               'no_R2_upload': True, 'shared_files_modified': False}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--symbols', nargs='+', default=['SOXL', 'SOXS', 'SOXX', 'QQQ'])
    parser.add_argument('--start', default='2024-10-14')
    parser.add_argument('--end', default='2026-10-09')
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=11111)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    result = acquire(args.output, args.symbols, args.start, args.end, args.host, args.port, args.offline)
    if not result['complete']:
        raise SystemExit(2)
