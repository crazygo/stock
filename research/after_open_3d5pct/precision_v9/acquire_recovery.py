"""Explicit R03 acquisition recovery version; never modifies its source run.

Run only after registration/selection by the R03 owner. This module has no
import-time network work and does not upload or access trading accounts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

import futu as ft
import pandas as pd

from scripts.fetch_research_data import normalize_kline_dataframe
from scripts.r2_client import R2Client
from . import acquire_batch as batch
from . import acquire_pilot as pilot
from .evaluate import HERE, ROOT, sha


VERSION = 'r03_acquisition_recovery_v1'
LOCK = '/tmp/stock_futu_acquisition.lock'
TEXT_COLUMNS = ('symbol', 'time_key', 'start_at', 'end_at', 'available_at',
                'start_at_et', 'end_at_et', 'session_date', 'session_type', 'price_basis')
FLOAT_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'turnover',
                 'pe_ratio', 'turnover_rate', 'change_rate', 'last_close')
CANONICAL_COLUMNS = ('symbol', 'time_key', 'start_at', 'end_at', 'available_at',
                     'start_at_et', 'end_at_et', 'session_date', 'session_type',
                     'open', 'high', 'low', 'close', 'volume', 'turnover',
                     'pe_ratio', 'turnover_rate', 'change_rate', 'last_close',
                     'price_basis')


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def canonical_empty() -> pd.DataFrame:
    """Schema of the normalizer's nonempty output, with no invented bars."""
    cols = {name: pd.Series(dtype='string') for name in TEXT_COLUMNS}
    cols.update({name: pd.Series(dtype='float64') for name in FLOAT_COLUMNS})
    return pd.DataFrame(cols)[list(CANONICAL_COLUMNS)]


def normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return canonical_empty()
    return batch.normalize(raw, symbol)


def _month_bounds(month: str) -> tuple[str, str]:
    for start, end in batch.MONTHS:
        if start[:7] == month:
            return start, end
    raise ValueError(f'outside frozen months: {month}')


def verified_complete(part: Path, symbol: str, month: str) -> dict:
    """Accept only a complete checkpoint whose stored bars still match it."""
    result = json.loads((part / 'result.json').read_text())
    if result.get('symbol') != symbol or result.get('month') != month:
        raise ValueError(f'checkpoint identity mismatch: {part}')
    if result.get('pagination_complete') is not True or result.get('failure') is not None:
        raise ValueError(f'incomplete checkpoint: {part}')
    bars = part / 'bars.parquet'
    if not bars.is_file() or not result.get('bars_sha256') or sha(bars) != result['bars_sha256']:
        raise ValueError(f'checkpoint bars missing or changed: {part}')
    frame = pd.read_parquet(bars)
    if list(frame.columns) != list(CANONICAL_COLUMNS):
        raise ValueError(f'checkpoint canonical schema mismatch: {part}')
    if result.get('quality') != batch.quality(frame, *_month_bounds(month), batch.make_calendar()):
        raise ValueError(f'checkpoint quality mismatch: {part}')
    if not (part / 'audit.json').is_file():
        raise ValueError(f'checkpoint audit missing: {part}')
    return result


def _source_registration(source: Path) -> dict:
    registration = json.loads((source / 'registration.json').read_text())
    config = registration['config']
    if config.get('months') != [list(m) for m in batch.MONTHS] or config.get('session') != 'ALL' \
            or config.get('basis') != 'NONE' or config.get('no_upload') is not True:
        raise ValueError('source does not match frozen R03 acquisition scope')
    if not config.get('symbols') or len(set(config['symbols'])) != len(config['symbols']):
        raise ValueError('invalid source symbols')
    snapshots = source / 'source_snapshot'
    if sha(snapshots / 'acquire_batch.py') != config.get('code_sha') or \
            sha(snapshots / 'acquire_pilot.py') != config.get('normalizer_sha'):
        raise ValueError('source code snapshot differs from registration')
    return registration


def _identity(source: Path) -> dict:
    return {'version': VERSION, 'source_run': str(source.resolve()),
            'source_registration_sha256': sha(source / 'registration.json'),
            'source_registration': _source_registration(source),
            'code_sha256': sha(Path(__file__)),
            'import_sha256': {
                'acquire_batch.py': sha(HERE / 'acquire_batch.py'),
                'acquire_pilot.py': sha(HERE / 'acquire_pilot.py'),
                'fetch_research_data.py': sha(ROOT / 'scripts/fetch_research_data.py'),
                'r2_client.py': sha(ROOT / 'scripts/r2_client.py'),
                'evaluate.py': sha(HERE / 'evaluate.py'),
            },
            'retry_limits': {'pages_per_month': 30, 'attempts_per_page': 3},
            'source_order': 'Local -> R2 -> Futu', 'session': 'ALL',
            'price_basis': 'NONE', 'no_upload': True, 'worker_count': 1}


def _initialize(source: Path, output: Path) -> dict:
    identity = _identity(source)
    reg = output / 'registration.json'
    if reg.exists():
        saved = json.loads(reg.read_text())
        if saved['identity'] != identity:
            raise ValueError('changed recovery job identity or imported source hash')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'nonempty unregistered output: {output}')
        output.mkdir(parents=True, exist_ok=True)
        _write(reg, {'created_at': datetime.now(timezone.utc).isoformat(), 'identity': identity})
        _write(output / 'calendar.json', batch.make_calendar())
    if json.loads((output / 'calendar.json').read_text()) != batch.make_calendar():
        raise ValueError('changed recovery calendar')
    return identity


def _inherit(source_part: Path, output_part: Path, symbol: str, month: str,
             output: Path) -> dict:
    result = verified_complete(source_part, symbol, month)
    link_record = output / 'inherited' / symbol / f'{month}.json'
    record = {'source_part': str(source_part.resolve()),
              'source_result_sha256': sha(source_part / 'result.json'),
              'source_audit_sha256': sha(source_part / 'audit.json'),
              'source_bars_sha256': result['bars_sha256']}
    if link_record.exists():
        if json.loads(link_record.read_text()) != record:
            raise ValueError(f'inherited checkpoint changed: {source_part}')
    else:
        _write(link_record, record)
    if output_part.is_symlink():
        if output_part.resolve() != source_part.resolve():
            raise ValueError(f'wrong inherited link: {output_part}')
    elif output_part.exists():
        raise ValueError(f'conflicting output part: {output_part}')
    else:
        output_part.parent.mkdir(parents=True, exist_ok=True)
        output_part.symlink_to(source_part.resolve(), target_is_directory=True)
    return result


def _new_result(part: Path, symbol: str, month: str, tier: str, frame: pd.DataFrame | None,
                complete: bool, failure: str | None, audit: list, cal: dict,
                elapsed: float) -> dict:
    if frame is not None:
        frame.to_parquet(part / 'bars.parquet', index=False, compression='zstd', compression_level=7)
    empty_success = bool(complete and frame is not None and frame.empty and tier == 'futu')
    result = {'symbol': symbol, 'month': month, 'tier': tier,
              'pagination_complete': bool(complete), 'failure': failure,
              'quality': batch.quality(frame, *_month_bounds(month), cal) if frame is not None else None,
              'seconds': elapsed,
              'bars_sha256': sha(part / 'bars.parquet') if frame is not None else None,
              'data_state': 'provider_returned_no_bars_unknown_listing_state' if empty_success
                            else ('unresolved_failure' if not complete else 'observed_bars'),
              'empty_success': empty_success}
    _write(part / 'audit.json', audit)
    _write(part / 'result.json', result)
    return result


def _fetch_pages(part: Path, symbol: str, start: str, end: str, context, sleeper=time.sleep):
    """Return raw pages and audit; an exhausted page/attempt budget stays failed."""
    audit, pages, cursor = [], [], None
    for page in range(30):
        sleeper(1.2 if page == 0 else .3)
        ret, chunk, nxt = None, None, None
        for attempt in range(3):
            sent = datetime.now(timezone.utc).isoformat()
            try:
                ret, chunk, nxt = context().request_history_kline(
                    'US.' + symbol, start=start, end=end, ktype=ft.KLType.K_5M,
                    autype=ft.AuType.NONE, max_count=1000, extended_time=True,
                    session=ft.Session.ALL, page_req_key=cursor)
            except Exception as exc:
                ret, chunk = None, f'{type(exc).__name__}: {str(exc)[:180]}'
            event = {'page': page + 1, 'attempt': attempt + 1,
                     'requested_at': sent, 'received_at': datetime.now(timezone.utc).isoformat(),
                     'ok': ret == ft.RET_OK}
            if ret == ft.RET_OK:
                if not isinstance(chunk, pd.DataFrame):
                    ret, chunk = None, 'successful provider reply was not a DataFrame'
                    event['ok'] = False
                else:
                    raw = part / f'raw_{page + 1:03d}.parquet'
                    chunk.to_parquet(raw, index=False, compression='zstd', compression_level=7)
                    event.update(rows=len(chunk), sha=sha(raw), has_more=nxt is not None)
                    audit.append(event)
                    _write(part / 'audit.json', audit)
                    break
            event['error'] = str(chunk)[:250]
            audit.append(event)
            _write(part / 'audit.json', audit)
            if attempt < 2:
                sleeper(30)
        if ret != ft.RET_OK:
            return pages, audit, False, f'provider_error: {str(chunk)[:250]}'
        pages.append(chunk)
        cursor = nxt
        if cursor is None:
            return pages, audit, True, None
    return pages, audit, False, 'page_cap'


def _progress(output: Path, results: list[dict], total: int, checkpoint=False) -> dict:
    complete = sum(r.get('pagination_complete') is True and not r.get('failure') for r in results)
    empty = sum(r.get('empty_success') is True for r in results)
    unresolved = sum(r.get('pagination_complete') is not True or bool(r.get('failure')) for r in results)
    pending = total - len(results)
    status = ('runtime_checkpoint' if checkpoint else
              'acquisition_complete' if pending == 0 and unresolved == 0 else
              'incomplete_unresolved')
    summary = {'status': status, 'total_parts': total, 'complete_parts': complete,
               'empty_successful_parts': empty, 'unresolved_failures': unresolved,
               'pending_parts': pending, 'results': results, 'no_training': True,
               'no_upload': True}
    _write(output / 'progress.json', summary)
    if not checkpoint and pending == 0:
        _write(output / 'summary.json', summary)
    return summary


def run(source: Path, output: Path, max_minutes=180, *, retry_source_incomplete=False) -> dict:
    if max_minutes <= 0:
        raise ValueError('max_minutes must be positive')
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('source and output must be separate runs')
    identity = _initialize(source, output)
    config = identity['source_registration']['config']
    cal = json.loads((output / 'calendar.json').read_text())
    total = len(config['symbols']) * len(config['months'])
    started, results, checked, remote = time.monotonic(), [], set(), {}
    lock = open(LOCK, 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    quote = None
    def context():
        nonlocal quote
        if quote is None:
            ft.SysConfig.enable_proto_encrypt(False)
            quote = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
            ret, quota = quote.get_history_kl_quota(get_detail=False)
            if ret == ft.RET_OK:
                _write(output / 'quota_before.json', {'used': int(quota[0]), 'remaining': int(quota[1])})
        return quote
    try:
        for symbol in config['symbols']:
            for start, end in config['months']:
                month = start[:7]
                source_part = source / 'parts' / symbol / month
                part = output / 'parts' / symbol / month
                if part.is_symlink():
                    results.append(_inherit(source_part, part, symbol, month, output))
                    continue
                if part.exists():
                    if (part / 'result.json').exists():
                        prior = json.loads((part / 'result.json').read_text())
                        if prior.get('pagination_complete') is True:
                            results.append(verified_complete(part, symbol, month))
                        else:
                            results.append(prior)
                        continue
                    raise ValueError(f'uncommitted recovery part requires new output: {part}')
                if source_part.is_dir() and (source_part / 'result.json').exists():
                    source_meta = json.loads((source_part / 'result.json').read_text())
                    if source_meta.get('pagination_complete') is True:
                        results.append(_inherit(source_part, part, symbol, month, output))
                        continue
                    if not retry_source_incomplete:
                        results.append({'symbol': symbol, 'month': month, 'pagination_complete': False,
                                        'failure': 'source_incomplete_requires_explicit_retry',
                                        'source_part': str(source_part)})
                        continue
                elif source_part.exists() and not retry_source_incomplete:
                    results.append({'symbol': symbol, 'month': month, 'pagination_complete': False,
                                    'failure': 'source_uncommitted_requires_explicit_retry',
                                    'source_part': str(source_part)})
                    continue
                if time.monotonic() - started > max_minutes * 60:
                    return _progress(output, results, total, checkpoint=True)
                part.mkdir(parents=True)
                timer, audit, frame, tier = time.monotonic(), [], None, None
                try:
                    local_paths = [ROOT / 'market_data/us_5m' / symbol / '2025.parquet',
                                   HERE.parent / 'runs/precision_v9_20260927/R02_normalized/market_data/us_5m' / symbol / '2025.parquet']
                    for path in local_paths:
                        if path.exists():
                            candidate = pd.read_parquet(path)
                            if batch.quality(candidate, start, end, cal)['rth_complete']:
                                frame = candidate[(candidate.session_date >= start) & (candidate.session_date <= end)].copy()
                                tier = 'local'
                                audit.append({'tier': tier, 'source': str(path), 'sha': sha(path)})
                                break
                    if frame is None:
                        if symbol not in checked:
                            key = f'us_5m/{symbol}/2025.parquet'
                            cache = output / 'r2_cache' / key
                            try:
                                found = R2Client().head_object(key)
                                audit.append({'tier': 'r2', 'key': key, 'found': found is not None})
                                if found:
                                    R2Client().get_object(key, cache)
                                    remote[symbol] = cache
                            except Exception as exc:
                                audit.append({'tier': 'r2', 'error_type': type(exc).__name__})
                            checked.add(symbol)
                        if symbol in remote:
                            candidate = pd.read_parquet(remote[symbol])
                            if batch.quality(candidate, start, end, cal)['rth_complete']:
                                frame = candidate[(candidate.session_date >= start) & (candidate.session_date <= end)].copy()
                                tier = 'r2'
                                audit.append({'tier': 'r2', 'source': str(remote[symbol]), 'sha': sha(remote[symbol])})
                    complete, failure = True, None
                    if frame is None:
                        tier = 'futu'
                        pages, events, complete, failure = _fetch_pages(part, symbol, start, end, context)
                        audit.extend(events)
                        if pages:
                            raw = pd.concat(pages, ignore_index=True)
                            if not raw.empty and (not 'time_key' in raw or raw.time_key.duplicated().any()):
                                complete, failure = False, 'duplicate_or_missing_source_times'
                            frame = normalize(raw, symbol)
                    result = _new_result(part, symbol, month, tier, frame, complete, failure,
                                         audit, cal, time.monotonic() - timer)
                except Exception as exc:
                    audit.append({'error_type': type(exc).__name__, 'error': str(exc)[:250]})
                    result = _new_result(part, symbol, month, tier or 'unknown', None, False,
                                         f'exception: {type(exc).__name__}', audit, cal,
                                         time.monotonic() - timer)
                results.append(result)
                _progress(output, results, total, checkpoint=True)
        return _progress(output, results, total)
    finally:
        if quote is not None:
            try:
                ret, quota = quote.get_history_kl_quota(get_detail=False)
                if ret == ft.RET_OK:
                    _write(output / 'quota_after.json', {'used': int(quota[0]), 'remaining': int(quota[1])})
            finally:
                quote.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-minutes', type=int, default=180)
    parser.add_argument('--retry-source-incomplete', action='store_true')
    args = parser.parse_args()
    summary = run(args.source, args.output, args.max_minutes,
                  retry_source_incomplete=args.retry_source_incomplete)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
