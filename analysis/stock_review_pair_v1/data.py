"""Daily QFQ Local -> R2 -> killable OpenD acquisition, with explicit coverage."""
from __future__ import annotations

import importlib.util
import math
import re
import shutil
import time
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import ROOT, digest, fingerprint, now, read, worker, write


def calendars():
    us_path = ROOT / 'scripts/model_history_calendar.py'
    spec = importlib.util.spec_from_file_location('pair_calendar', us_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse the official 2023-2026 transcription without changing frozen calendars.
    us = module.calendar(start='2023-04-01', end='2026-12-31')
    hk_path = ROOT / 'analysis/ai_trend_quadrant_v1/hk_calendar_2026.json'
    hk = read(hk_path)
    hk_days = [d.strftime('%Y-%m-%d') for d in pd.bdate_range(hk['start'], hk['end'])
               if d.strftime('%Y-%m-%d') not in hk['holidays']]
    return {
        'US': {'id': us['calendar_id'], 'start': us['start_date'], 'end': us['end_date'],
               'days': [r['session_date'] for r in us['sessions']],
               'closes': {r['session_date']: r.get('close_at_et', r.get('close_at', ''))
                          for r in us['sessions']},
               'early_close': [r['session_date'] for r in us['sessions'] if r['duration_minutes'] < 390],
               'source': us['sources'], 'sha256': digest(us_path)},
        'HK': {'id': hk['calendar_id'], 'start': hk['start'], 'end': hk['end'], 'days': hk_days,
               'early_close': [], 'holidays': hk['holidays'], 'source': hk['sources'],
               'sha256': digest(hk_path)},
    }


def cutoffs(cal, requested, instant=None):
    instant = instant or datetime.now(timezone.utc)
    result = {}
    for market, c in cal.items():
        local = instant.astimezone(ZoneInfo('America/New_York' if market == 'US' else 'Asia/Hong_Kong'))
        today = local.date().isoformat()
        required_end = today if requested == 'auto' else min(today, requested)
        if required_end > c['end']:
            raise ValueError(f'{market} calendar expires at {c["end"]}; cannot infer latest session')
        completed = []
        for day in c['days']:
            if day > today or (requested != 'auto' and day > requested):
                continue
            # US official calendar carries precise early closes. HK ordinary 16:00 is
            # a conservative availability cutoff on shortened sessions as well.
            close = c.get('closes', {}).get(day)
            if close:
                close_at = datetime.fromisoformat(close)
            else:
                hour = 13 if market == 'US' and day in c['early_close'] else 16
                close_at = datetime.fromisoformat(day + f'T{hour:02}:00:00').replace(tzinfo=local.tzinfo)
            if close_at <= instant:
                completed.append(day)
        if not completed:
            raise ValueError(f'No completed {market} session at requested cutoff')
        result[market] = completed[-1]
    return result


def membership(destination, args, deadline):
    old = read(ROOT / 'analysis/ai_trend_quadrant_v2/group_capture.json')
    if args.offline:
        capture = dict(old, live_check={'status': 'not_requested'})
    else:
        fresh = worker([], min(args.request_timeout, deadline - time.monotonic()), args.host,
                       args.port, capture=True)
        capture = fresh if fresh.get('ok') else dict(old, live_check=fresh)
        if fresh.get('ok'):
            # A failed group cannot silently erase old members or gain a fresh timestamp.
            capture['group_observed_at'] = {}
            for name, rows in old['watchlists'].items():
                if name not in capture['watchlists']:
                    capture['watchlists'][name] = rows
                    capture['group_observed_at'][name] = old['observed_at']
                else:
                    capture['group_observed_at'][name] = capture['observed_at']
            if any(e.get('group') == 'holdings' for e in capture['errors']):
                capture['holdings'] = old['holdings']
                capture['holdings_observed_at'] = old['observed_at']
    rows = {}
    excluded = []
    original = {r['code']: r for r in read(ROOT / 'analysis/ai_trend_quadrant_v2/results.json')['records']}
    old_comparison = {r['code']: r for r in read(ROOT / 'analysis/ai_trend_quadrant_v2/comparison.json')['records']}
    value_chain = read(ROOT / 'analysis/ai_value_chain_map_v1/data.json')
    company = {r['code']: r for r in value_chain['rows']}
    screen = read(ROOT / 'analysis/ai_value_chain_map_v1/quality/screen_current.json')
    ai = read(ROOT / 'analysis/ai_value_chain_map_v1/ai_basket_members.json')

    def add(item, group):
        code = item['code']
        kind = item.get('kind') or item.get('stock_type') or original.get(code, {}).get('kind', 'STOCK')
        if kind not in ['STOCK', 'ETF'] or not re.fullmatch(r'(US\.[A-Z][A-Z0-9.\-]{0,9}|HK\.\d{5})', code):
            excluded.append({'code': code, 'reason': 'unsupported_market_or_identity', 'group': group})
            return
        row = rows.setdefault(code, {**old_comparison.get(code, {}), 'code': code,
                                    'name': item.get('name') or code, 'kind': kind, 'groups': []})
        if group not in row['groups']:
            row['groups'].append(group)
        listing = str(item.get('listing_date') or original.get(code, {}).get('listing_date') or '')
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', listing) and listing > '1970-01-01':
            row['listing_date'] = listing
        rating = screen['records'].get(code, {})
        row['quality'] = {k: rating.get(k) for k in ['grade', 'verified_grade', 'status', 'as_of',
                                                   'formal_quality_as_of', 'expires_at']}
        business = company.get(code, {}).get('business', {})
        row['chains'] = business.get('groups', [])
        row['chain_label'] = {'status': 'captured' if code in company else 'not_in_source',
                             'primary': business.get('primary'), 'review_status': business.get('review_status')}
        row['symbol'] = code.split('.', 1)[1]
        row['structure'] = original.get(code, {}).get('structure')

    for name, key in [('全部', 'watchlist'), ('特别关注', 'special')]:
        for item in capture['watchlists'].get(name, []):
            add(item, key)
    for item in capture['holdings']:
        add(item, 'holdings')
    for item in ai['members']:
        grade = screen['records'].get(item['code'], {}).get('grade')
        if grade in ['A+', 'A']:
            add(item, 'ai_plus' if grade == 'A+' else 'ai_a')
    inputs = destination / 'inputs'
    write(inputs / 'members.json', {'observed_at': capture['observed_at'], 'capture': capture,
                                    'records': list(rows.values()), 'excluded': excluded,
                                    'membership_not_PIT': True})
    for name, source in [('quality.json', ROOT / 'analysis/ai_value_chain_map_v1/quality/screen_current.json'),
                         ('value_chain.json', ROOT / 'analysis/ai_value_chain_map_v1/data.json'),
                         ('intraday.json', ROOT / 'analysis/operation_cadence_v1/intraday.json')]:
        # Keep all legacy evidence untouched; its real source dates remain in the snapshot.
        shutil.copyfile(source, inputs / name)
    fundamental_source = ROOT / 'analysis/operation_cadence_v1/results.json'
    fundamental_rows = []
    for row in read(fundamental_source)['records']:
        facts = dict(row['fundamental'])
        for key in ['long_eligibility', 'training_included', 'grade', 'screen_grade', 'grade_type', 'status']:
            facts.pop(key, None)
        fundamental_rows.append({'code': row['code'], 'fundamental': facts})
    write(inputs / 'fundamentals.json', {'records': fundamental_rows, 'source_sha256': digest(fundamental_source),
                                       'source_status': 'carried_forward'})
    return rows, capture, value_chain


def normalize(frame, start, end, listing=None):
    if frame.empty or not {'open', 'high', 'low', 'close'} <= set(frame):
        return []
    frame = frame.copy()
    date_col = next((k for k in ['time_key', 'session_date', 'day'] if k in frame), None)
    if date_col is None:
        return []
    frame['day'] = frame[date_col].astype(str).str[:10]
    frame = frame[(frame.day >= max(start, listing or start)) & (frame.day <= end)].sort_values('day')
    if frame.day.duplicated().any():
        raise ValueError('duplicate daily dates')
    result = []
    for row in frame.to_dict('records'):
        o, h, l, c = [float(row[k]) for k in ['open', 'high', 'low', 'close']]
        if not all(math.isfinite(v) and v > 0 for v in [o, h, l, c]) or h < max(o, c, l) or l > min(o, c, h):
            continue
        volume = float(row.get('volume') or 0)
        result.append([row['day'], o, h, l, c, volume if math.isfinite(volume) else 0])
    return result


def coverage(bars, calendar, start, end, listing=None):
    present = {b[0] for b in bars}
    required = [d for d in calendar['days'] if max(start, listing or start) <= d <= end]
    missing = [d for d in required if d not in present]
    internal = [d for d in missing if bars and bars[0][0] <= d <= bars[-1][0]]
    unknown_calendar = start < calendar['start']
    status = ('unavailable' if not bars else 'internal_gaps' if internal else
              'stale' if bars[-1][0] != end else 'short_history' if missing else
              'calendar_partial' if unknown_calendar else 'complete')
    return {'status': status, 'requested_start': start, 'requested_end': end,
            'actual_start': bars[0][0] if bars else None, 'actual_end': bars[-1][0] if bars else None,
            'bars': len(bars), 'missing_sessions': missing, 'internal_missing': internal,
            'calendar_partial': unknown_calendar}


class Acquisition:
    def __init__(self, args, destination, cal, ends, deadline, shared=None):
        from storage import SharedData
        self.shared = shared or SharedData(args.store)
        self.staging = tempfile.TemporaryDirectory(prefix='price-capture-')
        self.args, self.destination, self.cal, self.ends, self.deadline = args, destination, cal, ends, deadline
        self.start = (pd.Timestamp(max(ends.values())) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        self.index = defaultdict(list)
        roots = [ROOT / 'market_data/trend_quadrant_v1/daily',
                 ROOT / 'market_data/operation_cadence_v1', ROOT / 'market_data/stock_traits_daily_v1',
                 args.store / 'runs', args.store / 'data/raw']
        for root in roots:
            for p in root.rglob('*.parquet'):
                if root.name == 'raw':
                    self.index[p.parent.name].append(p)
                elif root.name == 'daily' or 'daily' in p.parts or 'opend_day_qfq' in p.parts or 'market' in p.parts:
                    self.index[p.stem].append(p)
        self.remote = None
        self.client = None
        self.errors = []
        self.quota = None
        self.new_symbols = 0
        self.last_call = 0

    def cleanup(self):
        self.staging.cleanup()

    def local(self, row):
        code = row['code']
        candidates = []
        for path in self.index[code]:
            try:
                bars = normalize(pd.read_parquet(path), self.start, self.ends[code[:2]], row.get('listing_date'))
                if bars:
                    candidates.append((bars[-1][0], len(bars), path.stat().st_mtime, path, bars))
            except Exception:
                continue
        return max(candidates, default=None, key=lambda c: self.rank(c, row))

    def rank(self, candidate, row):
        c = coverage(candidate[4], self.cal[row['code'][:2]], self.start,
                     self.ends[row['code'][:2]], row.get('listing_date'))
        return (c['actual_end'], -len(c['internal_missing']), len(candidate[4]), candidate[2])

    def r2(self, row):
        if self.args.offline or time.monotonic() >= self.deadline:
            return []
        if self.remote is None:
            spec = importlib.util.spec_from_file_location('review_pair_r2', ROOT / 'scripts/r2_client.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.client = module.R2Client(timeout=min(5, self.args.request_timeout), deadline=self.deadline)
            self.remote = []
            for prefix in ['market_data/trend_quadrant_v1/daily/', 'market_data/operation_cadence_v1/']:
                try:
                    self.remote.extend(self.client.list_objects(prefix))
                except Exception as exc:
                    self.errors.append({'source': 'R2', 'prefix': prefix, 'error': type(exc).__name__})
        result = []
        for obj in self.remote:
            if Path(obj['key']).name != row['code'] + '.parquet' or time.monotonic() >= self.deadline:
                continue
            dest = Path(self.staging.name) / 'r2_cache' / (fingerprint(obj['key']) + '.parquet')
            try:
                self.client.get_object(obj['key'], dest)
                bars = normalize(pd.read_parquet(dest), self.start, self.ends[row['code'][:2]], row.get('listing_date'))
                if bars:
                    result.append((bars[-1][0], len(bars), 0, dest, bars, obj['key']))
            except Exception as exc:
                self.errors.append({'source': 'R2', 'code': row['code'], 'error': type(exc).__name__})
        return result

    def opend(self, row):
        if self.args.offline or self.args.no_opend or time.monotonic() >= self.deadline:
            return None
        if self.quota is None:
            probe = worker(['probe', '--quota-detail'], min(self.args.request_timeout, self.deadline - time.monotonic()),
                           self.args.host, self.args.port)
            self.quota = probe.get('history_kl_quota', {})
        free = row['code'] in self.quota.get('codes', [])
        if not free and (self.new_symbols >= self.args.max_new_symbols or
                         self.quota.get('remain', 0) - self.new_symbols <= self.args.min_remaining_quota):
            self.errors.append({'code': row['code'], 'source': 'OpenD', 'error': 'quota_budget'})
            return None
        pause = max(0, 1.1 - (time.monotonic() - self.last_call))
        if time.monotonic() + pause >= self.deadline:
            return None
        time.sleep(pause)
        self.last_call = time.monotonic()
        dest = Path(self.staging.name) / 'opend_day_qfq' / (row['code'] + '.parquet')
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = worker(['kline_day', '--code', row['code'], '--start', self.start,
                         '--end', self.ends[row['code'][:2]], '--out', str(dest), '--autype', 'QFQ'],
                        min(self.args.request_timeout, self.deadline - time.monotonic()), self.args.host, self.args.port)
        if not free:
            self.new_symbols += 1  # budget attempts conservatively, including failures
        if result.get('ok') and dest.exists():
            bars = normalize(pd.read_parquet(dest), self.start, self.ends[row['code'][:2]], row.get('listing_date'))
            if bars:
                return (bars[-1][0], len(bars), 0, dest, bars)
        self.errors.append({'code': row['code'], 'source': 'OpenD',
                            'error': result.get('error_type') or 'fetch_failed'})
        return None

    def acquire(self, row):
        code, market = row['code'], row['code'][:2]
        end = self.ends[market]
        candidate = self.local(row)
        source = 'local_qfq'
        ok = lambda c: c and coverage(c[4], self.cal[market], self.start, end,
                                     row.get('listing_date'))['status'] in ['complete', 'calendar_partial']
        if not ok(candidate):
            remote = self.r2(row)
            for entry in remote:
                if candidate is None or self.rank(entry, row) > self.rank(candidate, row):
                    candidate, source = entry, 'r2_qfq'
        if not ok(candidate):
            fresh = self.opend(row)
            if fresh:
                # A new response is a whole adjusted series; never splice old/new QFQ.
                candidate, source = fresh, 'OpenD_daily_QFQ_full_capture'
        bars = candidate[4] if candidate else []
        details = coverage(bars, self.cal[market], self.start, end, row.get('listing_date'))
        entry = {'security_id': code, 'market': market, 'price_basis': 'QFQ_full_series_single_capture',
                 'currency': 'USD' if market == 'US' else 'HKD', 'source': source if candidate else None,
                 'available_at': None, 'checked_at': now(), 'coverage': details,
                 'dataset_id': fingerprint([code, bars]), 'bars': bars}
        if candidate:
            # Copying an old file does not turn it into newly received evidence.
            entry.update(sha256=digest(candidate[3]),
                         source_sha256=digest(candidate[3]), source_path=str(candidate[3]),
                         received_at=datetime.fromtimestamp(candidate[3].stat().st_mtime, timezone.utc).isoformat(),
                         received_at_precision='file_mtime_proxy_not_first_receipt')
        return self.shared.prices(entry, bars, candidate[3] if candidate else None)
