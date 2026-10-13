"""Independent auxiliary raw-bar, official-calendar and action reconciliation.

No learned forecast, strategy return or hindsight candidate filtering is read.
"""
from pathlib import Path
import hashlib
import json
import math
import datetime
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
DATA = ROOT / '.cache/soxl_soxs_14d5pct_v70/data'
CLOCKS = ['09:25:30', '10:05:30', '10:35:30', '11:05:30',
          '12:05:30', '13:05:30', '14:05:30', '15:05:30']
SYMBOLS = ['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPY']
ET = 'America/New_York'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def cash_matches(futu, ibkr):
    return Decimal(str(futu)) == Decimal(str(ibkr)).quantize(Decimal('.00001'), rounding=ROUND_HALF_UP)


def audit_raw_lineage(symbol, bars):
    receipts = [json.loads(line) for line in (DATA / 'acquisition.jsonl').read_text().splitlines()]
    base = []; fetched = []; captures = []
    columns = ['time_key', 'open', 'high', 'low', 'close', 'volume']
    for row in receipts:
        if row['symbol'] != symbol:
            continue
        provider = row['source']
        if provider == 'OpenD' and 'sha256' in row:
            path = DATA / 'captured/OpenD' / symbol / (
                f"{row['start']}_{row['end']}_{row['page']:04d}.parquet")
            parts = fetched
        elif provider == 'local':
            path = DATA / 'captured/local' / symbol / Path(row['path']).name
            parts = base
        elif provider == 'r2' and row.get('found'):
            path = DATA / 'captured/r2' / symbol / Path(row['key']).name
            parts = base
        else:
            continue
        assert sha(path) == row['sha256']
        raw = pd.read_parquet(path)
        if provider != 'OpenD':
            raw = raw[raw.price_basis.eq('NONE')] if 'price_basis' in raw else raw.iloc[:0]
        if 'code' in raw and len(raw):
            assert raw.code.eq('US.' + symbol).all()
        parts.append(raw[columns])
        captures.append({'path': str(path.relative_to(ROOT)), 'sha256': row['sha256']})
    empty = pd.DataFrame(columns=columns)
    first = pd.concat(base).drop_duplicates('time_key', keep='first') if base else empty
    last = pd.concat(fetched).drop_duplicates('time_key', keep='last') if fetched else empty
    actual = pd.concat([x for x in [first, last] if len(x)]).drop_duplicates('time_key', keep='last')
    end = pd.to_datetime(actual.time_key)
    dates = end.dt.normalize() + pd.to_timedelta((end.dt.hour * 60 + end.dt.minute > 1200).astype(int), unit='D')
    actual = actual[dates.between('2024-10-14', '2026-04-30')].set_index('time_key').sort_index()
    derived = bars.set_index('time_key').sort_index()
    assert actual.index.equals(derived.index)
    assert np.array_equal(actual[columns[1:]].to_numpy(float), derived[columns[1:]].to_numpy(float))
    return {'raw_captures': captures, 'raw_price_volume_values_verified': len(actual) * 5}


def reconcile_actions(symbol):
    path = DATA / 'captured/actions' / f'{symbol}.parquet'
    actions = pd.read_parquet(path)
    rows = actions[actions.ex_div_date.between('2024-10-14', '2026-04-30')]
    record = {'symbol': symbol, 'futu_sha256': sha(path), 'events': [],
              'errors': [], 'quote_only_reference': symbol == 'SPY'}
    if symbol == 'SPY':
        # Existing SPY v42 action capture is an additional local comparison.
        splits = rows[rows.split_ratio.notna() & rows.split_ratio.ne(1)]
        record.update(splits=splits[['ex_div_date', 'split_ratio']].to_dict('records'),
                      dividend_payment_dates_required_for_label=False)
        return record
    other_path = DATA / 'captured/ibkr' / f'{symbol}_actions.json'
    other = read(other_path)['corp_actions']
    record['ibkr_sha256'] = sha(other_path)
    supported = {'Splits': 'split_ratio', 'CashDividends': 'per_cash_div'}
    matched = set()
    for action in other:
        kind = action['type']
        date = pd.Timestamp(action['date']).strftime('%Y-%m-%d')
        subset = rows[rows.ex_div_date == date]
        if kind not in supported or len(subset) != 1:
            record['errors'].append({'date': date, 'kind': kind, 'reason': 'unmatched_action'})
            continue
        value = float(subset.iloc[0][supported[kind]])
        expected = 1 / float(action['value']) if kind == 'Splits' else float(action['value'])
        valid = math.isfinite(value) and (cash_matches(value, action['value'])
                 if kind == 'CashDividends' else abs(value - expected) <= 1e-12)
        if kind == 'CashDividends':
            valid = valid and bool(action.get('payment_date')) and (
                pd.Timestamp(action['payment_date']) >= pd.Timestamp(date))
        event = {'date': date, 'kind': kind, 'futu': value,
                 'ibkr_comparable': expected, 'passed': bool(valid),
                 'comparison': 'exact_five_decimal_cash_rounding' if kind == 'CashDividends' else 'reciprocal_split_ratio',
                 'contract_value_uses_full_IBKR_precision': kind == 'CashDividends'}
        record['events'].append(event)
        if not valid:
            record['errors'].append(event)
        matched.add((date, kind))
    for row in rows.itertuples():
        for kind, column in supported.items():
            value = getattr(row, column)
            if pd.notna(value) and value != (1 if kind == 'Splits' else 0):
                if (str(row.ex_div_date), kind) not in matched:
                    record['errors'].append({'date': str(row.ex_div_date), 'kind': kind,
                                             'reason': 'Futu_event_missing_in_IBKR'})
        for column in ['special_dividend', 'spin_off_ratio', 'allotment_ratio']:
            value = getattr(row, column)
            if pd.notna(value) and value != 0:
                record['errors'].append({'date': str(row.ex_div_date), 'kind': column,
                                         'reason': 'unsupported_corporate_action'})
    return record


def audit():
    registration = read(SOURCE / 'evidence/data_registration.json')
    assert all(sha(ROOT / name) == digest for name, digest in registration['source_sha256'].items())
    calendar = read(DATA / 'calendar.json')['sessions']
    original_calendar = read(ROOT / '.cache/soxl_soxs_14d5pct_v2/data/calendar.json')['sessions']
    assert calendar == original_calendar
    sessions = [x for x in calendar if '2024-10-14' <= x['session_date'] <= '2026-04-30']
    daily_checks = []; records = []; actions = []
    for symbol in SYMBOLS:
        path = DATA / 'bars' / f'{symbol}.parquet'
        bars = pd.read_parquet(path)
        for column in ['start_at', 'end_at', 'available_at']:
            bars[column] = pd.to_datetime(bars[column], utc=True).dt.as_unit('ns')
        original_ends = pd.to_datetime(bars.time_key).dt.tz_localize(ET).dt.tz_convert('UTC').dt.as_unit('ns')
        assert bars.end_at.equals(original_ends.rename('end_at'))
        assert bars.symbol.eq(symbol).all() and bars.price_basis.eq('NONE').all()
        assert bars.start_at.eq(bars.end_at - pd.Timedelta(minutes=5)).all()
        assert bars.available_at.eq(bars.end_at + pd.Timedelta(seconds=1)).all()
        assert bars.availability_basis.eq('historical_assumption_bar_end_plus_1s').all()
        assert bars.end_at.is_monotonic_increasing and not bars.end_at.duplicated().any()
        assert bars.session_date.between('2024-10-14', '2026-04-30').all()
        prices = bars[['open', 'high', 'low', 'close']].to_numpy(float)
        assert np.isfinite(prices).all() and (prices > 0).all()
        assert bars.high.ge(bars[['open', 'low', 'close']].max(axis=1)).all()
        assert bars.low.le(bars[['open', 'high', 'close']].min(axis=1)).all()
        assert np.isfinite(bars.volume).all() and bars.volume.ge(0).all()
        lineage = audit_raw_lineage(symbol, bars)
        bars = bars.set_index('end_at', drop=False)
        complete_regular = 0; complete_prefix = 0; total_clocks = 0; quoted_clocks = 0
        for session in sessions:
            date = session['session_date']
            left = pd.Timestamp(session['open_at']); right = pd.Timestamp(session['close_at'])
            expected = pd.date_range(left, right, freq='5min', inclusive='right')
            part = bars[bars.index.isin(expected)]
            regular_ok = part.index.equals(expected)
            complete_regular += int(regular_ok)
            clocks = []
            for clock in CLOCKS:
                at = pd.Timestamp(date + ' ' + clock, tz=ET).tz_convert('UTC')
                if at >= right:
                    continue
                end = at - pd.Timedelta(seconds=30)
                start = pd.Timestamp(date + ' 04:00:00', tz=ET).tz_convert('UTC')
                prefix = pd.date_range(start, end, freq='5min', inclusive='right')
                present = bars.index.intersection(prefix)
                quoted = end in bars.index
                ready = present.equals(prefix) and bars.loc[present, 'available_at'].le(at).all()
                clocks.append({'clock': clock, 'quoted': quoted,
                               'prefix_complete': bool(ready),
                               'missing_bars': len(prefix) - len(present)})
                total_clocks += 1; quoted_clocks += int(quoted); complete_prefix += int(ready)
            daily_checks.append({'symbol': symbol, 'date': date,
                                 'regular_complete': regular_ok,
                                 'regular_missing_bars': len(expected) - len(part),
                                 'clocks': clocks})
        record = {'symbol': symbol, 'sha256': sha(path), 'rows': len(bars),
                  'requested_sessions': len(sessions), 'complete_regular': complete_regular,
                  'expected_clocks': total_clocks, 'quoted_clocks': quoted_clocks,
                  'complete_04am_prefix_clocks': complete_prefix,
                  'regular_complete': complete_regular == len(sessions),
                  **lineage}
        records.append(record); actions.append(reconcile_actions(symbol))
    all_regular = all(x['regular_complete'] for x in records)
    all_actions = all(not x['errors'] for x in actions)
    result = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'passed': all_regular and all_actions, 'bars': records, 'actions': actions,
              'daily_checks': daily_checks, 'official_calendar_matches_original': True,
              'raw_data_only': True, 'strategy_effect_not_evaluated': True,
              'missing_premarket_not_imputed': True,
              'source_sha256': sha(Path(__file__)),
              'registration_sha256': sha(SOURCE / 'evidence/data_registration.json'),
              'limits': 'Historical bar_end+1s and current corporate-action capture are retrospective assumptions; '
                        'all regular bars do not imply complete premarket features or qualified training.'}
    (SOURCE / 'evidence/data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k not in ['daily_checks', 'actions']}, flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    audit()
