"""Independent raw-price replay on fixed auxiliary contract fixtures."""
import datetime
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from market_adapter import load, label, SOURCE, DATA, sha

DATES = ['2024-11-06', '2024-11-07', '2025-03-25', '2025-11-19',
         '2025-11-20', '2025-11-28', '2025-12-23', '2026-04-30']
CLOCKS = ['09:25:30', '10:05:30', '10:35:30', '11:05:30',
          '12:05:30', '13:05:30', '14:05:30', '15:05:30']


def independent(market, symbol, at):
    raw = pd.read_parquet(DATA / 'bars' / f'{symbol}.parquet')
    raw['end'] = pd.to_datetime(raw.time_key).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    raw['start'] = raw.end - pd.Timedelta(minutes=5)
    date = at.tz_convert('America/New_York').tz_localize(None).normalize()
    entry_at = (at + pd.Timedelta(seconds=30)).floor('5min') + pd.Timedelta(minutes=5)
    entry_at = max(entry_at, pd.Timestamp(market.calendar[date]['open_at']))
    entry_rows = raw[raw.start.eq(entry_at)]
    if len(entry_rows) != 1:
        return {'reason': 'no_entry_bar'}
    actions = market.actions[symbol]
    factor = lambda day: math.prod(float(a['value']) for a in actions
                                   if a['type'] == 'Splits' and pd.Timestamp(a['date']) <= day)
    fill = float(entry_rows.open.iloc[0]) * 1.0005
    scale = factor(date); target = math.ceil(fill * 1.05 * 100 - 1e-10) / 100 * scale
    economic_entry = fill * scale
    expiry_date = max(day for day in market.sessions if day <= date + pd.Timedelta(days=13))
    expiry = pd.Timestamp(market.calendar[expiry_date]['close_at'])
    rows = []; event = None; split_days = set()
    for day in market.sessions[(market.sessions >= date) & (market.sessions <= expiry_date)]:
        session = market.calendar[day]
        left = max(entry_at, pd.Timestamp(session['open_at'])); right = pd.Timestamp(session['close_at'])
        dayrows = raw[raw.start.ge(left) & raw.end.le(right) & raw.start.lt(right)]
        for bar in dayrows.itertuples():
            sf = factor(day)
            target_raw = math.ceil(target / sf * 100 - 1e-10) / 100
            stop_raw = math.floor(economic_entry * .94 / sf * 100 + 1e-10) / 100
            rows.append(bar)
            if sf != scale:
                split_days.add(str(day.date()))
            if bar.low <= stop_raw:
                event = ('risk_stop', bar.end, min(bar.open, stop_raw) * .9995 * sf)
                break
            if bar.high >= target_raw:
                event = ('target', bar.end, target_raw * sf)
                break
        if event:
            break
    if event is None:
        if len(rows) and rows[-1].end == expiry:
            event = ('expiry', expiry, rows[-1].close * .9995 * factor(expiry_date))
        else:
            return {'reason': 'pending', 'entry_at': entry_at, 'entry_fill_raw': fill}
    reason, end, value = event
    exit_date = end.tz_convert('America/New_York').tz_localize(None).normalize()
    cash = sum(float(a['value']) * factor(pd.Timestamp(a['date'])) for a in actions
               if a['type'] == 'CashDividends' and date < pd.Timestamp(a['date']) <= exit_date)
    net = (value * .9995 + cash) / (economic_entry * 1.0005) - 1
    return {'reason': reason, 'entry_at': entry_at, 'exit_at': end,
            'entry_fill_raw': fill, 'exit_fill_economic': value,
            'dividend_per_economic_unit': cash, 'net_return': net,
            'split_days_in_contract': sorted(split_days)}


def audit():
    market = load(); results = []; outcomes = {}; check_values = 0
    for symbol in ['TQQQ', 'SQQQ', 'UPRO', 'SPXU']:
        for date in DATES:
            for clock in CLOCKS:
                at = pd.Timestamp(date + ' ' + clock, tz='America/New_York').tz_convert('UTC')
                if at >= pd.Timestamp(market.calendar[pd.Timestamp(date)]['close_at']):
                    continue
                actual = label(market, symbol, at); expected = independent(market, symbol, at)
                assert actual['reason'] == expected['reason'], (symbol, at, actual['reason'], expected['reason'])
                assert actual['entry_at'] == expected['entry_at']
                for key in ['entry_fill_raw', 'exit_fill_economic', 'dividend_per_economic_unit', 'net_return']:
                    if key in expected:
                        assert math.isclose(float(actual[key]), float(expected[key]), rel_tol=1e-11, abs_tol=1e-11), (symbol, at, key)
                        check_values += 1
                if 'exit_at' in expected:
                    assert actual['exit_at'] == expected['exit_at']
                outcomes[actual['reason']] = outcomes.get(actual['reason'], 0) + 1
                results.append({'symbol': symbol, 'at': str(at), **expected})
    output = {'passed': True, 'contracts': len(results), 'numeric_comparisons': check_values,
              'fixed_dates': DATES, 'outcomes': outcomes, 'rows': results,
              'adapter_sha256': sha(SOURCE / 'market_adapter.py'),
              'source_sha256': sha(Path(__file__)),
              'data_audit_sha256': sha(SOURCE / 'evidence/data_audit.json'),
              'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'strategy_outcomes_evaluated': False,
              'scope': 'Fixed hypothetical auxiliary contracts around corporate actions, a half-day, and the data cutoff. Not independent strategy observations or a tradable auxiliary portfolio.'}
    (SOURCE / 'evidence/contract_adapter_audit.json').write_text(json.dumps(output, indent=2, default=str) + '\n')
    print({k: v for k, v in output.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    audit()
