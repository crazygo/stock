"""Load only audited auxiliary histories into the original contract data type."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/soxl_soxs_14d5pct_v2'))
import core

DATA = ROOT / '.cache/soxl_soxs_14d5pct_v70/data'
SOURCE = Path(__file__).resolve().parent
SYMBOLS = ['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPY']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def load():
    audit = read(SOURCE / 'evidence/data_audit.json')
    if not audit['passed']:
        raise ValueError('Auxiliary data audit must pass before contract construction')
    calendar = {pd.Timestamp(row['session_date']): row
                for row in read(DATA / 'calendar.json')['sessions']}
    sessions = pd.DatetimeIndex(sorted(calendar))
    bars = {}; daily = {}; actions = {}; complete = {}; manifest = {}
    for symbol in SYMBOLS:
        path = DATA / 'bars' / f'{symbol}.parquet'
        receipt = next(row for row in audit['bars'] if row['symbol'] == symbol)
        assert sha(path) == receipt['sha256']
        action_receipt = next(row for row in audit['actions'] if row['symbol'] == symbol)
        action_path = DATA / 'captured/actions' / f'{symbol}.parquet'
        assert sha(action_path) == action_receipt['futu_sha256']
        if symbol == 'SPY':
            table = pd.read_parquet(action_path)
            table = table[table.ex_div_date.between('2024-10-14', '2026-04-30')]
            actions[symbol] = [{'type': 'Splits', 'date': row.ex_div_date,
                                'value': str(1 / float(row.split_ratio))}
                               for row in table.itertuples()
                               if pd.notna(row.split_ratio) and row.split_ratio != 1]
        else:
            other_path = DATA / 'captured/ibkr' / f'{symbol}_actions.json'
            assert sha(other_path) == action_receipt['ibkr_sha256']
            actions[symbol] = read(other_path)['corp_actions']
        frame = pd.read_parquet(path)
        for column in ['start_at', 'end_at', 'available_at']:
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit('ns')
        frame['date'] = pd.to_datetime(frame.session_date)
        factor = np.ones(len(frame))
        for action in actions[symbol]:
            if action['type'] == 'Splits':
                factor[frame.date >= pd.Timestamp(action['date'])] *= float(action['value'])
        frame['share_factor'] = factor
        for column in ['open', 'high', 'low', 'close']:
            frame['econ_' + column] = frame[column] * factor
        frame['econ_volume'] = frame.volume / factor
        opening = pd.to_datetime(frame.date.map({d: row['open_at'] for d, row in calendar.items()}), utc=True)
        closing = pd.to_datetime(frame.date.map({d: row['close_at'] for d, row in calendar.items()}), utc=True)
        inside = frame.start_at.ge(opening) & frame.end_at.le(closing)
        after = frame.start_at.ge(closing) & frame.session_type.ne('overnight')
        frame.loc[inside, 'session_type'] = 'regular'
        frame.loc[after, 'session_type'] = 'post_market'
        frame = frame.set_index('end_at', drop=False); bars[symbol] = frame
        regular = frame[frame.session_type.eq('regular')]
        checks = {}
        for date, group in regular.groupby('date'):
            row = calendar[date]
            expected = pd.date_range(row['open_at'], row['close_at'], freq='5min', inclusive='right')
            checks[date] = group.index.equals(expected)
        complete[symbol] = checks
        d = regular.groupby('date').agg(open=('econ_open', 'first'), high=('econ_high', 'max'),
             low=('econ_low', 'min'), close=('econ_close', 'last'), volume=('econ_volume', 'sum'),
             raw_close=('close', 'last'), share_factor=('share_factor', 'last'),
             available_at=('available_at', 'max'))
        d['complete'] = [checks[date] for date in d.index]; daily[symbol] = d
        manifest[symbol] = {'bars_sha256': receipt['sha256'],
                            'actions_sha256': action_receipt['futu_sha256'],
                            'ibkr_actions_sha256': action_receipt.get('ibkr_sha256'),
                            'can_label_contracts': symbol != 'SPY'}
    return core.Market(bars, daily, calendar, sessions, actions, complete, manifest)


def label(market, symbol, at):
    if symbol == 'SPY' or symbol not in SYMBOLS:
        raise ValueError('Reference-only or unregistered auxiliary instrument')
    sys.path.insert(0, str(ROOT / 'research/soxl_soxs_14d5pct_v3'))
    from risk import risk_label
    return risk_label(market, symbol, at,
                      {'id': 'fixed6', 'stop_fraction': .06, 'breakeven_trigger': None})
