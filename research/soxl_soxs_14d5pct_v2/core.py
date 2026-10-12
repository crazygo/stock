"""Causal arbitrary-time features and explicit 14-day limit/expiry labels.

Unadjusted execution prices are retained. A cumulative *past* split-share
factor converts prices to an economic unit for relative features and labels.
No order is routed to any brokerage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / '.cache/soxl_soxs_14d5pct_v2/data'
V1_INPUTS = ROOT / '.cache/soxl_soxs_14d5pct_v1/imports/2026-10-12/SOXL_SOXS_research/raw'
ETFS = ['SOXL', 'SOXS']
SYMBOLS = ETFS + ['SOXX', 'QQQ']
FEE = 0.0005
SLIP = 0.0005
PRE_SLIP = 0.001
ET = 'America/New_York'


def timestamp(value):
    result = pd.Timestamp(value)
    if result.tz is None:
        raise ValueError('decision_at must include its timezone')
    return result.tz_convert('UTC')


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


@dataclass
class Market:
    bars: dict
    daily: dict
    calendar: dict
    sessions: pd.DatetimeIndex
    actions: dict
    day_complete: dict
    manifest: dict


def load_market(data_root=DATA_ROOT):
    data_root = Path(data_root)
    cal = json.loads((data_root / 'calendar.json').read_text())
    official = {pd.Timestamp(x['session_date']): x for x in cal['sessions']}
    sessions = pd.DatetimeIndex(sorted(official))
    bars = {}; daily = {}; actions = {}; complete = {}; manifest = {}
    for symbol in SYMBOLS:
        path = data_root / 'bars' / f'{symbol}.parquet'
        f = pd.read_parquet(path)
        for col in ['start_at', 'end_at', 'available_at']:
            f[col] = pd.to_datetime(f[col], utc=True).dt.as_unit('ns')
        received = pd.to_datetime(f.get('received_at', pd.Series(pd.NaT, index=f.index)), utc=True, errors='coerce').dt.as_unit('ns')
        f['live_available_at'] = f.available_at.where(received <= f.available_at, received)
        if f.end_at.duplicated().any() or not f.end_at.is_monotonic_increasing:
            raise ValueError(f'{symbol}: duplicate or unsorted timestamps')
        if not f.price_basis.eq('NONE').all():
            raise ValueError(f'{symbol}: execution price basis must be NONE')
        f['date'] = pd.to_datetime(f.session_date)
        # Half-day bars after the official close are extended-session bars.
        regular_open = pd.to_datetime(f.date.map({d: x['open_at'] for d, x in official.items()}), utc=True)
        regular_close = pd.to_datetime(f.date.map({d: x['close_at'] for d, x in official.items()}), utc=True)
        inside = (f.start_at >= regular_open) & (f.end_at <= regular_close)
        after_close = (f.start_at >= regular_close) & (f.session_type != 'overnight')
        f.loc[inside, 'session_type'] = 'regular'
        f.loc[after_close, 'session_type'] = 'post_market'
        source = json.loads((V1_INPUTS / f'{symbol}_ibkr_daily.json').read_text())
        actions[symbol] = source.get('corp_actions', [])
        # Existing Futu corporate-action metadata independently checks split ratios.
        local_actions = ROOT / 'market_data/corporate_actions' / f'{symbol}.parquet'
        if local_actions.exists():
            table = pd.read_parquet(local_actions)
            for action in actions[symbol]:
                if action['type'] == 'Splits':
                    day = pd.Timestamp(action['date']).strftime('%Y-%m-%d')
                    rows = table[table.ex_div_date == day]
                    if len(rows) != 1 or not np.isclose(float(rows.split_ratio.iloc[0]), 1 / float(action['value'])):
                        raise ValueError(f'{symbol}: split sources disagree at {day}')
        factor = np.ones(len(f))
        for action in actions[symbol]:
            if action['type'] == 'Splits':
                factor[f.date >= pd.Timestamp(action['date'])] *= float(action['value'])
        f['share_factor'] = factor
        for price in ['open', 'high', 'low', 'close']:
            f[f'econ_{price}'] = f[price] * factor
        f['econ_volume'] = f.volume / factor
        f = f.set_index('end_at', drop=False)
        bars[symbol] = f
        reg = f[f.session_type == 'regular']
        d = reg.groupby('date').agg(open=('econ_open', 'first'), high=('econ_high', 'max'),
                                    low=('econ_low', 'min'), close=('econ_close', 'last'),
                                    volume=('econ_volume', 'sum'), raw_close=('close', 'last'),
                                    share_factor=('share_factor', 'last'), available_at=('available_at', 'max'),
                                    live_available_at=('live_available_at', 'max'))
        checks = {}
        for day, group in reg.groupby('date'):
            session = official.get(day)
            expected = pd.date_range(session['open_at'], session['close_at'], freq='5min', inclusive='right') if session else pd.DatetimeIndex([], tz='UTC')
            checks[day] = bool(session) and group.index.equals(expected)
        complete[symbol] = checks
        d['complete'] = [checks[x] for x in d.index]
        daily[symbol] = d
        manifest[symbol] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'rows': len(f), 'complete_days': sum(checks.values()),
                            'incomplete_days': [str(d.date()) for d, ok in checks.items() if not ok]}
    return Market(bars, daily, official, sessions, actions, complete, manifest)


def feature_row(market, decision_at, require_received=False):
    decision_at = timestamp(decision_at)
    date = decision_at.tz_convert(ET).tz_localize(None).normalize()
    minute = decision_at.tz_convert(ET).hour * 60 + decision_at.tz_convert(ET).minute
    result = {'minute_of_day': minute / 1440, 'is_premarket': float(minute < 570),
              'is_after_regular': float(minute >= 960)}
    cutoffs = {}; missing = []
    for symbol in SYMBOLS:
        completed = market.daily[symbol]
        # Never use an incomplete current day or data available after analysis.
        availability = 'live_available_at' if require_received else 'available_at'
        expected_days = [day for day in market.sessions
                         if pd.Timestamp(market.calendar[day]['close_at']) + pd.Timedelta(seconds=1) <= decision_at]
        expected_days = pd.DatetimeIndex(expected_days[-21:])
        history = completed.reindex(expected_days)
        if len(history) < 21 or not history.complete.fillna(False).all() or not (history[availability] <= decision_at).all():
            missing.append(f'{symbol}:daily_lookback_incomplete'); continue
        closes = history.close.iloc[-21:]
        changes = closes.pct_change(fill_method=None).iloc[1:]
        for lag in [1, 5, 20]:
            result[f'{symbol}_ret{lag}'] = float(closes.iloc[-1] / closes.iloc[-1-lag] - 1)
        for lag in [10, 20]:
            result[f'{symbol}_vol{lag}'] = float(changes.iloc[-lag:].std())
        result[f'{symbol}_dd20'] = float(closes.iloc[-1] / closes.iloc[-20:].max() - 1)
        h = history.iloc[-1]
        result[f'{symbol}_range'] = float((h.high - h.low) / h.close)
        result[f'{symbol}_relative_volume'] = float(h.volume / history.volume.iloc[-20:].mean())
        f = market.bars[symbol]
        # Available timestamps are monotonic in the captured historical contract.
        if require_received:
            prefix = f.loc[:decision_at]
            available = prefix[prefix.live_available_at <= decision_at]
        else:
            available = f.iloc[:f.available_at.searchsorted(decision_at, side='right')]
        effective_date = date + pd.Timedelta(days=1) if minute > 1200 else date
        today = available[available.date == effective_date]
        if today.empty:
            missing.append(f'{symbol}:no_current_session_bar'); continue
        newest = today.iloc[-1]
        age = (decision_at - newest.end_at).total_seconds()
        if age > 360:
            missing.append(f'{symbol}:bar_stale_{int(age)}s'); continue
        result[f'{symbol}_today_return'] = float(newest.econ_close / h.close - 1)
        result[f'{symbol}_today_range'] = float((today.econ_high.max() - today.econ_low.min()) / h.close)
        result[f'{symbol}_current_volume_relative'] = float(today.econ_volume.sum() / history.volume.iloc[-20:].mean())
        trailing = available[available.end_at > decision_at - pd.Timedelta(minutes=60)]
        result[f'{symbol}_ret60m'] = float(newest.econ_close / trailing.econ_open.iloc[0] - 1)
        result[f'{symbol}_trailing_bars'] = float(len(trailing))
        cutoffs[symbol] = str(newest.end_at)
    if missing:
        return None, {'decision_at': str(decision_at), 'status': 'unavailable', 'reasons': missing, 'data_cutoffs': cutoffs}
    result['sector_relative_5'] = result['SOXX_ret5'] - result['QQQ_ret5']
    result['sector_relative_20'] = result['SOXX_ret20'] - result['QQQ_ret20']
    result['sector_relative_today'] = result['SOXX_today_return'] - result['QQQ_today_return']
    return result, {'decision_at': str(decision_at), 'status': 'ready', 'data_cutoffs': cutoffs,
                    'availability_basis': 'actual_observed_receipt_and_completed_bar' if require_received else 'historical_bar_end_plus_1s; received_at stored separately'}


def expiry_day(market, entry_date):
    target = pd.Timestamp(entry_date) + pd.Timedelta(days=13)
    if target > market.sessions[-1]:
        return None
    eligible = market.sessions[market.sessions <= target]
    return eligible[-1] if len(eligible) else None


def label_entry(market, symbol, decision_at, cost_multiple=1.0):
    """Hypothetical order outcome; all missing/immature cases remain explicit."""
    decision_at = timestamp(decision_at)
    date = decision_at.tz_convert(ET).tz_localize(None).normalize()
    f = market.bars[symbol]
    local_day = f.loc[decision_at:decision_at + pd.Timedelta(hours=12)]
    eligible = local_day[(local_day.start_at > decision_at + pd.Timedelta(seconds=30))
                         & (local_day.date == date) & local_day.session_type.isin(['pre_market', 'regular'])]
    if eligible.empty:
        return {'symbol': symbol, 'decision_at': decision_at, 'reason': 'no_entry_bar', 'complete_window': False}
    first = eligible.iloc[0]; start = first.start_at
    minute = start.tz_convert(ET).hour * 60 + start.tz_convert(ET).minute
    session = market.calendar.get(date)
    if minute > 930 or not session or start >= pd.Timestamp(session['close_at']):
        return {'symbol': symbol, 'decision_at': decision_at, 'reason': 'outside_entry_window', 'complete_window': False}
    if (start - decision_at).total_seconds() > 360:
        return {'symbol': symbol, 'decision_at': decision_at, 'reason': 'entry_bar_missing', 'complete_window': False}
    slip = (PRE_SLIP if first.session_type == 'pre_market' else SLIP) * cost_multiple
    fee = FEE * cost_multiple
    entry = float(first.open) * (1 + slip)
    raw_limit = math.ceil(entry * 1.05 * 100 - 1e-10) / 100
    economic_entry = entry * float(first.share_factor)
    economic_limit = raw_limit * float(first.share_factor)
    expiry = expiry_day(market, date)
    if expiry is None:
        return {'symbol': symbol, 'decision_at': decision_at, 'reason': 'pending_calendar', 'complete_window': False}
    close_at = pd.Timestamp(market.calendar[expiry]['close_at'])
    window = f.loc[start:close_at]
    path = window[(window.start_at >= start) & (window.session_type == 'regular')]
    if path.empty:
        return {'symbol': symbol, 'decision_at': decision_at, 'entry_at': start, 'reason': 'missing_prefix', 'complete_window': False}
    # Adjust the nominal order on a split and round its price to the then-current cent tick.
    limits = np.ceil((economic_limit / path.share_factor.to_numpy()) * 100 - 1e-10) / 100
    hitmask = path.high.to_numpy() >= limits
    first_hit = int(np.flatnonzero(hitmask)[0]) if hitmask.any() else None
    event_day = path.iloc[first_hit].date if first_hit is not None else expiry
    end_row = path.iloc[first_hit] if first_hit is not None else None
    last_observed = f[f.session_type == 'regular'].date.max()
    full_window = expiry <= last_observed and all(
        market.day_complete[symbol].get(d, False) for d in market.sessions[(market.sessions >= date) & (market.sessions <= expiry)])
    prefix_end = min(event_day, last_observed)
    prefix_complete = all(market.day_complete[symbol].get(d, False)
                          for d in market.sessions[(market.sessions >= date) & (market.sessions <= prefix_end)])
    reason = 'target' if first_hit is not None else ('expiry' if full_window else 'pending')
    if not prefix_complete:
        reason = 'missing_prefix'; end_row = None
    if reason == 'target':
        # Timestamp is a bar interval bound, not an invented exact fill time.
        exit_at = end_row.end_at; exit_start = end_row.start_at
        exit_economic = float(limits[first_hit] * end_row.share_factor)
    elif reason == 'expiry':
        last = path.iloc[-1]
        if last.end_at != close_at:
            reason = 'missing_prefix'; exit_at = None; exit_start = None; exit_economic = np.nan
        else:
            exit_at = close_at; exit_start = last.start_at
            exit_economic = float(last.econ_close) * (1 - SLIP * cost_multiple)
    else:
        exit_at = None; exit_start = None; exit_economic = np.nan
    dividends = []
    dividend_per_initial_unit = 0.0
    dividend_end = (exit_at.tz_convert(ET).tz_localize(None).normalize()
                    if exit_at is not None else min(expiry, last_observed))
    for action in market.actions[symbol]:
            ex = pd.Timestamp(action['date'])
            if action['type'] != 'CashDividends' or not (date < ex <= dividend_end):
                continue
            # A constant economic unit holds share_factor(ex) raw shares.
            after = f[f.date >= ex]
            factor = float(after.share_factor.iloc[0]) if len(after) else float(first.share_factor)
            amount = float(action['value']) * factor
            dividend_per_initial_unit += amount
            dividends.append({'ex': ex, 'pay': pd.Timestamp(action['payment_date']), 'economic_amount': amount})
    net = ((exit_economic * (1 - fee) + dividend_per_initial_unit) / (economic_entry * (1 + fee)) - 1
           if exit_at is not None else np.nan)
    return {'symbol': symbol, 'decision_at': decision_at, 'entry_date': date,
            'entry_at': start, 'entry_bar_end': first.end_at, 'entry_fill_raw': entry,
            'entry_fill_economic': economic_entry, 'limit_raw': raw_limit,
            'limit_economic': economic_limit, 'entry_share_factor': float(first.share_factor),
            'expiry_date': expiry, 'expiry_at': close_at, 'exit_at': exit_at,
            'exit_bar_start': exit_start, 'exit_fill_economic': exit_economic,
            'reason': reason, 'target_hit': int(reason == 'target') if exit_at is not None else None,
            'net_return': net, 'complete_window': full_window,
            'prefix_complete': prefix_complete, 'fee_each_side': fee,
            'dividend_per_economic_unit': dividend_per_initial_unit, 'dividends': dividends,
            'attached_order': {'side': 'SELL', 'type': 'LIMIT', 'tif': 'GTC',
                               'limit_price': raw_limit, 'active_session': 'regular',
                               'created_at': start, 'broker_submission': False}}


def candidate_table(market, clocks):
    features = []; metadata = []; labels = []
    available_dates = market.daily['SOXL'].index
    for date in available_dates:
        for clock in clocks:
            decision = pd.Timestamp(str(date.date()) + ' ' + clock, tz=ET).tz_convert('UTC')
            row, audit = feature_row(market, decision)
            metadata.append(audit)
            if row is None:
                continue
            features.append({'decision_at': decision, 'date': date, 'clock': clock, **row})
            for symbol in ETFS:
                labels.append(label_entry(market, symbol, decision))
    if not features:
        raise ValueError('No causal feature rows are available; inspect candidate/data coverage before training')
    X = pd.DataFrame(features).set_index('decision_at').sort_index()
    return X, labels, metadata
