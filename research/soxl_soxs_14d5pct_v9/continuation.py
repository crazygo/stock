"""Causal, daily entry and remaining-contract valuation. No broker integration."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v7'))
from breadth import core
from path_engine import day_cutoff
from walkforward import make_labels

RULE = {'id': 'fixed_stop6', 'stop_fraction': .06, 'breakeven_trigger': None}
OWN = ['ret1', 'ret5', 'ret20', 'vol20', 'today_return', 'today_range',
       'current_volume_relative', 'relative_volume']
SIGNED = ['SOXX_ret1', 'SOXX_ret5', 'SOXX_ret20', 'SOXX_today_return',
          'QQQ_ret1', 'QQQ_ret5', 'QQQ_ret20', 'QQQ_today_return',
          'sector_relative_5', 'sector_relative_20', 'sector_relative_today',
          'breadth_ret5_mean', 'breadth_today_mean']
UNSIGNED = ['SOXX_vol20', 'breadth_today_std', 'breadth_vol20_mean']
ENTRY_COLUMNS = ['direction'] + ['own_' + c for c in OWN] + ['signed_' + c for c in SIGNED] + UNSIGNED
POSITION_COLUMNS = ['age_calendar', 'remaining_calendar', 'mark_return', 'target_distance',
                    'stop_distance', 'observed_peak_drawdown']
MANAGER_COLUMNS = ENTRY_COLUMNS + POSITION_COLUMNS


def entry_state(row, symbol):
    sign = 1. if symbol == 'SOXL' else -1.
    state = {'direction': sign, **{'own_' + c: float(row[symbol + '_' + c]) for c in OWN},
             **{'signed_' + c: sign * float(row[c]) for c in SIGNED},
             **{c: float(row[c]) for c in UNSIGNED}}
    if not np.isfinite(list(state.values())).all():
        raise ValueError('Missing causal entry features')
    return state


def label_key(at, symbol):
    return pd.Timestamp(at).isoformat() + '|' + symbol


def baseline_labels(market, features, multiple=1):
    labels = make_labels(market, features, RULE, multiple)
    for label in labels:
        if 'entry_fill_economic' in label:
            # The protection exists even if no stop occurs in the observed path.
            label['attached_order'] = {**label['attached_order'], 'protective_stop_fraction': .06,
                'breakeven_trigger': None, 'oco_cancel_other_exit_after_fill': True}
    return {label_key(x['decision_at'], x['symbol']): x for x in labels}


def joint_releases(features, baseline):
    releases = {}
    for at, row in features.iterrows():
        pair = [baseline[label_key(at, s)] for s in core.ETFS]
        if all(x.get('complete_window') and x.get('label_available_at') is not None
               and np.isfinite(x.get('net_return', np.nan)) for x in pair):
            releases[row.date] = max(x['label_available_at'] for x in pair)
    return releases


def eligible_dates(releases, cutoff, protocol):
    return [d for d, release in sorted(releases.items())
            if d.tz_localize(core.ET) < cutoff and release < cutoff][-protocol['training_window_dates']:]


def alive(label, at):
    """Only past exits affect survival; a future recorded outcome does not affect features."""
    return ('entry_fill_economic' in label and label['entry_at'] < at < label['expiry_at']
            and (label.get('exit_at') is None or label['exit_at'] > at))


def review_state(market, row, label, at):
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    if date <= label['entry_date']:
        return None, {'reason': 'entry_day_has_no_review'}
    bars = market.bars[label['symbol']]
    past = bars[(bars.available_at <= at) & (bars.end_at <= at)]
    if past.empty:
        return None, {'reason': 'no_observed_mark'}
    quote = past.iloc[-1]
    if quote.date != date or (at - quote.end_at).total_seconds() > 360:
        return None, {'reason': 'stale_observed_mark'}
    regular = past[(past.session_type == 'regular') & (past.start_at >= label['entry_at'])]
    needed = market.sessions[(market.sessions >= label['entry_date']) & (market.sessions < date)]
    # Only completed earlier sessions, never the review day's future completeness.
    if not all(market.day_complete[label['symbol']].get(d, False) for d in needed) or regular.empty:
        return None, {'reason': 'unknown_holding_prefix'}
    mark = float(quote.econ_close); entry = label['entry_fill_economic']
    peak = max(entry, float(regular.econ_high.max()))
    age = (date - label['entry_date']).days
    state = {**entry_state(row, label['symbol']), 'age_calendar': float(age),
             'remaining_calendar': float(13 - age), 'mark_return': mark / entry - 1,
             'target_distance': label['limit_economic'] / mark - 1,
             'stop_distance': entry * .94 / mark - 1, 'observed_peak_drawdown': mark / peak - 1}
    assert np.isfinite(list(state.values())).all()
    return state, {'mark_bar_end': quote.end_at, 'mark_available_at': quote.available_at,
                   'regular_peak_last_bar': regular.end_at.iloc[-1], 'observed_peak': peak}


def immediate_sale(market, label, at, multiple=1):
    """Future execution label only. Never pass its prices into review_state/model inputs."""
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    bars = market.bars[label['symbol']]
    path = bars[(bars.date == date) & (bars.session_type == 'regular')
                & (bars.start_at > at + pd.Timedelta(seconds=30))]
    if path.empty:
        return None
    first = path.iloc[0]
    if (first.start_at - at).total_seconds() > 360:
        return None
    factor = float(first.share_factor)
    target = math.ceil(label['limit_economic'] / factor * 100 - 1e-10) / 100
    fill = min(float(first.open), target) * factor * (1 - core.SLIP * multiple)
    return {'exit_at': first.start_at, 'exit_bar_start': first.start_at,
            'exit_fill_economic': fill, 'sale_net': fill * (1 - core.FEE * multiple)}


def continuation_target(label, sale, review_date):
    if sale is None or label.get('exit_at') is None or not np.isfinite(label.get('net_return', np.nan)):
        return np.nan
    remaining_div = sum(d['economic_amount'] for d in label['dividends'] if d['ex'] > review_date)
    hold = label['exit_fill_economic'] * (1 - label['fee_each_side']) + remaining_div
    return float(np.log(hold / sale['sale_net']))


def managed_exit(label, sale, at):
    """Attach a previously issued intent's realized Open exit; preserve full-window maturity."""
    if sale is None:
        return label
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    divs = [d for d in label['dividends'] if d['ex'] <= date]
    dividend = sum(d['economic_amount'] for d in divs)
    net = (sale['sale_net'] + dividend) / (label['entry_fill_economic'] * (1 + label['fee_each_side'])) - 1
    return {**label, **{k: v for k, v in sale.items() if k != 'sale_net'},
            'manager_decision_at': at, 'reason': 'continuation_exit', 'target_hit': 0,
            'net_return': net, 'prefix_complete': True, 'dividends': divs,
            'dividend_per_economic_unit': dividend, 'same_bar_both_barriers': False,
            'gap_below_stop': False, 'protection_raised': False}


def review_table(market, features, baseline, multiple=1):
    records = []
    for key, label in baseline.items():
        if 'entry_fill_economic' not in label:
            continue
        for at, row in features[(features.date > label['entry_date']) &
                                (features.date <= label['expiry_date'])].iterrows():
            if not alive(label, at):
                continue
            state, audit = review_state(market, row, label, at)
            sale = immediate_sale(market, label, at, multiple)
            records.append({'id': key + '@' + at.isoformat(), 'key': key, 'at': at,
                'entry_date': label['entry_date'], 'symbol': label['symbol'], 'state': state,
                'audit': audit, 'sale': sale, 'target': continuation_target(label, sale, row.date)})
    return records


def fit(rows, columns, protocol):
    dates = sorted({x['entry_date'] for x in rows})
    info = {'dates': dates, 'members': [x['id'] for x in rows], 'rows': len(rows),
            'ready': len(dates) >= protocol['minimum_training_dates']}
    if not info['ready']:
        return None, info
    X = np.array([[r['state'][c] for c in columns] for r in rows])
    y = np.array([r['target'] for r in rows])
    counts = pd.Series([r['entry_date'] for r in rows]).value_counts()
    weights = np.array([1 / counts[r['entry_date']] for r in rows])
    assert np.isfinite(X).all() and np.isfinite(y).all() and np.isclose(weights.sum(), len(dates))
    model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
        learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=814)
    model.fit(X, y, sample_weight=weights)
    info['weight_sum'] = float(weights.sum())
    return model, info


def calibration(records, dates, protocol):
    """Saved raw forecasts, one mean residual per mature parent entry date."""
    selected = [r for r in records if r['entry_date'] in dates and np.isfinite(r['target'])]
    last = sorted({r['entry_date'] for r in selected})[-protocol['calibration_window_dates']:]
    selected = [r for r in selected if r['entry_date'] in last]
    groups = {}
    for r in selected:
        groups.setdefault(r['entry_date'], []).append(r['target'] - r['raw'])
    correction = float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
    return {'dates': last, 'members': [r['id'] for r in selected], 'correction': correction,
            'ready': len(last) >= protocol['minimum_calibration_dates']}


def predict(model, state, columns):
    return float(model.predict(np.array([[state[c] for c in columns]]))[0])


def choose_entry(raw, cal):
    if not raw:
        return {'action': 'CASH', 'ready': False, 'raw': {}, 'top_symbol': None, 'adjusted': None}
    top = max(core.ETFS, key=lambda s: raw[s])  # insertion order resolves exact ties as SOXL
    adjusted = raw[top] + cal['correction'] if cal['correction'] is not None else None
    return {'action': top if cal['ready'] and adjusted > 0 else 'CASH',
            'ready': cal['ready'], 'raw': raw, 'top_symbol': top, 'adjusted': adjusted}
