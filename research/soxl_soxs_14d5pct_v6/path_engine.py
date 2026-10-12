"""Observed intraday context and delayed, causal protective exits."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

V4 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v4'
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))
from walkforward import core, day_cutoff, digest

RULES = ['range_breakout', 'vwap_momentum', 'pullback_resume', 'range_failure', 'ridge_log', 'boosted_log']
EXITS = ['flat_close', 'trail_close', 'adaptive_overnight']
EXPERTS = [r + '__' + e for e in EXITS for r in RULES]


def build_context(market):
    parts = []
    for symbol, prefix in [('SOXX', 'x'), ('QQQ', 'q')]:
        rows = []
        for day, f in market.bars[symbol][market.bars[symbol].session_type == 'regular'].groupby('date'):
            if len(f) < 6:
                continue
            close = f.econ_close; opening = float(f.econ_open.iloc[0])
            volume = f.econ_volume.cumsum()
            vwap = (((f.econ_high + f.econ_low + close) / 3) * f.econ_volume).cumsum() / volume.replace(0, np.nan)
            high = f.econ_high.cummax(); low = f.econ_low.cummin()
            initial_high = float(f.econ_high.iloc[:6].max()); initial_low = float(f.econ_low.iloc[:6].min())
            changes = close.diff(); changes.iloc[0] = close.iloc[0] - opening
            expected = pd.date_range(market.calendar[day]['open_at'], market.calendar[day]['close_at'], freq='5min', inclusive='right')
            ordinals = expected.get_indexer(f.index)
            prefix_complete = np.logical_and.accumulate(ordinals == np.arange(len(f)))
            frame = pd.DataFrame({
                prefix + '_session_return': close / opening - 1,
                prefix + '_vwap_distance': close / vwap - 1,
                prefix + '_vwap_change15': vwap / vwap.shift(3) - 1,
                prefix + '_range_high_distance': close / initial_high - 1,
                prefix + '_range_low_distance': close / initial_low - 1,
                prefix + '_broke_high': (high > initial_high).astype(float),
                prefix + '_broke_low': (low < initial_low).astype(float),
                prefix + '_recent15': close / f.econ_open.shift(2) - 1,
                prefix + '_previous15': close.shift(3) / f.econ_open.shift(5) - 1,
                prefix + '_range_position': (close - low) / (high - low).replace(0, np.nan),
                prefix + '_efficiency': (close - opening) / changes.abs().cumsum().replace(0, np.nan),
                prefix + '_available_at': f.available_at,
                prefix + '_prefix_complete': prefix_complete,
            }, index=f.index)
            # Flat synthetic/zero-range observations have a defined neutral state.
            frame[prefix + '_range_position'] = frame[prefix + '_range_position'].fillna(.5)
            frame[prefix + '_efficiency'] = frame[prefix + '_efficiency'].fillna(0.)
            # Opening-range statistics cannot be inspected before six bars exist.
            rows.append(frame.iloc[5:])
        parts.append(pd.concat(rows).sort_index())
    result = parts[0].join(parts[1], how='inner')
    result['available_at'] = result[['x_available_at', 'q_available_at']].max(axis=1)
    result['prefix_complete'] = result.x_prefix_complete & result.q_prefix_complete
    return result


def feature_at(market, context, at):
    row, audit = core.feature_row(market, at)
    if row is None:
        return None, audit
    if not all(np.isfinite(value) for value in row.values()):
        return None, {'reason': 'base_context_not_finite', 'audit': audit}
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    if at < pd.Timestamp(market.calendar[date]['open_at']):
        return {**row, 'regular_context_ready': 0.}, audit
    available = context[(context.available_at <= at) & context.prefix_complete]
    if available.empty:
        return None, {'reason': 'regular_prefix_unavailable'}
    latest = available.iloc[-1]
    if latest.name.tz_convert(core.ET).date() != date.date() or (at - latest.name).total_seconds() > 360:
        return None, {'reason': 'regular_prefix_stale'}
    fields = {c: float(latest[c]) for c in context if c.startswith(('x_', 'q_')) and not c.endswith(('available_at', 'prefix_complete'))}
    if not all(np.isfinite(x) for x in fields.values()):
        return None, {'reason': 'regular_context_not_finite'}
    return {**row, 'regular_context_ready': 1., **fields}, audit


def rule_action(row, name):
    if not row.get('regular_context_ready', 0):
        return 'CASH'
    for sign, symbol in [(1, 'SOXL'), (-1, 'SOXS')]:
        if name == 'range_breakout':
            edge = row.x_range_high_distance if sign == 1 else row.x_range_low_distance
            good = sign * edge > 0 and sign * row.x_vwap_distance > 0 and sign * row.q_session_return > 0
        elif name == 'vwap_momentum':
            good = all(sign * row[c] > 0 for c in ['x_session_return', 'x_vwap_distance', 'x_recent15', 'q_session_return'])
        elif name == 'pullback_resume':
            good = all(sign * row[c] > 0 for c in ['x_session_return', 'x_vwap_distance', 'x_recent15', 'q_session_return']) and sign * row.x_previous15 < 0
        elif name == 'range_failure':
            crossed = row.x_broke_low if sign == 1 else row.x_broke_high
            edge = row.x_range_low_distance if sign == 1 else row.x_range_high_distance
            good = crossed > 0 and sign * edge > 0 and sign * row.x_recent15 > 0 and sign * row.q_recent15 > 0
        else:
            raise ValueError(name)
        if good:
            return symbol
    return 'CASH'


def prefix_complete(market, symbol, start, end):
    pieces = []
    first_day = start.tz_convert(core.ET).tz_localize(None).normalize()
    last_day = end.tz_convert(core.ET).tz_localize(None).normalize()
    days = market.sessions[(market.sessions >= first_day) & (market.sessions <= last_day)]
    for day in days:
        session = market.calendar[day]
        opening, closing = pd.Timestamp(session['open_at']), pd.Timestamp(session['close_at'])
        if closing < start or opening >= end:
            continue
        pieces.append(pd.date_range(max(opening, start), min(closing, end), freq='5min', inclusive='right'))
    expected = pieces[0].append(pieces[1:]) if pieces else pd.DatetimeIndex([], tz='UTC')
    f = market.bars[symbol].loc[start:end]
    actual = f[(f.start_at >= start) & (f.end_at <= end) & f.session_type.eq('regular')].index
    return len(expected) > 0 and actual.equals(expected)


def exit_label(market, context, symbol, at, policy, multiple=1):
    base = core.label_entry(market, symbol, at, multiple)
    if 'entry_fill_economic' not in base:
        return {**base, 'exit_policy': policy, 'label_available_at': None}
    fee = core.FEE * multiple; slip = core.SLIP * multiple
    entry = base['entry_fill_economic']; target = base['limit_economic']
    session_close = pd.Timestamp(market.calendar[base['entry_date']]['close_at'])
    horizon = base['expiry_at'] if policy == 'adaptive_overnight' else session_close
    f = market.bars[symbol]
    window = f.loc[base['entry_at']:horizon]
    path = window[(window.start_at >= base['entry_at']) & window.session_type.eq('regular')]
    stop_fraction = .03 if policy == 'adaptive_overnight' else .015
    stop = entry * (1 - stop_fraction)
    pending_stops = []; pending_exit = None; raised = False; peak = entry; event = None
    for bar in path.itertuples():
        factor = float(bar.share_factor)
        raw_target = math.ceil(target / factor * 100 - 1e-10) / 100
        if pending_exit is not None and bar.start_at > pending_exit:
            event = (bar.start_at, bar.start_at, min(float(bar.open), raw_target) * factor * (1 - slip), 'trend_exit')
            break
        ready_stops = [x for x in pending_stops if bar.start_at > x[1]]
        if ready_stops:
            stop = max(stop, max(x[0] for x in ready_stops)); raised = True
            pending_stops = [x for x in pending_stops if bar.start_at <= x[1]]
        raw_stop = math.floor(stop / factor * 100 + 1e-10) / 100
        if bar.low <= raw_stop:
            event = (bar.end_at, bar.start_at, min(float(bar.open), raw_stop) * factor * (1 - slip), 'trailing_stop' if raised else 'risk_stop')
            break
        if bar.high >= raw_target:
            event = (bar.end_at, bar.start_at, raw_target * factor, 'target')
            break
        peak = max(peak, float(bar.econ_high))
        if policy != 'flat_close' and peak >= entry * 1.02:
            flat = math.ceil((entry * (1 + fee) / ((1 - fee) * (1 - slip))) / factor * 100 - 1e-10) / 100 * factor
            level = max(flat, peak * .985)
            if level > max([stop] + [x[0] for x in pending_stops]):
                pending_stops.append((level, bar.available_at + pd.Timedelta(seconds=30)))
        if policy == 'adaptive_overnight' and bar.end_at in context.index and pending_exit is None:
            state = context.loc[bar.end_at]
            sign = 1 if symbol == 'SOXL' else -1
            if state.prefix_complete and all(sign * state[c] < 0 for c in ['x_session_return', 'x_vwap_distance', 'q_session_return']):
                pending_exit = max(bar.available_at, state.available_at) + pd.Timedelta(seconds=30)
    if event is None and len(path) and path.end_at.iloc[-1] == horizon:
        last = path.iloc[-1]
        event = (horizon, last.start_at, float(last.econ_close) * (1 - slip), 'expiry' if policy == 'adaptive_overnight' else 'session_close')
    out = {**base, 'exit_policy': policy, 'stop_fraction': stop_fraction, 'label_available_at': None,
           'label_window_end': horizon, 'complete_window': False, 'reason': 'pending',
           'exit_at': None, 'exit_bar_start': None, 'exit_fill_economic': np.nan, 'net_return': np.nan, 'target_hit': None}
    out['attached_order'] = {**base['attached_order'], 'protective_stop_fraction': stop_fraction,
         'exit_policy': policy, 'planned_early_close': session_close if policy != 'adaptive_overnight' else None,
         'oco_cancel_other_exit_after_fill': True}
    if event is not None:
        # Open executions need the complete price prefix up to that Open, not its later High/Low.
        observed_end = event[0] if event[3] != 'trend_exit' else event[0]
        required_symbols = [symbol, 'SOXX', 'QQQ'] if policy == 'adaptive_overnight' else [symbol]
        if not all(prefix_complete(market, s, base['entry_at'], observed_end) for s in required_symbols):
            out['reason'] = 'missing_prefix'
            return out
        exit_day = event[0].tz_convert(core.ET).tz_localize(None).normalize()
        dividends = [d for d in base['dividends'] if d['ex'] <= exit_day]
        dividend = sum(d['economic_amount'] for d in dividends)
        net = (event[2] * (1 - fee) + dividend) / (entry * (1 + fee)) - 1
        out.update({'exit_at': event[0], 'exit_bar_start': event[1], 'exit_fill_economic': event[2],
                    'reason': event[3], 'target_hit': int(event[3] == 'target'), 'net_return': net,
                    'prefix_complete': True, 'dividends': dividends, 'dividend_per_economic_unit': dividend})
    full_start = pd.Timestamp(market.calendar[base['entry_date']]['open_at'])
    complete = prefix_complete(market, symbol, full_start, horizon)
    if complete and event is not None:
        # Adaptive exit decisions require the sector/QQQ prefix through their event.
        if policy == 'adaptive_overnight' and not all(prefix_complete(market, s, full_start, horizon) for s in ['SOXX', 'QQQ']):
            return out
        out['complete_window'] = True
        seen = f.loc[full_start:horizon].available_at.max()
        if policy == 'adaptive_overnight':
            seen = max(seen, context.loc[base['entry_at']:horizon].available_at.max())
        out['label_available_at'] = max(horizon + pd.Timedelta(seconds=1), seen)
    return out
