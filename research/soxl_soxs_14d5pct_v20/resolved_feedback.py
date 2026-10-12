"""Consume paired terminal feedback without skipping an earlier pending date."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v18'))
from adaptive_blend import (core, bank, expert_bank, forecast, risk_decision,
                            day_cutoff, restrict_market, development_gate, STATE_COLUMNS)
from verify_continuation import key

ROUTES = {
    'terminal_frontier_hard': ('frontier', 'state', 'hard'),
    'terminal_available_hard': ('available', 'state', 'hard'),
    'terminal_frontier_global': ('frontier', 'global', 'hard'),
    'terminal_frontier_blend': ('frontier', 'state', 'blend'),
    'full_window_hard': ('full', 'state', 'hard'),
    'calibrated_fixed': ('full', 'state', 'calibrated'),
    'raw_fixed': ('full', 'state', 'raw'),
}


def prefix_receipt(market, label):
    """Validate only data through the terminal bar, including every scheduled bar.

    Historical available_at is a simulation assumption, not a live receipt.
    Dividend-bearing outcomes conservatively retain the original window wait.
    """
    out = {'known': False, 'release': None, 'terminal_available_at': None,
           'reason': 'unresolved', 'bars': 0, 'dividend_fallback': False}
    if (label.get('reason') not in ['target', 'risk_stop', 'expiry']
        or label.get('exit_at') is None or label.get('entry_at') is None
        or label.get('net_return') is None or not np.isfinite(label['net_return'])):
        return out
    start = pd.Timestamp(label['entry_at']); end = pd.Timestamp(label['exit_at'])
    start_day = start.tz_convert(core.ET).tz_localize(None).normalize()
    end_day = end.tz_convert(core.ET).tz_localize(None).normalize()
    expected = pd.DatetimeIndex([], tz='UTC')
    for date in market.sessions[(market.sessions >= start_day) & (market.sessions <= end_day)]:
        session = market.calendar[date]
        left = max(start, pd.Timestamp(session['open_at']))
        right = min(end, pd.Timestamp(session['close_at']))
        expected = expected.append(pd.date_range(left, right, freq='5min', inclusive='right'))
    f = market.bars[label['symbol']]
    path = f[(f.start_at >= start) & (f.end_at <= end) & f.session_type.eq('regular')]
    out['bars'] = len(path)
    if (not len(expected) or expected[-1] != end or not path.index.equals(expected)
        or not path.start_at.eq(path.end_at - pd.Timedelta(minutes=5)).all()
        or path.available_at.isna().any() or not path.available_at.ge(path.end_at).all()):
        out['reason'] = 'incomplete_terminal_prefix'; return out
    available = path.available_at.max()
    out.update(terminal_available_at=available, known=True, release=available, reason='terminal_prefix_observed')
    if label.get('dividends'):
        release = label.get('label_available_at') if label.get('complete_window') else None
        out.update(dividend_fallback=True, known=release is not None,
                   release=max(pd.Timestamp(release), available) if release is not None else None,
                   reason='dividend_retains_original_window')
    return out


def prepare_feedback(market, panel, events, labels):
    frame = expert_bank(panel, events); receipts = []
    frame['terminal_release'] = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns, UTC]')
    for at, row in frame.iterrows():
        pair = {s: prefix_receipt(market, labels[key(at, s)]) for s in core.ETFS}
        release = max(v['release'] for v in pair.values()) if all(v['known'] for v in pair.values()) else None
        if release is not None:
            frame.loc[at, 'terminal_release'] = release
        receipts.append({'at': at, 'date': row.date, 'terminal_release': release,
                         'original_full_release': row.release if pd.notna(row.release) else None, 'sides': pair})
    return frame, receipts


def feedback_pool(frame, cutoff, mode):
    day = cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    prior = frame[(frame.index < cutoff) & (frame.date < day)]
    columns = [*STATE_COLUMNS, 'raw_C', 'raw_D', 'cal_C', 'cal_D', 'actual_C', 'actual_D']
    finite = np.isfinite(prior[columns]).all(axis=1)
    terminal = prior.terminal_release.notna() & prior.terminal_release.lt(cutoff) & finite
    full = prior.release.notna() & prior.release.lt(cutoff) & finite
    info = {'mode': mode, 'prior_dates': len(prior), 'terminal_known_dates': int(terminal.sum()),
        'full_window_known_dates': int(full.sum()), 'first_pending_date': None,
        'known_but_waiting_behind_pending': 0, 'expired_unresolved_prefix': False}
    if mode == 'full':
        pool = prior[full].tail(120)
    elif mode == 'available':
        pool = prior[terminal].tail(120)
    elif mode == 'frontier':
        gaps = np.flatnonzero(~terminal.to_numpy()); stop = int(gaps[0]) if len(gaps) else len(prior)
        pool = prior.iloc[:stop].tail(120)
        if stop < len(prior):
            blocked = prior.iloc[stop].date
            info.update(first_pending_date=blocked, known_but_waiting_behind_pending=int(terminal.iloc[stop+1:].sum()),
                        expired_unresolved_prefix=bool(blocked + pd.Timedelta(days=14) <= day))
    else:
        raise ValueError(mode)
    info.update(selected_dates=len(pool), first_selected=pool.date.min() if len(pool) else None,
                last_selected=pool.date.max() if len(pool) else None)
    return pool, info


def feedback_forecast(frame, cutoff, at, event, pool_mode, weight_mode, method):
    pool, receipt = feedback_pool(frame, cutoff, pool_mode)
    current = frame.loc[at] if event['ready'] else None
    if not event['ready']:
        pool = frame.iloc[:0]
    model = forecast(pool, current, event, weight_mode, method)
    if pool_mode == 'frontier' and receipt['expired_unresolved_prefix']:
        model.update(ready=False, adjusted=None)
    model['feedback'] = receipt
    return model
