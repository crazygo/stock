"""Complete, as-of semiconductor panel features and event freshness."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd

V6 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v6'
sys.path.insert(0, str(V6))
from path_engine import core

PANEL = ['NVDA', 'AMD', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'INTC']
PANEL_ROOT = core.ROOT / '.cache/soxl_soxs_14d5pct_v7/data'


def load_panel(calendar, data_root=PANEL_ROOT):
    bars, daily, manifest = {}, {}, {}
    for symbol in PANEL:
        path = Path(data_root) / 'bars' / (symbol + '.parquet')
        f = pd.read_parquet(path)
        for c in ['start_at', 'end_at', 'available_at']:
            f[c] = pd.to_datetime(f[c], utc=True).dt.as_unit('ns')
        if not f.price_basis.eq('NONE').all() or f.end_at.duplicated().any() or not f.end_at.is_monotonic_increasing:
            raise ValueError(symbol + ': invalid price basis or timestamps')
        f['date'] = pd.to_datetime(f.session_date)
        opening = pd.to_datetime(f.date.map({d: x['open_at'] for d, x in calendar.items()}), utc=True)
        closing = pd.to_datetime(f.date.map({d: x['close_at'] for d, x in calendar.items()}), utc=True)
        f['regular'] = (f.start_at >= opening) & (f.end_at <= closing)
        action_path = Path(data_root) / 'corporate_actions' / (symbol + '.parquet')
        actions = pd.read_parquet(action_path)
        factor = np.ones(len(f))
        # Historical metadata is used only on/after the effective date. Later
        # actions cannot change earlier rows, unlike a current QFQ price level.
        for a in actions[actions.split_ratio.notna()].itertuples():
            if float(a.split_ratio) <= 0:
                raise ValueError(symbol + ': invalid split')
            if pd.Timestamp(a.ex_div_date) >= f.date.min():
                factor[f.date >= pd.Timestamp(a.ex_div_date)] *= 1 / float(a.split_ratio)
        for c in ['open', 'high', 'low', 'close']:
            f['econ_' + c] = f[c] * factor
        f['econ_volume'] = f.volume / factor
        f = f.set_index('end_at', drop=False)
        reg = f[f.regular]
        d = reg.groupby('date').agg(open=('econ_open', 'first'), high=('econ_high', 'max'),
             low=('econ_low', 'min'), close=('econ_close', 'last'), volume=('econ_volume', 'sum'),
             available_at=('available_at', 'max'))
        checks = {}
        for day, group in reg.groupby('date'):
            s = calendar[day]
            expected = pd.date_range(s['open_at'], s['close_at'], freq='5min', inclusive='right')
            checks[day] = group.index.equals(expected)
        d['complete'] = [checks[day] for day in d.index]
        bars[symbol], daily[symbol] = f, d
        manifest[symbol] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'actions_sha256': hashlib.sha256(action_path.read_bytes()).hexdigest(), 'rows': len(f),
            'complete_days': sum(checks.values()), 'incomplete_days': [str(d.date()) for d, ok in checks.items() if not ok]}
    return bars, daily, manifest


def freshness(frame, calendar):
    """Only closes already seen can establish an outside-range event."""
    rows = []
    for date, f in frame.groupby('date'):
        if len(f) < 6:
            continue
        expected = pd.date_range(calendar[date]['open_at'], calendar[date]['close_at'], freq='5min', inclusive='right')
        complete = np.logical_and.accumulate(expected.get_indexer(f.index) == np.arange(len(f)))
        h, l = f.econ_high.iloc[:6].max(), f.econ_low.iloc[:6].min()
        clock = pd.Series(np.arange(len(f)) * 5., index=f.index)
        out = pd.DataFrame(index=f.index)
        for side, condition in [('high', f.econ_close > h), ('low', f.econ_close < l)]:
            # First30m defines the range; it cannot simultaneously break it.
            condition.iloc[:6] = False
            last = clock.where(condition).ffill()
            age = clock - last
            out[side + '_ever'] = last.notna().astype(float)
            out[side + '_age'] = age.fillna(390.).clip(upper=390.) / 390.
            inside = f.econ_close <= h if side == 'high' else f.econ_close >= l
            out[side + '_fresh_failure'] = (last.notna() & (age > 0) & (age <= 15) & inside).astype(float)
        out['available_at'] = f.available_at.cummax()
        out['prefix_complete'] = complete
        rows.append(out.iloc[5:])
    return pd.concat(rows).sort_index()


def timing_at(timing, at):
    fields, audit = {}, {}
    day = at.tz_convert(core.ET).date()
    for symbol, f in timing.items():
        seen = f[(f.available_at <= at) & f.prefix_complete]
        if seen.empty or seen.index[-1].tz_convert(core.ET).date() != day or (at - seen.index[-1]).total_seconds() > 360:
            return None, {'reason': symbol + ':timing_unavailable'}
        row = seen.iloc[-1]
        fields.update({'timing_' + symbol + '_' + c: float(row[c]) for c in f if c not in ['available_at', 'prefix_complete']})
        audit[symbol] = str(row.available_at)
    return fields, audit


def panel_at(bars, daily, sessions, calendar, at):
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    past = sessions[sessions < date][-21:]
    before_open = at < pd.Timestamp(calendar[date]['open_at'])
    fields, audit, aggregate = {}, {}, {}
    for symbol in PANEL:
        history = daily[symbol].reindex(past)
        if len(history) != 21 or not history.complete.fillna(False).all() or not (history.available_at <= at).all():
            return None, {'reason': symbol + ':daily_history_incomplete'}
        f = bars[symbol]
        # Slice by time first, then availability: a delayed earlier bar does not
        # justify reading a later unavailable record using searchsorted.
        seen = f.loc[:at]; seen = seen[(seen.available_at <= at) & seen.date.eq(date)]
        if seen.empty or (at - seen.index[-1]).total_seconds() > 360:
            return None, {'reason': symbol + ':current_bar_stale_or_absent'}
        last = seen.iloc[-1]; closes = history.close
        row = {f'ret{lag}': float(closes.iloc[-1] / closes.iloc[-1-lag] - 1) for lag in [1, 5, 20]}
        row.update({'vol20': float(closes.pct_change(fill_method=None).iloc[1:].std()),
             'today': float(last.econ_close / closes.iloc[-1] - 1),
             'relative_volume': float(seen.econ_volume.sum() / history.volume.iloc[-20:].mean())})
        reg = seen[seen.regular]
        if not before_open:
            expected = pd.date_range(calendar[date]['open_at'], last.name, freq='5min', inclusive='right')
            if len(reg) < 6 or not reg.index.equals(expected):
                return None, {'reason': symbol + ':regular_prefix_incomplete'}
            close = reg.econ_close; opening = reg.econ_open.iloc[0]
            volume = reg.econ_volume.sum()
            if volume <= 0:
                return None, {'reason': symbol + ':zero_regular_volume'}
            vwap = (((reg.econ_high + reg.econ_low + close) / 3) * reg.econ_volume).sum() / volume
            changes = close.diff(); changes.iloc[0] = close.iloc[0] - opening
            row.update({'session_return': float(last.econ_close / opening - 1),
                 'vwap_distance': float(last.econ_close / vwap - 1),
                 'recent15': float(last.econ_close / reg.econ_open.iloc[-3] - 1),
                 'efficiency': float((last.econ_close - opening) / changes.abs().sum()) if changes.abs().sum() else 0.})
        else:
            row.update({c: 0. for c in ['session_return', 'vwap_distance', 'recent15', 'efficiency']})
        if not all(np.isfinite(v) for v in row.values()):
            return None, {'reason': symbol + ':nonfinite_panel_feature'}
        fields.update({'leader_' + symbol + '_' + k: v for k, v in row.items()})
        for key, value in row.items():
            aggregate.setdefault(key, []).append(value)
        audit[symbol] = {'last_bar_end': str(last.name), 'last_available': str(seen.available_at.max()),
                         'history_last': str(past[-1]), 'regular_rows': len(reg)}
    for key, values in aggregate.items():
        a = np.array(values)
        fields['breadth_' + key + '_mean'] = float(a.mean())
        fields['breadth_' + key + '_std'] = float(a.std(ddof=0))
        fields['breadth_' + key + '_positive_fraction'] = float((a > 0).mean())
    fields['breadth_regular_ready'] = float(not before_open)
    return fields, audit


def variant_columns(features, variant):
    columns = [c for c in features if c not in ['date', 'clock']]
    if variant == 'base':
        return [c for c in columns if not c.startswith(('timing_', 'leader_', 'breadth_'))]
    if variant == 'timing':
        return [c for c in columns if not c.startswith(('leader_', 'breadth_'))]
    if variant == 'breadth':
        return columns
    raise ValueError(variant)
