"""Daily, matched-date partial pooling of real paired ETF contract returns."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/soxl_soxs_14d5pct_v69'))
from path_learning import (core, read, dump, sha, labels, day_cutoff, registry,
                           fit_day, issue, decide, development_gate, restrict_market,
                           CLOCKS, tag)
sys.path.insert(0, str(ROOT / 'research/soxl_soxs_14d5pct_v70'))
from market_adapter import load as auxiliary_market, label as auxiliary_label
from partial_pool import fit, predict

BASE = ROOT / '.cache/soxl_soxs_14d5pct_v69/inputs/development_20261013'
PAIRS = {'semiconductor': ('SOXL', 'SOXS', 'SOXX'),
         'nasdaq': ('TQQQ', 'SQQQ', 'QQQ'),
         'sp500': ('UPRO', 'SPXU', 'SPY')}
VARIANTS = ['partial', 'independent', 'pooled']
ROUTES = [k + '_' + suffix for k in VARIANTS for suffix in ['band', 'point']] + [
    'premarket_only', 'cash', 'double_cost']
FIELDS = ['trend1', 'trend5', 'trend20', 'log_vol20', 'vol_ratio', 'relative5',
          'current', 'recent15', 'market_current', 'market_trend5', 'efficiency20',
          'pair_activity', 'direction_activity', 'relative_activity',
          'direction_confirmation', 'relative_confirmation', 'trend_confirmation',
          'pair_current_decay']
CUTOFF = pd.Timestamp('2026-05-01', tz=core.ET).tz_convert('UTC')


def load_market():
    base = restrict_market(core.load_market(), CUTOFF)
    extra = restrict_market(auxiliary_market(), CUTOFF)
    assert base.calendar == extra.calendar and base.sessions.equals(extra.sessions)
    return core.Market(*[{**getattr(base, k), **getattr(extra, k)} for k in
                        ['bars', 'daily']], base.calendar, base.sessions,
                       *[{**getattr(base, k), **getattr(extra, k)} for k in
                         ['actions', 'day_complete', 'manifest']])


def prefix_table(market, index):
    """Full 04:00-to-clock prefixes; no omitted or future bars are eligible."""
    symbols = sorted({s for pair in PAIRS.values() for s in pair})
    output = {}
    for s in symbols:
        frame = market.bars[s]
        rows = []
        for at in index:
            day = at.tz_convert(core.ET).tz_localize(None).normalize()
            end = at - pd.Timedelta(seconds=30)
            opening = pd.Timestamp(str(day.date()) + ' 04:00', tz=core.ET).tz_convert('UTC')
            expected = pd.date_range(opening, end, freq='5min', inclusive='right')
            g = frame.loc[opening + pd.Timedelta(minutes=5):end]
            ok = (g.index.equals(expected) and len(g) >= 4 and
                  g.available_at.notna().all() and g.available_at.between(g.end_at, at).all() and
                  g.start_at.eq(g.end_at - pd.Timedelta(minutes=5)).all())
            values = g[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
            ok = bool(ok and np.isfinite(values).all() and (values[:, :4] > 0).all()
                      and (values[:, 4] >= 0).all())
            amount = float(((g.high + g.low + g.close) / 3 * g.volume).sum()) if ok else np.nan
            ok = bool(ok and amount > 0)
            rows.append({'at': at, 'date': day, 'clock': at.tz_convert(core.ET).strftime('%H:%M:%S'),
                         'prefix_ready': ok, 'amount': amount if ok else np.nan,
                         'quote': float(g.econ_close.iloc[-1]) if ok else np.nan,
                         'quote15': float(g.econ_close.iloc[-4]) if ok else np.nan,
                         'available_at': g.available_at.max() if ok else pd.NaT})
        output[s] = pd.DataFrame(rows).set_index('at')
    return output


def feature_row(market, prefixes, pair, at):
    long, short, bench = PAIRS[pair]
    day = at.tz_convert(core.ET).tz_localize(None).normalize()
    clock = at.tz_convert(core.ET).strftime('%H:%M:%S')
    previous = market.sessions[market.sessions < day][-21:]
    receipt = {'ready': False, 'history_dates': list(previous), 'activity_dates': [],
               'max_daily_available': None, 'max_prefix_available': None}
    if len(previous) != 21:
        return None, receipt
    stats = {}; activity = {}; daily_times = []; prefix_times = []
    eligible = [d for d in market.sessions[market.sessions < day]
                if pd.Timestamp(str(d.date()) + ' ' + clock, tz=core.ET) <
                pd.Timestamp(market.calendar[d]['close_at'])][-20:]
    receipt['activity_dates'] = eligible
    if len(eligible) != 20:
        return None, receipt
    oldticks = [pd.Timestamp(str(d.date()) + ' ' + clock, tz=core.ET).tz_convert('UTC') for d in eligible]
    for s in dict.fromkeys([long, short, bench, 'SPY']):
        d = market.daily[s].reindex(previous)
        if (not d.complete.fillna(False).all() or d.available_at.isna().any() or
            not d.available_at.lt(day_cutoff(at)).all()):
            return None, receipt
        prices = d.close.to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            return None, receipt
        p = prefixes[s]
        group = p.reindex([*oldticks, at])
        if not group.prefix_ready.fillna(False).all():
            return None, receipt
        returns = prices[1:] / prices[:-1] - 1
        sigma = max(float(np.std(returns, ddof=1)), 1e-4)
        current = float(group.quote.iloc[-1] / prices[-1] - 1)
        stats[s] = dict(prices=prices, sigma=sigma, current=current,
                        ret1=prices[-1] / prices[-2] - 1, ret5=prices[-1] / prices[-6] - 1,
                        ret20=prices[-1] / prices[0] - 1,
                        vol5=float(np.std(returns[-5:], ddof=1)),
                        recent15=float(group.quote.iloc[-1] / group.quote15.iloc[-1] - 1))
        activity[s] = float(np.clip(np.log(group.amount.iloc[-1] / group.amount.iloc[:-1].mean()), -3, 3))
        daily_times.append(d.available_at.max()); prefix_times.append(group.available_at.max())
    b, m = stats[bench], stats['SPY']
    clip = lambda x: float(np.clip(x, -8, 8))
    t1, t5, t20 = clip(b['ret1']/b['sigma']), clip(b['ret5']/(b['sigma']*np.sqrt(5))), clip(b['ret20']/(b['sigma']*np.sqrt(20)))
    current = clip(b['current']/b['sigma'])
    direction = (activity[long] - activity[short])/2
    relative = (activity[bench] - activity['SPY'])/2
    variation = np.abs(np.diff(b['prices'])).sum()
    vals = [t1, t5, t20, np.log(b['sigma']), min(b['vol5']/b['sigma'], 8),
            clip((b['ret5']-m['ret5'])/(b['sigma']*np.sqrt(5))), current,
            clip(b['recent15']/b['sigma']), clip(m['current']/m['sigma']),
            clip(m['ret5']/(m['sigma']*np.sqrt(5))),
            (b['prices'][-1]-b['prices'][0])/variation if variation > 0 else 0.,
            (activity[long]+activity[short])/2, direction, relative,
            clip(direction*current), clip(relative*current), clip(t5*current),
            clip((stats[long]['current']+stats[short]['current'])/(stats[long]['sigma']+stats[short]['sigma']))]
    receipt.update(ready=True, max_daily_available=max(daily_times), max_prefix_available=max(prefix_times))
    assert max(daily_times) < day_cutoff(at) and max(prefix_times) <= at
    return np.asarray(vals), receipt


def matched_registry(panels, market):
    result = registry(panels['semiconductor'], market)
    others = {k: registry(v, market) for k, v in panels.items()}
    for i, entry in enumerate(result):
        same = [r[i] for r in others.values()]
        assert all(e['date'] == entry['date'] and e['members'] == entry['members'] for e in same)
        entry['feature_ready'] = all(e['feature_ready'] for e in same)
        releases = [e['release'] for e in same]
        entry['release'] = max(releases) if entry['feature_ready'] and all(r is not None for r in releases) else None
    return result


def mature_prefix(reg, at):
    cutoff = day_cutoff(at); day = cutoff.tz_convert(core.ET).tz_localize(None)
    prior = [r for r in reg if r['date'] < day and r['feature_ready']]
    prefix = []
    for row in prior:
        if row['release'] is None or row['release'] >= cutoff:
            break
        prefix.append(row)
    pending = prior[len(prefix)] if len(prefix) < len(prior) else None
    return prefix, {'cutoff': cutoff, 'real_dates': len(prefix),
                    'first_pending': pending['date'] if pending else None,
                    'expired_unresolved_prefix': bool(pending and pending['date'] + pd.Timedelta(days=14) <= day)}


def training_arrays(panels, parents, clock):
    ids = [at for r in parents for at in r['members'] if panels['semiconductor'].loc[at, 'clock'] == clock]
    n = len(ids); weights = 2.**(-np.arange(n)[::-1]/126.)
    weights *= n/weights.sum()
    X = np.concatenate([p.loc[ids, FIELDS].to_numpy(float) for p in panels.values()])
    Y = np.concatenate([p.loc[ids, ['C', 'D']].to_numpy(float) for p in panels.values()])
    names = np.repeat(list(panels), n); w = np.tile(weights/len(panels), len(panels))
    return X, Y, names, w, ids


def fit_clock(panels, parents, clock):
    if len(parents) < 120:
        return {}, []
    X, Y, names, w, ids = training_arrays(panels, parents, clock)
    if len(ids) < 120:
        return {}, ids
    models = {kind: fit(X, Y, names, w, len(ids), mode=kind) for kind in VARIANTS}
    return models, ids


def raw_prediction(model, row):
    if model is None or not row.feature_ready:
        return None
    cd = predict(model, row[FIELDS].to_numpy(float)[None, :], np.array(['semiconductor']))[0]
    return np.array([cd[0]+cd[1], cd[0]-cd[1]])
