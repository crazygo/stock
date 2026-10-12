"""Causal sector participation and daily paired payoff learning."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v13'))
from linear_entry import (core, baseline_labels, joint_releases, eligible, restrict_market,
                          day_cutoff, development_gate, label_key, date_weights)
from breadth import PANEL, load_panel

BASE_COLUMNS = ['sector_1', 'sector_5', 'sector_20', 'market_5', 'sector_premarket',
                'breadth_premarket', 'log_sector_volatility']
REGIME_COLUMNS = ['relative_5', 'relative_20', 'breadth_5', 'breadth_20', 'breadth_today',
    'strength_persistence_5', 'breadth_change_5', 'volume_log_mean', 'volume_direction_diffusion',
    'trend_volume_alignment', 'relative_volume_interaction', 'breadth_volume_interaction']
MODELS = {'regret_regime': ('regret', True), 'regret_base': ('regret', False),
          'sign_regime': ('sign', True), 'ridge_regime': ('ridge', True), 'mean_pair': ('mean', False)}


def premarket_volumes(bars, sessions, cutoff):
    """Exact 04:00-09:25 ET prefix, available by 09:25:30, per official date."""
    volumes = {}; checks = []
    for symbol in PANEL:
        f = bars[symbol]; f = f[(f.end_at < cutoff) & (f.available_at < cutoff)]
        local = f.end_at.dt.tz_convert(core.ET); minutes = local.dt.hour * 60 + local.dt.minute
        f = f[(minutes > 240) & (minutes <= 565)]
        records = {}
        for day, group in f.groupby('date'):
            start = day.tz_localize(core.ET) + pd.Timedelta(hours=4)
            end = day.tz_localize(core.ET) + pd.Timedelta(hours=9, minutes=25)
            expected = pd.date_range(start, end, freq='5min', inclusive='right').tz_convert('UTC')
            ok = (group.index.equals(expected) and (group.available_at <= end + pd.Timedelta(seconds=30)).all()
                  and group.econ_volume.ge(0).all() and group.econ_volume.sum() > 0)
            records[day] = float(group.econ_volume.sum()) if ok else np.nan
            checks.append({'symbol': symbol, 'date': day, 'rows': len(group), 'complete': bool(ok),
                           'latest_available': group.available_at.max()})
        volumes[symbol] = pd.Series(records).reindex(sessions[sessions.tz_localize(core.ET) < cutoff])
    return pd.DataFrame(volumes), checks


def factor_frame(features, volumes):
    rows = []
    relative = features.sector_relative_5
    breadth = 2 * features.breadth_ret5_positive_fraction - 1
    for i, (at, row) in enumerate(features.iterrows()):
        sv = max(float(row.SOXX_vol20), 1e-4); qv = max(float(row.QQQ_vol20), 1e-4)
        six = np.clip([row.SOXX_ret1/sv, row.SOXX_ret5/(sv*np.sqrt(5)),
            row.SOXX_ret20/(sv*np.sqrt(20)), row.QQQ_ret5/(qv*np.sqrt(5)),
            row.SOXX_today_return/sv, row.breadth_today_mean/sv], -5, 5)
        base = dict(zip(BASE_COLUMNS, map(float, [*six, np.log(sv)])))
        past = volumes.loc[volumes.index < row.date].tail(20)
        ready = i >= 5 and len(past) == 20 and past.notna().all().all() and volumes.loc[row.date].notna().all()
        extra = {c: np.nan for c in REGIME_COLUMNS}
        if ready:
            ratios = volumes.loc[row.date].to_numpy() / past.mean().to_numpy()
            log_ratio = np.clip(np.log(ratios), -3, 3)
            today = np.array([row['leader_'+s+'_today'] for s in PANEL])
            trend = np.array([row['leader_'+s+'_ret5'] for s in PANEL])
            r5 = float(np.clip(row.sector_relative_5/(sv*np.sqrt(5)), -5, 5))
            r20 = float(np.clip(row.sector_relative_20/(sv*np.sqrt(20)), -5, 5))
            b5 = float(breadth.iloc[i]); vm = float(log_ratio.mean())
            extra = dict(zip(REGIME_COLUMNS, map(float, [r5, r20, b5,
                2*row.breadth_ret20_positive_fraction-1, 2*row.breadth_today_positive_fraction-1,
                np.sign(relative.iloc[i-4:i+1]).mean(), b5-breadth.iloc[i-5:i].mean(), vm,
                ((ratios>1)*(today>0)).mean()-((ratios>1)*(today<0)).mean(),
                (np.sign(trend)*log_ratio).mean(), r5*vm, b5*vm])))
        rows.append({'at': at, 'date': row.date, 'feature_ready': bool(ready), **base, **extra})
    return pd.DataFrame(rows).set_index('at')


def prepare(features, labels, releases, factors):
    panel = factors.copy()
    for at, row in features.iterrows():
        values = [labels[label_key(at, s)].get('net_return', np.nan) for s in core.ETFS]
        l, s = np.log1p(values)
        panel.loc[at, 'C'] = (l+s)/2; panel.loc[at, 'D'] = (l-s)/2
    panel['M'] = panel.D.abs(); panel['release'] = panel.date.map(releases)
    return panel


def fit(panel, dates, name, minimum=120):
    kind, regime = MODELS[name]; rows = panel[panel.date.isin(dates)]
    cols = BASE_COLUMNS + (REGIME_COLUMNS if regime else [])
    w = date_weights(len(dates), True)
    info = {'dates': dates, 'members': list(rows.index), 'columns': cols, 'ready': len(dates)>=minimum,
        'parent_weights': w.tolist(), 'weight_sum': float(w.sum()),
        'parent_effective_n': float(w.sum()**2/(w@w)) if len(w) else 0.}
    assert len(rows) == len(dates) and rows.feature_ready.all()
    if not info['ready']: return None, info
    X = rows[cols].to_numpy(); C = rows.C.to_numpy(); D = rows.D.to_numpy(); M = np.abs(D)
    assert np.isfinite(X).all() and np.isfinite(C).all() and np.isfinite(D).all()
    scaler = StandardScaler().fit(X, sample_weight=w); Z = scaler.transform(X)
    model = {'kind': kind, 'columns': cols, 'scaler': scaler, 'heads': {},
             'means': {'C': float(np.average(C, weights=w)), 'D': float(np.average(D, weights=w))}}
    cw = w*M if kind=='regret' else w.copy()
    if cw.sum() > 0: cw = cw*len(w)/cw.sum()
    fallback_q = float(np.average(D>0, weights=cw)) if cw.sum()>0 else .5
    model['q'] = fallback_q
    if kind in ['regret', 'sign']:
        for target, y in [('C', C), ('M', M)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
        if len(np.unique((D[cw>0]>0).astype(int))) == 2:
            classifier = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
            classifier.fit(Z, (D>0).astype(int), sample_weight=cw)
            assert classifier.n_iter_.max() < 2000
            model['heads']['q'] = classifier
    elif kind=='ridge':
        for target, y in [('C', C), ('D', D)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
    info.update(classification_weights=cw.tolist(), fallback_q=fallback_q,
                center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), means=model['means'],
                heads={k: {'coefficients': h.coef_.tolist(), 'intercept': np.asarray(h.intercept_).tolist()}
                       for k,h in model['heads'].items()})
    return model, info


def predict(model, current):
    if model is None or not bool(current.feature_ready.iloc[0]): return {}
    Z = model['scaler'].transform(current[model['columns']].to_numpy()); heads = model['heads']
    C = float(heads['C'].predict(Z)[0]) if 'C' in heads else model['means']['C']
    q = float(heads['q'].predict_proba(Z)[0, 1]) if 'q' in heads else model['q']
    M = max(0., float(heads['M'].predict(Z)[0])) if 'M' in heads else None
    D = M*(2*q-1) if M is not None else float(heads['D'].predict(Z)[0]) if 'D' in heads else model['means']['D']
    return {'C': C, 'D': D, 'M': M, 'q': q, 'SOXL': C+D, 'SOXS': C-D}


def paired_calibration(history, panel, dates):
    valid = set(dates); old = [e for e in history if e['date'] in valid and e['raw']][-120:]
    ids = [e['at'] for e in old]
    info = {'dates': [e['date'] for e in old], 'members': ids, 'ready': len(old)>=60, 'maps': {}}
    for target in ['C', 'D']:
        if not old: continue
        x = np.array([e['raw'][target] for e in old]); y = panel.loc[ids, target].to_numpy(dtype=float)
        center = float(x.mean()); scale = float(x.std()) if x.std()>=1e-12 else 1.
        z = (x-center)/scale; intercept = float(y.mean())
        info['maps'][target] = {'center': center, 'scale': scale, 'intercept': intercept,
            'coefficient': float(z@(y-intercept)/(z@z+20.)), 'score_min': float(x.min()), 'score_max': float(x.max())}
    return info


def decision(raw, calibration, method='paired'):
    event = {'raw': raw, 'adjusted': None, 'top_symbol': None, 'ready': False, 'action': 'CASH'}
    if not raw or not calibration['ready']: return event
    values = raw.copy() if method=='raw' else {k: m['intercept']+m['coefficient']*(raw[k]-m['center'])/m['scale']
                                                for k,m in calibration['maps'].items()}
    C, D = values['C'], values['D']; top = 'SOXL' if D>=0 else 'SOXS'
    event.update(adjusted={'C': C, 'D': D, 'SOXL': C+D, 'SOXS': C-D}, top_symbol=top, ready=True,
        action=top if method=='direction' or C+abs(D)>0 else 'CASH',
        within_calibration_range={k: m['score_min']<=raw[k]<=m['score_max'] for k,m in calibration['maps'].items()})
    return event
