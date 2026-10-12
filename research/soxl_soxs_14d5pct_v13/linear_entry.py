"""Small regularized entry models and monotone selected-value calibration."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v12'))
from entry_value import (core, ENTRY_COLUMNS, entry_state, label_key, baseline_labels,
                        joint_releases, eligible, restrict_market, day_cutoff, development_gate)

FACTOR_COLUMNS = ['direction', 'sector_1', 'sector_5', 'sector_20', 'market_5',
                  'sector_premarket', 'breadth_premarket', 'own_log_volatility']
MODELS = {
    'logit_factor_recent': ('logit', 'factor', True),
    'logit_factor_uniform': ('logit', 'factor', False),
    'logit_full_recent': ('logit', 'full', True),
    'ridge_factor_recent': ('ridge', 'factor', True),
    'mean_recent': ('mean', 'factor', True),
}


def factors(row, symbol):
    sign = 1. if symbol == 'SOXL' else -1.
    sv = max(float(row['SOXX_vol20']), 1e-4); qv = max(float(row['QQQ_vol20']), 1e-4)
    own = max(float(row[symbol + '_vol20']), 1e-4)
    returns = [sign * row['SOXX_ret1'] / sv,
               sign * row['SOXX_ret5'] / (sv * np.sqrt(5)),
               sign * row['SOXX_ret20'] / (sv * np.sqrt(20)),
               sign * row['QQQ_ret5'] / (qv * np.sqrt(5)),
               sign * row['SOXX_today_return'] / sv,
               sign * row['breadth_today_mean'] / sv]
    values = np.r_[sign, np.clip(returns, -5, 5), np.log(own)]
    assert np.isfinite(values).all()
    return dict(zip(FACTOR_COLUMNS, map(float, values)))


def prepare(features, labels, releases):
    records = []
    for at, row in features.iterrows():
        for s in core.ETFS:
            lab = labels[label_key(at, s)]; net = lab.get('net_return', np.nan)
            records.append({'id': label_key(at, s), 'at': at, 'date': row.date, 'symbol': s,
                'release': releases.get(row.date, pd.NaT), 'hit': lab.get('target_hit', np.nan),
                'log_return': float(np.log1p(net)) if np.isfinite(net) else np.nan,
                **{'factor_' + k: v for k, v in factors(row, s).items()},
                **{'full_' + k: v for k, v in entry_state(row, s).items()}})
    return pd.DataFrame(records).set_index('id')


def date_weights(n, recent):
    if n == 0: return np.array([], dtype=float)
    w = 2. ** (-(np.arange(n)[::-1]) / 126.) if recent else np.ones(n)
    return w * n / w.sum()


def fit(panel, dates, name, minimum=120):
    kind, group, recent = MODELS[name]
    rows = panel[panel.date.isin(dates)]
    parent_weights = date_weights(len(dates), recent)
    weights = rows.date.map(dict(zip(dates, parent_weights))).to_numpy(dtype=float) / 2
    cols = [group + '_' + c for c in (FACTOR_COLUMNS if group == 'factor' else ENTRY_COLUMNS)]
    info = {'dates': dates, 'members': list(rows.index), 'columns': cols, 'ready': len(dates) >= minimum,
        'parent_weights': parent_weights.tolist(), 'weight_sum': float(weights.sum()),
        'parent_effective_n': float(parent_weights.sum() ** 2 / np.square(parent_weights).sum()) if len(dates) else 0.}
    assert len(rows) == 2 * len(dates)
    if not info['ready']: return None, info
    X = rows[cols].to_numpy(); y = rows.log_return.to_numpy(); hit = rows.hit.to_numpy(dtype=int)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    means = {}
    for s in core.ETFS:
        mask = rows.symbol.eq(s).to_numpy(); won = mask & (hit == 1); lost = mask & (hit == 0)
        means[s] = {'p': float(np.average(hit[mask], weights=weights[mask])),
            'win': float(np.average(y[won], weights=weights[won])) if won.any() else 0.,
            'failure': float(np.average(y[lost], weights=weights[lost])) if lost.any() else 0.,
            'value': float(np.average(y[mask], weights=weights[mask]))}
    scaler = StandardScaler().fit(X, sample_weight=weights); standardized = scaler.transform(X)
    estimator = None
    if kind == 'logit' and len(np.unique(hit)) == 2:
        estimator = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
        estimator.fit(standardized, hit, sample_weight=weights)
        assert int(estimator.n_iter_.max()) < 2000
    elif kind == 'ridge':
        estimator = Ridge(alpha=100., solver='svd').fit(standardized, y, sample_weight=weights)
    model = {'kind': kind, 'columns': cols, 'means': means, 'scaler': scaler, 'estimator': estimator}
    info.update(means=means, center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficients=estimator.coef_.tolist() if estimator is not None else None,
                intercept=np.asarray(estimator.intercept_).tolist() if estimator is not None else None)
    return model, info


def predict(model, current):
    if model is None: return {}
    assert list(current.symbol) == list(core.ETFS)
    X = model['scaler'].transform(current[model['columns']].to_numpy())
    estimator = model['estimator']; values = {}
    for i, s in enumerate(core.ETFS):
        means = model['means'][s]
        p = float(estimator.predict_proba(X)[i, 1]) if model['kind'] == 'logit' and estimator is not None else means['p']
        value = (p * means['win'] + (1 - p) * means['failure'] if model['kind'] == 'logit'
                 else float(estimator.predict(X)[i]) if estimator is not None else means['value'])
        values[s] = {'p': p, 'score': value, 'baseline_p': means['p'],
                     'win': means['win'], 'failure': means['failure']}
    return values


def selected_calibration(history, panel, dates):
    valid = set(dates)
    old = [e for e in history if e['date'] in valid and e['top_symbol'] is not None][-120:]
    ids = [label_key(e['at'], e['top_symbol']) for e in old]
    x = np.array([e['raw'][e['top_symbol']]['score'] for e in old])
    y = panel.loc[ids, 'log_return'].to_numpy(dtype=float) if ids else np.array([])
    info = {'dates': [e['date'] for e in old], 'members': ids, 'ready': len(old) >= 60,
            'center': None, 'scale': None, 'intercept': None, 'coefficient': None,
            'unconstrained_coefficient': None, 'offset': None}
    if not len(old): return info
    assert np.isfinite(x).all() and np.isfinite(y).all()
    center = float(x.mean()); scale = float(x.std())
    if scale < 1e-12: scale = 1.
    z = (x - center) / scale; intercept = float(y.mean())
    coefficient = float(z @ (y - intercept) / (z @ z + 20.))
    info.update(center=center, scale=scale, intercept=intercept,
                coefficient=max(0., coefficient), unconstrained_coefficient=coefficient,
                offset=float(np.mean(y - x)), score_min=float(x.min()), score_max=float(x.max()))
    return info


def decision(raw, calibration, method='conditional'):
    event = {'raw': raw, 'top_symbol': None, 'adjusted': None, 'ready': False, 'action': 'CASH'}
    if not raw: return event
    top = max(core.ETFS, key=lambda s: raw[s]['score']); score = raw[top]['score']
    event['top_symbol'] = top
    if not calibration['ready']: return event
    if method == 'conditional':
        adjusted = calibration['intercept'] + calibration['coefficient'] * (score - calibration['center']) / calibration['scale']
    elif method == 'offset': adjusted = score + calibration['offset']
    elif method == 'raw': adjusted = score
    else: raise ValueError(method)
    event.update(adjusted=float(adjusted), ready=True, action=top if adjusted > 0 else 'CASH',
                 within_calibration_score_range=calibration['score_min'] <= score <= calibration['score_max'])
    return event
