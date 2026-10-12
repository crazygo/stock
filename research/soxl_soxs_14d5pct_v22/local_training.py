"""Query-conditioned base fitting and calibration using observable sector state."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v21'))
from early_training import (core, terminal_panel, training_dates, capital_decision, bank,
    restrict_market, development_gate, day_cutoff, BASE_COLUMNS, REGIME_COLUMNS)
from paired_entry import fit as global_fit, predict, paired_calibration, decision
from adaptive_blend import STATE_COLUMNS

ROUTES = {
    'state_base_cal': ('state', 'state', 'paired'),
    'state_base_global_cal': ('state', 'global', 'paired'),
    'global_base_state_cal': ('global', 'state', 'paired'),
    'global_base_cal': ('global', 'global', 'paired'),
    'state_raw': ('state', 'state', 'raw'),
}


def similarity_weights(pool, current, recency):
    n = len(pool)
    if n < 40:
        raise ValueError('At least40 historical states required')
    base = 2. ** (-np.arange(n)[::-1]/126.) if recency else np.ones(n)
    base /= base.sum()
    X = pool[STATE_COLUMNS].to_numpy(dtype=float); query = current[STATE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(X).all() and np.isfinite(query).all()
    center = np.average(X, axis=0, weights=base)
    scale = np.sqrt(np.average((X-center)**2, axis=0, weights=base)); scale[scale < 1e-12] = 1.
    distances = np.mean(((X-query)/scale)**2, axis=1)
    bandwidth = max(float(np.sort(distances)[39]), 1e-12)
    logw = np.log(base)-distances/(2*bandwidth); logw -= logw.max()
    q = np.exp(logw); q /= q.sum()
    # Same total loss weight as the global control; the concentration guard is
    # separate. Effective dates summarize weights, not independent observations.
    weights = n*q
    return weights, {'members': list(pool.index), 'recency': recency, 'base_weights': base.tolist(),
        'center': center.tolist(), 'scale': scale.tolist(), 'distances': distances.tolist(),
        'bandwidth': bandwidth, 'weights': weights.tolist(), 'effective_dates': float(1/(q@q))}


def state_fit(panel, dates, at):
    rows = panel[panel.date.isin(dates)]; columns = BASE_COLUMNS+REGIME_COLUMNS
    info = {'dates': dates, 'members': list(rows.index), 'columns': columns,
            'ready': False, 'weights': None, 'conditioned_at': at}
    if len(dates) < 120 or not bool(panel.loc[at, 'feature_ready']):
        return None, info
    w, weights = similarity_weights(rows, panel.loc[at], True); info['weights'] = weights
    if weights['effective_dates'] < 60:
        return None, info
    X = rows[columns].to_numpy(dtype=float); C = rows.C.to_numpy(); D = rows.D.to_numpy(); M = abs(D)
    assert np.isfinite(X).all() and np.isfinite(C).all() and np.isfinite(D).all()
    scaler = StandardScaler().fit(X, sample_weight=w); Z = scaler.transform(X)
    heads = {target: Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w) for target, y in [('C', C), ('M', M)]}
    cw = w*M
    if cw.sum() > 0:
        cw *= len(w)/cw.sum()
    fallback = float(np.average(D > 0, weights=cw)) if cw.sum() > 0 else .5
    if len(np.unique((D[cw > 0] > 0).astype(int))) == 2:
        classifier = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
        classifier.fit(Z, (D > 0).astype(int), sample_weight=cw)
        assert classifier.n_iter_.max() < 2000
        heads['q'] = classifier
    model = {'kind': 'regret', 'columns': columns, 'scaler': scaler, 'heads': heads, 'q': fallback,
             'means': {'C': float(np.average(C, weights=w)), 'D': float(np.average(D, weights=w))}}
    info.update(ready=True, parent_weights=w.tolist(), classification_weights=cw.tolist(), fallback_q=fallback,
        center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), means=model['means'],
        heads={k: {'coefficients': h.coef_.tolist(), 'intercept': np.asarray(h.intercept_).tolist()} for k, h in heads.items()})
    return model, info


def state_calibration(history, panel, dates, at):
    valid = set(dates); old = [e for e in history if e['date'] in valid and e['raw']][-120:]
    ids = [e['at'] for e in old]
    info = {'dates': [e['date'] for e in old], 'members': ids, 'ready': False, 'maps': {},
            'weights': None, 'conditioned_at': at}
    if len(old) < 60 or not bool(panel.loc[at, 'feature_ready']):
        return info
    w, weights = similarity_weights(panel.loc[ids], panel.loc[at], False); info['weights'] = weights
    if weights['effective_dates'] < 20:
        return info
    for target in ['C', 'D']:
        x = np.array([e['raw'][target] for e in old]); y = panel.loc[ids, target].to_numpy(dtype=float)
        center = float(np.average(x, weights=w)); scale = float(np.sqrt(np.average((x-center)**2, weights=w)))
        if scale < 1e-12:
            scale = 1.
        z = (x-center)/scale; intercept = float(np.average(y, weights=w))
        coefficient = float((w*z)@(y-intercept)/((w*z)@z+20.))
        info['maps'][target] = {'center': center, 'scale': scale, 'intercept': intercept,
            'coefficient': coefficient, 'score_min': float(x.min()), 'score_max': float(x.max())}
    info['ready'] = True
    return info


def forecast_routes(panel, at, histories):
    admitted, receipt = training_dates(panel, day_cutoff(at)); dates = admitted['early']
    local, local_info = state_fit(panel, dates, at)
    global_model, global_info = global_fit(panel, dates, 'regret_regime')
    models = {'state': local, 'global': global_model}
    raw = {name: predict(model, panel.loc[[at]]) for name, model in models.items()}
    record = {'fits': {'state': local_info, 'global': global_info}, 'feedback': receipt,
              'conditioning_at': at, 'forecasts': {}}
    for name, (base_mode, cal_mode, method) in ROUTES.items():
        calibration = (state_calibration(histories[base_mode], panel, dates, at) if cal_mode == 'state'
                       else paired_calibration(histories[base_mode], panel, dates))
        event = decision(raw[base_mode], calibration, method)
        if receipt['expired_unresolved_prefix']:
            event.update(ready=False, action='CASH', adjusted=None, top_symbol=None)
        record['forecasts'][name] = {'calibration': calibration, 'event': event}
    return models, record, raw
