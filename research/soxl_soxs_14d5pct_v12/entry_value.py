"""Daily two-part contract value, with immutable past forecast calibration."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v11'))
from uncertainty import core, restrict_market, development_gate
from continuation import ENTRY_COLUMNS, entry_state, baseline_labels, joint_releases, label_key
from walkforward import day_cutoff

PARAMS = dict(max_iter=80, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=.05,
              l2_regularization=5, early_stopping=False, random_state=814)
LEARNERS = ['two_part_expanding', 'direct_expanding', 'mean_expanding', 'two_part_252']


def eligible(releases, cutoff, rolling=False):
    dates = [d for d, t in sorted(releases.items()) if t < cutoff and d.tz_localize(core.ET) < cutoff]
    return dates[-252:] if rolling else dates


def examples(features, labels, dates):
    return [{'id': label_key(at, s), 'date': r.date, 'symbol': s,
             'state': entry_state(r, s), 'hit': int(labels[label_key(at, s)]['target_hit']),
             'log_return': float(np.log1p(labels[label_key(at, s)]['net_return']))}
            for at, r in features[features.date.isin(dates)].iterrows() for s in core.ETFS]


def fit(rows, method, protocol):
    dates = sorted({r['date'] for r in rows})
    info = {'dates': dates, 'members': [r['id'] for r in rows], 'rows': len(rows),
            'ready': len(dates) >= protocol['minimum_training_dates']}
    if not info['ready']:
        return None, info
    X = np.array([[r['state'][c] for c in ENTRY_COLUMNS] for r in rows])
    y = np.array([r['log_return'] for r in rows]); hit = np.array([r['hit'] for r in rows])
    w = np.full(len(rows), .5)
    assert len(rows) == 2 * len(dates) and np.isfinite(X).all() and np.isfinite(y).all()
    means = {}
    for s in core.ETFS:
        mask = np.array([r['symbol'] == s for r in rows])
        wins = y[mask & (hit == 1)]; losses = y[mask & (hit == 0)]
        means[s] = {'p': float(hit[mask].mean()), 'log_return': float(y[mask].mean()),
                    'win': float(wins.mean()) if len(wins) else 0.,
                    'failure': float(losses.mean()) if len(losses) else 0.}
    models = {'method': method, 'means': means, 'direct': None, 'classifier': None, 'failure': None}
    info.update(weight_sum=float(w.sum()), means=means)
    if method == 'direct_expanding':
        models['direct'] = HistGradientBoostingRegressor(**PARAMS).fit(X, y, sample_weight=w)
    elif method.startswith('two_part'):
        if len(np.unique(hit)) == 2:
            models['classifier'] = HistGradientBoostingClassifier(**PARAMS).fit(X, hit, sample_weight=w)
        failed_dates = sorted({r['date'] for r in rows if r['hit'] == 0})
        info['failure_dates'] = failed_dates
        info['failure_members'] = [r['id'] for r in rows if r['hit'] == 0]
        info['failure_model_ready'] = len(failed_dates) >= protocol['minimum_failure_dates']
        if info['failure_model_ready']:
            # Preserve the original equal ETF/date observation measure after conditioning.
            models['failure'] = HistGradientBoostingRegressor(**PARAMS).fit(X[hit == 0], y[hit == 0], sample_weight=w[hit == 0])
    elif method != 'mean_expanding':
        raise ValueError(method)
    return models, info


def forecast(model, row):
    if model is None:
        return {}
    X = np.array([[entry_state(row, s)[c] for c in ENTRY_COLUMNS] for s in core.ETFS])
    out = {}
    for i, s in enumerate(core.ETFS):
        m = model['means'][s]
        p = float(model['classifier'].predict_proba(X)[i, 1]) if model['classifier'] is not None else m['p']
        fail = float(model['failure'].predict(X)[i]) if model['failure'] is not None else m['failure']
        direct = float(model['direct'].predict(X)[i]) if model['direct'] is not None else m['log_return']
        out[s] = {'p': p, 'win': m['win'], 'failure': fail, 'direct': direct}
    return out


def calibrate(history, labels, dates, protocol):
    """Only old as-issued forecasts of jointly mature dates; never in-sample refits."""
    valid = set(dates)
    past = [e for e in history if e['date'] in valid and e['raw']][-protocol['calibration_window_dates']:]
    probability = {}
    for s in core.ETFS:
        residuals = [labels[label_key(e['at'], s)]['target_hit'] - e['raw'][s]['p'] for e in past]
        probability[s] = float(np.mean(residuals)) if residuals else None
    chosen = [e for e in history if e['date'] in valid and e['top_symbol'] is not None
              and e['score_ready']][-protocol['calibration_window_dates']:]
    residuals = [np.log1p(labels[label_key(e['at'], e['top_symbol'])]['net_return']) - e['scores'][e['top_symbol']]
                 for e in chosen]
    return {'probability_dates': [e['date'] for e in past], 'probability_correction': probability,
            'probability_ready': len(past) >= protocol['minimum_calibration_dates'],
            'top_dates': [e['date'] for e in chosen],
            'top_members': [label_key(e['at'], e['top_symbol']) for e in chosen],
            'top_correction': float(np.mean(residuals)) if residuals else None,
            'top_ready': len(chosen) >= protocol['minimum_calibration_dates']}


def decide(raw, cal, method):
    event = {'raw': raw, 'scores': {}, 'probabilities': {}, 'top_symbol': None,
             'score_ready': False, 'ready': False, 'adjusted': None, 'action': 'CASH'}
    if not raw:
        return event
    two_part = method.startswith('two_part')
    event['score_ready'] = bool(cal['probability_ready'] if two_part else True)
    for s, r in raw.items():
        correction = cal['probability_correction'][s]
        p = float(np.clip(r['p'] + (correction or 0.), .001, .999))
        event['probabilities'][s] = p
        event['scores'][s] = p * r['win'] + (1 - p) * r['failure'] if two_part else r['direct']
    top = max(core.ETFS, key=lambda s: event['scores'][s])
    adjusted = event['scores'][top] + cal['top_correction'] if cal['top_correction'] is not None else None
    ready = event['score_ready'] and cal['top_ready']
    event.update(top_symbol=top, adjusted=adjusted, ready=ready,
                 action=top if ready and adjusted > 0 else 'CASH')
    return event
