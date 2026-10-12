"""Mature, as-issued expert errors determine a bounded paired-payoff blend."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v16'))
from anchored_allocation import (core, bank, prior_pool, day_cutoff,
                                 restrict_market, development_gate)

STATE_COLUMNS = ['relative_5', 'relative_20', 'breadth_5',
                 'strength_persistence_5', 'volume_direction_diffusion',
                 'log_sector_volatility']
ROUTES = {'state_blend': ('state', 'blend'), 'global_blend': ('global', 'blend'),
          'state_hard': ('state', 'hard'), 'equal_blend': ('state', 'equal'),
          'calibrated_fixed': ('state', 'calibrated'), 'raw_fixed': ('state', 'raw')}


def expert_bank(panel, events):
    rows = []
    for e in events:
        if not e['ready']:
            continue
        at = pd.Timestamp(e['at']); row = panel.loc[at]
        rows.append({'at': at, 'date': row.date, 'release': row.release,
            **{c: float(row[c]) for c in STATE_COLUMNS},
            **{f'raw_{k}': e['raw'][k] for k in ['C', 'D']},
            **{f'cal_{k}': e['adjusted'][k] for k in ['C', 'D']},
            **{f'actual_{k}': float(row[k]) for k in ['C', 'D']}})
    return pd.DataFrame(rows).set_index('at').sort_index()


def matured_pool(frame, cutoff):
    day = cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    f = frame[(frame.index < cutoff) & (frame.release < cutoff) & (frame.date < day)]
    finite = np.isfinite(f[[*STATE_COLUMNS, 'raw_C', 'raw_D', 'cal_C', 'cal_D', 'actual_C', 'actual_D']]).all(axis=1)
    return f[finite].tail(120)


def state_weights(pool, current, mode):
    base = 2. ** (-np.arange(len(pool))[::-1] / 63.)
    base /= base.sum(); w = base.copy()
    info = {'mode': mode, 'base_weights': base.tolist(), 'center': None,
            'scale': None, 'distance': None, 'bandwidth': None}
    if mode == 'state':
        X = pool[STATE_COLUMNS].to_numpy(dtype=float)
        q = current[STATE_COLUMNS].to_numpy(dtype=float)
        center = np.average(X, axis=0, weights=base)
        scale = np.sqrt(np.average((X - center) ** 2, axis=0, weights=base))
        scale[scale < 1e-12] = 1.
        distance = np.mean(((X - q) / scale) ** 2, axis=1)
        bandwidth = max(float(np.sort(distance)[min(40, len(pool)) - 1]), 1e-12)
        w *= np.exp(-distance / (2 * bandwidth))
        info.update(center=center.tolist(), scale=scale.tolist(),
                    distance=distance.tolist(), bandwidth=bandwidth)
    elif mode != 'global':
        raise ValueError(mode)
    assert np.isfinite(w).all() and w.sum() > 0
    w /= w.sum()
    info.update(weights=w.tolist(), effective_dates=float(1 / (w @ w)))
    return w, info


def blend_head(raw, calibrated, actual, weights, method='blend'):
    raw, calibrated, actual, w = [np.asarray(x, dtype=float) for x in [raw, calibrated, actual, weights]]
    assert raw.shape == calibrated.shape == actual.shape == w.shape
    assert all(np.isfinite(x).all() for x in [raw, calibrated, actual, w])
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-12
    delta = calibrated - raw; residual = actual - raw
    curvature = float(w @ (delta * delta)); linear = float(w @ (delta * residual))
    effective = float(1 / (w @ w)); penalty = 2. / effective * curvature
    alpha = .5 if curvature <= 1e-16 else float(np.clip((linear + .5 * penalty) / (curvature + penalty), 0., 1.))
    mse_raw = float(w @ (residual ** 2)); mse_cal = float(w @ ((actual - calibrated) ** 2))
    if method == 'hard':
        alpha = .5 if abs(mse_raw - mse_cal) <= 1e-12 else float(mse_cal < mse_raw)
    elif method in ['raw', 'calibrated', 'equal']:
        alpha = {'raw': 0., 'calibrated': 1., 'equal': .5}[method]
    elif method != 'blend':
        raise ValueError(method)
    mse_blend = float(w @ ((actual - (raw + alpha * delta)) ** 2))
    return {'alpha_calibrated': alpha, 'curvature': curvature, 'linear': linear,
        'penalty': penalty, 'mse_raw': mse_raw, 'mse_calibrated': mse_cal,
        'mse_blended': mse_blend, 'objective': mse_blend + penalty * (alpha - .5) ** 2}


def forecast(pool, current, event, mode, method):
    info = {'ready': False, 'members': [str(t) for t in pool.index],
            'dates': list(pool.date), 'weights': None, 'heads': {}, 'adjusted': None}
    if len(pool) < 60 or not event['ready']:
        return info
    w, wi = state_weights(pool, current, mode); info['weights'] = wi
    if wi['effective_dates'] < 20 and method not in ['raw', 'calibrated', 'equal']:
        return info
    values = {}
    for k in ['C', 'D']:
        head = blend_head(pool['raw_' + k], pool['cal_' + k], pool['actual_' + k], w, method)
        alpha = head['alpha_calibrated']; info['heads'][k] = head
        values[k] = (1 - alpha) * event['raw'][k] + alpha * event['adjusted'][k]
    C, D = values['C'], values['D']
    info.update(ready=True, adjusted={**values, 'SOXL': C + D, 'SOXS': C - D},
                top_symbol='SOXL' if D >= 0 else 'SOXS')
    return info


def risk_decision(frame, cutoff, at, blend):
    if not blend['ready']:
        return {'ready': False, 'action': 'CASH', 'allocation': 0.,
                'reason': 'blend_unready', 'top_symbol': None, 'fit': None}
    symbol = blend['top_symbol']; pool = prior_pool(frame, cutoff, symbol)
    current = frame.loc[(at, symbol)].copy()
    # Preserve the old prior-distance coordinates for exact calibrated control.
    # Only the entropy mean target is changed, after the prior is computed.
    from allocation import weights, utility, weighted_tail
    from anchored_allocation import mean_anchor
    fit = {'ready': False, 'members': [str(t) for t in pool.index],
           'dates': list(pool.date), 'prior': None, 'anchor': None, 'distribution': None}
    result = {'ready': False, 'action': 'CASH', 'allocation': 0.,
              'reason': 'insufficient_risk_history', 'top_symbol': symbol, 'fit': fit}
    if len(pool) < 60:
        return result
    w, prior = weights(pool, current, 'local'); fit['prior'] = prior
    if prior['effective_dates'] < 20:
        result['reason'] = 'insufficient_prior_effective_dates'; return result
    r = pool.net_return.to_numpy(dtype=float)
    target = blend['adjusted'][symbol]
    q, anchor = mean_anchor(r, w, target); fit['anchor'] = anchor
    if q is None:
        result['reason'] = 'mean_outside_empirical_support'; return result
    effective = float(1 / (q @ q))
    fit['distribution'] = {'weights': q.tolist(), 'returns': r.tolist(), 'effective_dates': effective,
        'mean_return': float(q @ r), 'mean_log_return': float(q @ np.log1p(r)),
        'lower_5pct_mean_return': weighted_tail(r, q)}
    if effective < 20:
        result['reason'] = 'insufficient_posterior_effective_dates'; return result
    scores = utility(r, np.array([0., 1.]), 5.) @ q
    allocation = float(scores[1] > scores[0] + 1e-12)
    fit.update(ready=True, utility_scores=scores.tolist())
    result.update(ready=True, action=symbol if allocation else 'CASH', allocation=allocation,
                  reason='positive_utility' if allocation else 'cash_best')
    return result
