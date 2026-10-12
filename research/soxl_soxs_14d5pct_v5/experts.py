"""Causal context, entry-frozen risk labels, and heterogeneous daily experts."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

V4 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v4'
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))
from walkforward import core, label_tables, eligible_index, day_cutoff, digest
from risk import risk_label

ALGORITHMS = ['ridge_pair', 'boosted_log', 'analog_tail', 'quantile_tail']
RISKS = ['expiry14', 'atr_stop', 'atr_protect']
EXPERTS = [a + '__' + r for r in RISKS for a in ALGORITHMS] + ['trend20__expiry14', 'trend20__atr_protect']


def extend_features(market, base):
    output = []
    for at, original in base.iterrows():
        row = original.to_dict()
        valid = True
        for symbol in core.SYMBOLS:
            frame = market.daily[symbol]
            expected = [d for d in market.sessions if pd.Timestamp(market.calendar[d]['close_at']) + pd.Timedelta(seconds=1) <= at][-61:]
            d = frame.reindex(expected)
            if len(d) < 61 or not d.complete.fillna(False).all() or not (d.available_at <= at).all():
                valid = False; break
            close = d.close
            previous = close.shift(1)
            tr = pd.concat([d.high - d.low, (d.high - previous).abs(), (d.low - previous).abs()], axis=1).max(axis=1)
            row[symbol + '_atr14'] = float(tr.iloc[-14:].mean() / close.iloc[-1])
            if symbol not in ['SOXX', 'QQQ']:
                continue
            for n in [2, 10, 60]:
                row[f'{symbol}_ret{n}'] = float(close.iloc[-1] / close.iloc[-1 - n] - 1)
            for n in [5, 20, 60]:
                row[f'{symbol}_ma{n}_distance'] = float(close.iloc[-1] / close.iloc[-n:].mean() - 1)
            changes = close.diff()
            for n in [2, 5]:
                positive = changes.iloc[-n:].clip(lower=0).sum()
                absolute = changes.iloc[-n:].abs().sum()
                row[f'{symbol}_rsi{n}'] = float(positive / absolute) if absolute else .5
            row[symbol + '_efficiency20'] = float((close.iloc[-1] - close.iloc[-21]) / max(changes.iloc[-20:].abs().sum(), 1e-12))
            row[symbol + '_prior_gap'] = float(d.open.iloc[-1] / close.iloc[-2] - 1)
            row[symbol + '_close_position'] = float((close.iloc[-1] - d.low.iloc[-1]) / max(d.high.iloc[-1] - d.low.iloc[-1], 1e-12))
        if valid:
            output.append({'decision_at': at, **row})
    return pd.DataFrame(output).set_index('decision_at').sort_index()


def entry_rule(row, symbol, risk):
    if risk == 'expiry14':
        return None
    return {'id': risk, 'stop_fraction': float(np.clip(2 * row[symbol + '_atr14'], .06, .18)),
            'breakeven_trigger': .02 if risk == 'atr_protect' else None}


def outcome(market, row, symbol, at, risk, multiple=1):
    rule = entry_rule(row, symbol, risk)
    label = risk_label(market, symbol, at, rule, multiple) if rule else core.label_entry(market, symbol, at, multiple)
    label = {**label, 'risk_rule': risk, 'label_available_at': None, 'held_sessions': None}
    if label.get('exit_at') is not None:
        exit_day = label['exit_at'].tz_convert(core.ET).tz_localize(None).normalize()
        label['held_sessions'] = int(((market.sessions >= label['entry_date']) & (market.sessions <= exit_day)).sum())
    if label.get('complete_window') and np.isfinite(label.get('net_return', np.nan)):
        bars = market.bars[symbol]
        window = bars[(bars.start_at >= label['entry_at']) & (bars.end_at <= label['expiry_at']) & (bars.session_type == 'regular')]
        label['label_available_at'] = max(label['expiry_at'] + pd.Timedelta(seconds=1), window.available_at.max())
    return label


def columns(features):
    return [c for c in features if c not in ['date', 'clock']]


def context_columns(features):
    return [c for c in columns(features) if c.startswith(('SOXX_', 'QQQ_', 'sector_relative_'))]


def hgb(loss='squared_error', quantile=None):
    return HistGradientBoostingRegressor(loss=loss, quantile=quantile, max_iter=80, max_leaf_nodes=7,
        min_samples_leaf=20, learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=814)


def fit_predict(features, train, table, row):
    """Four models, one shared historical training slice and no current outcomes."""
    selected = context_columns(features)
    X = features.loc[train, selected].to_numpy(dtype=float)
    current = row[selected].to_numpy(dtype=float).reshape(1, -1)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X); z = scaler.transform(current)
    Y = np.column_stack([table[s].loc[train, 'net_return'].to_numpy(dtype=float) for s in core.ETFS])
    if (Y <= -1).any():
        raise ValueError('Invalid limited-loss long ETF trade return')
    logs = np.log1p(Y)
    advantage = Ridge(alpha=20).fit(Z, logs[:, 0] - logs[:, 1]).predict(z)[0]
    center = float(logs.mean())
    predictions = {'ridge_pair': np.array([center + advantage / 2, center - advantage / 2])}
    distances = np.linalg.norm(Z - z, axis=1)
    near = np.argsort(distances, kind='stable')[:40]
    weights = 1 / (distances[near] + .1); weights /= weights.sum()
    predictions['analog_tail'] = (weights[:, None] * (logs[near] + .25 * np.minimum(logs[near], 0))).sum(axis=0)
    predictions['boosted_log'] = np.array([hgb().fit(X, logs[:, i]).predict(current)[0] for i in range(2)])
    quantiles = []
    for i in range(2):
        q = sorted(hgb('quantile', quantile).fit(X, Y[:, i]).predict(current)[0] for quantile in [.2, .5, .8])
        quantiles.append(q)
    quantiles = np.array(quantiles)
    predictions['quantile_tail'] = .25 * quantiles[:, 0] + .5 * quantiles[:, 1] + .25 * quantiles[:, 2]
    return predictions, {'quantile20_50_80': quantiles.tolist(), 'analog_max_distance': float(distances[near].max())}


def corrected_choice(raw, history, table, eligible, protocol):
    calibration = eligible.intersection(history.index)
    if len(calibration):
        finite = np.isfinite(history.loc[calibration, ['SOXL_raw', 'SOXS_raw']].to_numpy(dtype=float)).all(axis=1)
        calibration = calibration[finite]
    calibration = calibration[-protocol['calibration_window_dates']:]
    adjustments = []
    for i, symbol in enumerate(core.ETFS):
        if len(calibration):
            actual = np.log1p(table[symbol].loc[calibration, 'net_return'])
            adjustments.append(float((actual - history.loc[calibration, symbol + '_raw']).mean()))
        else:
            adjustments.append(None)
    ready = len(calibration) >= protocol['minimum_calibration_dates']
    adjusted = raw + np.array(adjustments, dtype=float)
    chosen = int(np.argmax(adjusted)) if ready else 0
    action = core.ETFS[chosen] if ready and adjusted[chosen] > 0 else 'CASH'
    return action, {'calibration_dates': len(calibration), 'calibration_members': list(calibration),
                    'correction': adjustments, 'adjusted_scores': adjusted.tolist(), 'ready': ready}


def select_expert(histories, outcomes, features, cutoff, protocol):
    trials = []
    for expert in EXPERTS:
        history = histories[expert]
        risk = expert.split('__')[1]
        tables = outcomes[risk]
        eligible = eligible_index(features, tables, cutoff)
        dates = eligible.intersection(history.index)[-protocol['selector_window_dates']:]
        samples = []; signals = 0; net_logs = []
        for at in dates:
            signal = history.loc[at, 'action']
            if signal == 'CASH':
                samples.append(0.); continue
            label = tables[signal].loc[at]
            log_return = float(np.log1p(label.net_return))
            samples.append(log_return / max(int(label.held_sessions), 1))
            net_logs.append(log_return); signals += 1
        sample = np.array(samples)
        mean = float(sample.mean()) if len(sample) else 0.
        penalty = protocol['selector_uncertainty_penalty'] * float(sample.std(ddof=1)) / np.sqrt(max(len(sample) / 5, 1)) if len(sample) > 1 else 0.
        score = mean - penalty
        passes = len(dates) == protocol['selector_window_dates'] and signals >= protocol['selector_minimum_signals'] and bool(net_logs) and np.mean(net_logs) > 0 and score > 0
        trials.append({'expert': expert, 'score': score, 'mean_daily_log_rate': mean, 'penalty': penalty,
                       'dates': len(dates), 'signals': signals, 'qualified': bool(passes),
                       'last_history_at': dates[-1] if len(dates) else None,
                       'last_label_available': max(tables[s].loc[dates, 'label_available_at'].max() for s in core.ETFS) if len(dates) else None,
                       'history_sha256': digest({'dates': list(dates), 'samples': samples})})
    qualified = [t for t in trials if t['qualified']]
    selected = sorted(qualified, key=lambda x: (-x['score'], x['expert']))[0]['expert'] if qualified else None
    return selected, trials
