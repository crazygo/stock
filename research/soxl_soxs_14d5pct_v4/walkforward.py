"""Daily, as-of fitting and prequential return-bias correction (no probabilities)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

V3 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v3'
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from risk import core, risk_label


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def feature_columns(features):
    return [c for c in features.columns if c not in {'date', 'clock'}]


def day_cutoff(at):
    return core.timestamp(at).tz_convert(core.ET).normalize().tz_convert('UTC')


def make_labels(market, features, rule, multiple=1):
    """Keep unknown labels; release completed windows symmetrically for wins/losses."""
    labels = []
    for at in features.index:
        for symbol in core.ETFS:
            label = risk_label(market, symbol, at, rule, multiple)
            available = None
            if label.get('complete_window') and np.isfinite(label.get('net_return', np.nan)):
                bars = market.bars[symbol]
                window = bars[(bars.start_at >= label['entry_at']) &
                              (bars.end_at <= label['expiry_at']) & (bars.session_type == 'regular')]
                available = max(label['expiry_at'] + pd.Timedelta(seconds=1), window.available_at.max())
            labels.append({**label, 'label_available_at': available})
    return labels


def label_tables(labels):
    tables = {}
    for symbol in core.ETFS:
        table = pd.DataFrame([x for x in labels if x['symbol'] == symbol])
        table['decision_at'] = pd.to_datetime(table.decision_at, utc=True)
        table['label_available_at'] = pd.to_datetime(table.label_available_at, utc=True)
        table['net_return'] = pd.to_numeric(table.net_return, errors='coerce')
        tables[symbol] = table.set_index('decision_at').sort_index()
    return tables


def eligible_index(features, tables, cutoff):
    index = features.index[features.index < cutoff]
    for symbol in core.ETFS:
        t = tables[symbol].reindex(index)
        valid = (t.label_available_at < cutoff) & t.complete_window.eq(True) & np.isfinite(t.net_return)
        index = index[valid.to_numpy()]
    return index


def fit_day(features, tables, forecasts, at, protocol):
    """No case-month list is accepted here; all access is bounded by today's cutoff.

    `forecasts` contains immutable raw forecasts emitted by earlier daily models.
    Current training may learn a matured past calibration outcome, but the raw
    forecast used to measure its error was made before that outcome existed.
    """
    cutoff = day_cutoff(at)
    eligible = eligible_index(features, tables, cutoff)
    train = eligible[-protocol['training_window_dates']:]
    columns = feature_columns(features)
    record = {'date': str(cutoff.tz_convert(core.ET).date()), 'training_cutoff': cutoff,
              'eligible_dates': len(eligible), 'training_dates': len(train),
              'feature_columns': columns, 'status': 'insufficient_training'}
    if len(train) < protocol['minimum_training_dates']:
        return None, record
    # An API caller cannot inject today's or a future forecast into calibration.
    calibration = eligible.intersection(forecasts.index)
    if len(calibration):
        history = forecasts.loc[calibration]
        valid = pd.to_datetime(history.forecast_at, utc=True) == history.index
        valid &= pd.to_datetime(history.training_cutoff, utc=True) <= history.index
        for s in core.ETFS:
            valid &= np.isfinite(history[f'{s}_raw_ev'])
        calibration = calibration[valid.to_numpy()][-protocol['calibration_window_dates']:]
    input_records = []
    models = {}; corrections = {}; calibration_details = {}
    for symbol in core.ETFS:
        y = tables[symbol].loc[train, 'net_return']
        models[symbol] = ExtraTreesRegressor(**protocol['forest']).fit(features.loc[train, columns], y)
        if len(calibration):
            residual = (tables[symbol].loc[calibration, 'net_return'] - forecasts.loc[calibration, f'{symbol}_raw_ev'])
            corrections[symbol] = float(residual.mean())
            calibration_details[symbol] = {'mean_residual': corrections[symbol],
                                           'raw_rmse': float(np.sqrt(np.mean(residual ** 2)))}
        else:
            corrections[symbol] = None
            calibration_details[symbol] = {'mean_residual': None, 'raw_rmse': None}
        input_records.append({'symbol': symbol, 'y': y.tolist(),
                              'available_at': tables[symbol].loc[train, 'label_available_at'].tolist()})
    ready = len(calibration) >= protocol['minimum_calibration_dates']
    max_available = max(tables[s].loc[train, 'label_available_at'].max() for s in core.ETFS)
    calibration_available = max(tables[s].loc[calibration, 'label_available_at'].max() for s in core.ETFS) if len(calibration) else None
    assert max_available < cutoff and (calibration_available is None or calibration_available < cutoff)
    record.update({'status': 'ready' if ready else 'insufficient_calibration',
                   'train_first': train[0], 'train_last': train[-1],
                   'train_last_label_available': max_available,
                   'calibration_dates': len(calibration),
                   'calibration_first': calibration[0] if len(calibration) else None,
                   'calibration_last': calibration[-1] if len(calibration) else None,
                   'calibration_last_label_available': calibration_available,
                   'corrections': calibration_details,
                   'training_input_sha256': digest({'dates': list(train), 'X': features.loc[train, columns].to_numpy().tolist(), 'labels': input_records}),
                   'calibration_input_sha256': digest({'dates': list(calibration),
                       'forecasts': forecasts.loc[calibration].to_dict('records') if len(calibration) else [],
                       'outcomes': {s: tables[s].loc[calibration, 'net_return'].tolist() for s in core.ETFS}})})
    record['model_id'] = digest({'protocol': protocol, 'registration': record})[:24]
    # Membership lists provide a direct causal audit without raw market prices.
    record['train_members'] = list(train)
    record['calibration_members'] = list(calibration)
    return {'models': models, 'corrections': corrections, 'columns': columns,
            'ready': ready, 'model_id': record['model_id'], 'training_cutoff': cutoff}, record


def forecast(bundle, row, at, protocol):
    at = core.timestamp(at)
    if bundle['training_cutoff'] > at:
        raise ValueError('A future model cannot forecast an earlier decision')
    X = pd.DataFrame([row])[bundle['columns']]
    out = {'forecast_at': at, 'training_cutoff': bundle['training_cutoff'],
           'model_id': bundle['model_id'], 'model_ready': bundle['ready'], 'action': 'CASH'}
    for symbol in core.ETFS:
        raw = float(bundle['models'][symbol].predict(X)[0])
        correction = bundle['corrections'][symbol]
        out[f'{symbol}_raw_ev'] = raw
        out[f'{symbol}_adjusted_ev'] = raw + correction if correction is not None else None
    if bundle['ready']:
        chosen = max(core.ETFS, key=lambda s: out[f'{s}_adjusted_ev'])
        if out[f'{chosen}_adjusted_ev'] > protocol['minimum_expected_return']:
            out['action'] = chosen
    return out
