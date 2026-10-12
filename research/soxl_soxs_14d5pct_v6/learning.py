"""Date-weighted daily fitting and selection from observed shadow capital."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from path_engine import core, day_cutoff


def feature_columns(features):
    return [c for c in features if c not in ['date', 'clock']]


def eligible_dates(features, tables, cutoff, protocol):
    eligible = []
    for date, group in features.groupby('date'):
        if date >= cutoff.tz_convert(core.ET).tz_localize(None).normalize():
            continue
        valid = True
        for symbol in core.ETFS:
            t = tables[symbol].reindex(group.index)
            if not ((t.label_available_at < cutoff) & t.complete_window.eq(True) & np.isfinite(t.net_return)).all():
                valid = False; break
        if valid:
            eligible.append(date)
    return pd.DatetimeIndex(eligible[-protocol['training_window_dates']:])


def fit_daily(features, tables, at, protocol):
    cutoff = day_cutoff(at)
    dates = eligible_dates(features, tables, cutoff, protocol)
    index = features.index[features.date.isin(dates)]
    record = {'training_cutoff': cutoff, 'training_dates': len(dates), 'training_rows': len(index),
              'first_date': dates[0] if len(dates) else None, 'last_date': dates[-1] if len(dates) else None,
              'last_label_available': max(tables[s].loc[index, 'label_available_at'].max() for s in core.ETFS) if len(index) else None,
              'members': list(index), 'ready': len(dates) >= protocol['minimum_training_dates']}
    if not record['ready']:
        return None, record, index
    columns = feature_columns(features)
    X = features.loc[index, columns]
    counts = features.loc[index].groupby('date').size()
    weights = features.loc[index, 'date'].map(1 / counts).to_numpy(dtype=float)
    assert np.isclose(weights.sum(), len(dates))
    fitted = {name: {} for name in ['ridge_log', 'boosted_log']}
    for symbol in core.ETFS:
        y = np.log1p(tables[symbol].loc[index, 'net_return'].to_numpy(dtype=float))
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=20))
        ridge.fit(X, y, standardscaler__sample_weight=weights, ridge__sample_weight=weights)
        boost = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
                      learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=814)
        boost.fit(X, y, sample_weight=weights)
        fitted['ridge_log'][symbol] = ridge; fitted['boosted_log'][symbol] = boost
    record['sample_weight_sum'] = float(weights.sum())
    return (fitted, columns), record, index


def predict(fit, name, row, history, tables, eligible, protocol):
    models, columns = fit
    X = pd.DataFrame([row])[columns]
    old = eligible.intersection(history.index)
    if len(old):
        old = old[history.loc[old, 'clock'].eq(row.clock).to_numpy()]
        finite = np.isfinite(history.loc[old, ['SOXL_raw', 'SOXS_raw']].to_numpy(dtype=float)).all(axis=1)
        old = old[finite][-protocol['calibration_window_dates']:]
    ready = len(old) >= protocol['minimum_calibration_dates']
    result = {'action': 'CASH', 'clock': row.clock, 'calibration_dates': len(old), 'calibration_members': list(old), 'ready': ready}
    adjusted = []
    for symbol in core.ETFS:
        raw = float(models[name][symbol].predict(X)[0])
        correction = float((np.log1p(tables[symbol].loc[old, 'net_return']) - history.loc[old, symbol + '_raw']).mean()) if len(old) else None
        result[symbol + '_raw'] = raw; result[symbol + '_correction'] = correction
        result[symbol + '_adjusted'] = raw + correction if correction is not None else None
        adjusted.append(result[symbol + '_adjusted'])
    if ready:
        winner = int(np.argmax(adjusted))
        if adjusted[winner] > 0:
            result['action'] = core.ETFS[winner]
    return result


def select_shadow(books, date, protocol):
    cutoff = pd.Timestamp(date).tz_localize(core.ET)
    candidates = []
    for expert, book in books.items():
        equity, trades = book[0], book[1]
        past = equity[equity.index < date].tail(protocol['selector_sessions'] + 1)
        row = {'expert': expert, 'dates': len(past), 'qualified': False}
        if len(past) == protocol['selector_sessions'] + 1 and past.mark_complete.all():
            curve = past.equity.to_numpy(dtype=float)
            daily = np.diff(np.log(curve))
            score = float(daily.mean() - protocol['selector_penalty'] * daily.std(ddof=1) / np.sqrt(len(daily)))
            drawdown = float((curve / np.maximum.accumulate(curve) - 1).min())
            end_return = float(curve[-1] / curve[0] - 1)
            recent = float(curve[-1] / curve[-22] - 1)
            closed = 0
            if len(trades):
                closed = int(((trades.entry_date >= past.index[1]) & (trades.entry_date < date) & trades.exit_at.notna() &
                              (pd.to_datetime(trades.exit_at, utc=True) < cutoff)).sum())
            row.update({'score': score, 'return63': end_return, 'return21': recent, 'drawdown': drawdown,
                        'closed_trades': closed, 'last_mark_date': past.index[-1],
                        'qualified': bool(score > 0 and end_return > 0 and recent > 0 and
                             drawdown >= -protocol['selector_max_drawdown'] and closed >= protocol['selector_min_closed_trades'])})
        candidates.append(row)
    good = [x for x in candidates if x['qualified']]
    chosen = sorted(good, key=lambda x: (-x['score'], x['expert']))[0]['expert'] if good else None
    return chosen, candidates
