"""Lagged option activity and volatility, using immutable local provider histories."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar

PANEL = ['NVDA', 'AMD', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'INTC']
SYMBOLS = PANEL + ['SOXX', 'QQQ']
HISTORY = ROOT / 'analysis/preopen_5d5pct_v6/R06/cache/history'
NAMES = ['balance', 'balance_change5', 'activity', 'active_fraction',
         'iv_hv', 'iv_change5', 'sector_market_balance', 'sector_market_iv_hv']


def option_columns(lag):
    return [f'option_lag{lag}_{n}' for n in NAMES]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw():
    raw, manifest = {}, {}
    for symbol in SYMBOLS:
        raw[symbol] = {}
        for family in ['options', 'volatility']:
            path = HISTORY / family / (symbol + '.parquet')
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text())
            assert sha(path) == meta['sha256'] and not meta['errors']
            f = pd.read_parquet(path)
            assert set(f.code) == {'US.' + symbol} and not f.source_day.duplicated().any()
            assert len(f) == meta['rows']
            raw[symbol][family] = f
            manifest[str(path.relative_to(ROOT))] = {
                'sha256': sha(path), 'metadata_path': str(meta_path.relative_to(ROOT)),
                'metadata_sha256': sha(meta_path), 'capture': meta,
            }
    return raw, manifest


def daily_measurements(raw, sessions):
    dates = pd.DatetimeIndex([s['session_date'] for s in sessions])
    by_symbol, coverage = {}, []
    for s in SYMBOLS:
        frames = {}
        for family, cols in [('options', ['call_volume', 'put_volume', 'option_volume']),
                             ('volatility', ['iv', 'hv'])]:
            f = raw[s][family].copy()
            f.index = pd.to_datetime(f.source_day)
            assert not f.index.duplicated().any()
            coverage.append({'symbol': s, 'family': family, 'rows': len(f),
                             'first': str(f.index.min().date()), 'last': str(f.index.max().date()),
                             'missing_dates': [str(d.date()) for d in dates.difference(f.index)],
                             'extra_dates': [str(d.date()) for d in f.index.difference(dates)]})
            frames[family] = f[cols].apply(pd.to_numeric, errors='coerce').reindex(dates)
        o, v = frames['options'], frames['volatility']
        total = o.call_volume + o.put_volume
        valid_o = (o.ge(0).all(axis=1) & np.isfinite(o).all(axis=1) & total.gt(0)
                   & total.eq(o.option_volume))
        valid_v = v.gt(0).all(axis=1) & np.isfinite(v).all(axis=1)
        balance = ((o.call_volume - o.put_volume) / total).where(valid_o)
        volume = total.where(valid_o)
        activity = np.log(volume / volume.shift(1).rolling(20, min_periods=20).mean())
        iv, hv = v.iv.where(valid_v), v.hv.where(valid_v)
        by_symbol[s] = pd.DataFrame({
            'balance': balance, 'balance_change5': balance - balance.shift(5),
            'activity': activity, 'active_fraction': activity.gt(0).astype(float).where(activity.notna()),
            'iv_hv': np.log(iv / hv), 'iv_change5': np.log(iv / iv.shift(5)),
        }, index=dates)
    daily = pd.DataFrame(index=dates)
    # All eight required. Missing inputs are never silently dropped from means.
    for col in NAMES[:6]:
        daily[col] = pd.concat([by_symbol[s][col] for s in PANEL], axis=1).mean(axis=1, skipna=False)
    daily['sector_market_balance'] = by_symbol['SOXX'].balance - by_symbol['QQQ'].balance
    daily['sector_market_iv_hv'] = by_symbol['SOXX'].iv_hv - by_symbol['QQQ'].iv_hv
    return daily, by_symbol, coverage


def build_features(features, raw, sessions):
    daily, by_symbol, coverage = daily_measurements(raw, sessions)
    dates = daily.index
    rows, receipts = [], []
    for day in sorted(set(features.date)):
        j = int(dates.searchsorted(day))
        assert j < len(dates) and dates[j] == day
        values, ready = {}, True
        receipt = {'date': str(day.date()), 'lags': {}}
        for lag in [2, 5]:
            k = j - lag
            x = daily.iloc[k].to_numpy() if k >= 0 else np.full(len(NAMES), np.nan)
            ok = bool(np.isfinite(x).all())
            values.update(dict(zip(option_columns(lag), x)))
            ready &= ok
            receipt['lags'][str(lag)] = {
                'ready': ok, 'latest_source_day': str(dates[k].date()) if k >= 0 else None,
                'assumed_available_day': str(day.date()),
                'source_days_21': [str(d.date()) for d in dates[max(0, k-20):k+1]] if k >= 0 else [],
                'values': dict(zip(NAMES, [float(v) if np.isfinite(v) else None for v in x])),
            }
        rows.append({'date': day, 'option_ready': ready, **values})
        receipts.append(receipt)
    out = features[['date']].join(pd.DataFrame(rows).set_index('date'), on='date').drop(columns='date')
    return out, daily, {'coverage': coverage, 'days': receipts,
                        'availability': 'source day + 2/5 official sessions at 00:00 ET, assumption only; no historical publication/receipt vintages'}
