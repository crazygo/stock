"""Frozen, local-only daily leader pilot. No imports from existing model projects."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALPHA = 20.0
OWN = ['r1', 'r3', 'r5', 'r20', 'rv5', 'rv20', 'range', 'relvol']
LEAD = ['r1', 'r3', 'r5', 'rv5', 'range', 'relvol']
MARKET = ['r1', 'r5', 'rv5']


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def same_company(a, b):
    return a == b or {a, b} <= {'GOOG', 'GOOGL'}


def load_panel():
    cal = json.loads((ROOT / 'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    sessions = [s for s in cal['sessions'] if '2026-01-01' <= s['session_date'] <= '2026-09-23']
    idx = pd.Index([s['session_date'] for s in sessions], name='date')
    calendar = {s['session_date']: s for s in sessions}
    meta = json.loads((ROOT / 'market_data/universe/qqq_retrospective_v1.json').read_text())
    candidates = sorted(m['symbol'] for m in meta['members'] if m['role'] == 'candidate')
    proxies = {m['symbol']: m['industry_proxy'] for m in meta['members'] if m['role'] == 'candidate'}
    symbols = sorted(set(candidates) | {'QQQ'} | set(proxies.values()))
    features, labels, audit, daily = {}, {}, {}, {}
    for symbol in symbols:
        path = ROOT / f'market_data/us_5m/{symbol}/2026.parquet'
        bars = pd.read_parquet(path)
        assert set(bars.price_basis) == {'NONE'}
        bars = bars[bars.session_type.eq('regular') & bars.session_date.isin(idx)]
        bars = bars.sort_values('end_at')
        records, rejected = [], []
        for date, g in bars.groupby('session_date'):
            s = calendar[date]
            op, cl = pd.Timestamp(s['open_at']), pd.Timestamp(s['close_at'])
            expected = pd.date_range(op + pd.Timedelta(minutes=5), cl, freq='5min')
            ends = pd.DatetimeIndex(pd.to_datetime(g.end_at, utc=True))
            starts = pd.DatetimeIndex(pd.to_datetime(g.start_at, utc=True))
            price = g[['open', 'high', 'low', 'close']].to_numpy(float)
            valid = (len(g) == len(expected) and ends.equals(expected)
                     and starts.equals(expected - pd.Timedelta(minutes=5))
                     and np.isfinite(price).all() and (price > 0).all()
                     and (g.high >= g[['open', 'close', 'low']].max(axis=1)).all()
                     and (g.low <= g[['open', 'close', 'high']].min(axis=1)).all()
                     and np.isfinite(g.volume).all() and (g.volume >= 0).all()
                     and (pd.to_datetime(g.available_at, utc=True) <= cl + pd.Timedelta(minutes=1)).all())
            if not valid:
                rejected.append(date)
                continue
            records.append(dict(date=date, open=g.open.iloc[0], high=g.high.max(),
                                low=g.low.min(), close=g.close.iloc[-1], volume=g.volume.sum()))
        d = pd.DataFrame(records).set_index('date').reindex(idx)
        actions = pd.read_parquet(ROOT / f'market_data/corporate_actions/{symbol}.parquet')
        factor = pd.Series(1.0, index=idx)
        complex_dates = []
        for _, a in actions.iterrows():
            date = str(a.ex_div_date)
            if date not in idx:
                continue
            is_complex = any(pd.notna(a.get(c)) and float(a[c]) != 0 for c in
                             ['spin_off_ratio', 'per_share_div_ratio', 'per_share_trans_ratio', 'allotment_ratio', 'stk_spo_ratio'])
            if is_complex:
                complex_dates.append(date)
            elif pd.notna(a.get('split_ratio')):
                assert float(a.split_ratio) > 0
                factor.loc[date] *= float(a.split_ratio)
        cumulative = factor.cumprod()
        adjusted = d.copy()
        for c in ['open', 'high', 'low', 'close']:
            adjusted[c] = d[c] / cumulative
        adjusted['volume'] = d.volume * cumulative
        d_feature = adjusted.copy()
        for date in complex_dates:
            d_feature.loc[date, :] = np.nan
        f = pd.DataFrame(index=idx)
        r1 = d_feature.close.pct_change(fill_method=None)
        for h in [1, 3, 5, 20]:
            f[f'r{h}'] = ((1 + r1).rolling(h, min_periods=h).apply(np.prod, raw=True) - 1) * 100
        for h in [5, 20]:
            f[f'rv{h}'] = r1.rolling(h, min_periods=h).std() * 100
        f['range'] = (d_feature.high / d_feature.low - 1) * 100
        f['relvol'] = d_feature.volume / d_feature.volume.shift(1).rolling(20, min_periods=20).mean()
        for date in complex_dates:
            j = idx.get_loc(date)
            f.iloc[j:j + 21] = np.nan
        f = f.replace([np.inf, -np.inf], np.nan)
        y = pd.Series(np.nan, index=idx, name='target')
        for i in range(len(idx) - 5):
            future = adjusted.iloc[i + 1:i + 6]
            if future.isna().any(axis=None) or any(idx[i + 1] <= x <= idx[i + 5] for x in complex_dates):
                continue
            y.iloc[i] = (future.high.max() / future.open.iloc[0] - 1) * 100
        assert y.dropna().ge(-1e-10).all()
        features[symbol], labels[symbol], daily[symbol] = f, y, adjusted
        audit[symbol] = dict(valid_days=int(d.close.notna().sum()), rejected_days=rejected,
                             complex_action_dates=complex_dates,
                             split_dates={k: float(v) for k, v in factor.items() if v != 1},
                             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ends = pd.Series([idx[i + 5] if i + 5 < len(idx) else None for i in range(len(idx))], index=idx)
    return idx, candidates, proxies, features, labels, ends, audit, daily


def fit_predict(x, y, z):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-10, 1, std)
    xs = (x - mean) / std
    beta = np.linalg.solve(xs.T @ xs + ALPHA * np.eye(x.shape[1]), xs.T @ (y - y.mean()))
    return np.maximum(0, y.mean() + ((z - mean) / std) @ beta)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def bootstrap_matrix(d, block, n=5000):
    rng = np.random.default_rng(20260927 + block)
    t = len(d)
    starts = rng.integers(0, t, (n, math.ceil(t / block)))
    positions = ((starts[:, :, None] + np.arange(block)) % t).reshape(n, -1)[:, :t]
    values = d.to_numpy(float)
    # Missing target days keep their original masks; all symbols share sampled dates.
    boot = np.nanmean(values[positions], axis=1)
    means = np.nanmean(values, axis=0)
    p = (1 + np.sum(boot - means >= means, axis=0)) / (n + 1)
    return means, np.nanquantile(boot, [0.025, 0.975], axis=0), p, boot


def holm(p):
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def main():
    idx, candidates, proxies, features, labels, ends, audit, daily = load_panel()
    dump('data_audit.json', audit)
    print('loaded', len(candidates), 'candidate stocks;', len(idx), 'calendar sessions', flush=True)
    lead_matrices = {(a, lag): features[a][LEAD].shift(lag).to_numpy(float)
                     for a in candidates for lag in [0, 1, 2]}
    frozen, skipped, material = {}, {}, {}
    # No validation targets are used in this loop.
    for b in candidates:
        markets = ['QQQ'] + ([proxies[b]] if proxies[b] != 'QQQ' else [])
        own = features[b][OWN].to_numpy(float)
        market = np.column_stack([features[m][MARKET].to_numpy(float) for m in markets])
        base = np.column_stack([own, market])
        y = labels[b].to_numpy(float)
        # Common 22-session warmup allows every frozen leader lag (0/1/2)
        # its full 20-day feature history on the same training dates.
        valid = np.isfinite(base).all(axis=1) & np.isfinite(y) & (np.arange(len(idx)) >= 22)
        train = valid & (idx <= '2026-05-15') & (ends.fillna('9999') < '2026-05-26')
        select = valid & (idx >= '2026-05-26') & (idx <= '2026-06-30')
        if train.sum() < 50 or select.sum() < 20:
            skipped[b] = dict(reason='insufficient training/selection history', train=int(train.sum()), select=int(select.sum()))
            continue
        baseline = fit_predict(base[train], y[train], base[select])
        base_mae = mae(y[select], baseline)
        ranked = []
        for (a, lag), l in lead_matrices.items():
            if same_company(a, b) or not np.isfinite(l[train | select]).all():
                continue
            x = np.column_stack([base, l])
            loss = mae(y[select], fit_predict(x[train], y[train], x[select]))
            ranked.append((loss, a, lag))
        if not ranked:
            skipped[b] = dict(reason='no complete-history leader')
            continue
        ranked.sort()
        loss, a, lag = ranked[0]
        fixed_a = sorted({x[1] for x in ranked if x[2] == 0})[0]
        frozen[b] = dict(leader=a, lag=lag, fixed_control=fixed_a, candidate_pairs=len(ranked),
                         train_n=int(train.sum()), select_n=int(select.sum()),
                         selection_baseline_mae=base_mae, selection_leader_mae=loss,
                         selection_improvement_pp=base_mae-loss)
        material[b] = own, market, base, y, valid
    dump('frozen_pairs.json', frozen)
    dump('skipped.json', skipped)
    print('frozen pairs', len(frozen), 'skipped', len(skipped), flush=True)
    predictions = []
    for b, spec in frozen.items():
        own, market, base, y, valid = material[b]
        l = lead_matrices[spec['leader'], spec['lag']]
        fixed = lead_matrices[spec['fixed_control'], 0]
        fit = valid & (idx <= '2026-06-30') & (ends.fillna('9999') < '2026-07-13')
        # Require leader history at fit time; selection eligibility already enforces most rows.
        fit &= np.isfinite(l).all(axis=1) & np.isfinite(fixed).all(axis=1)
        test = valid & (idx >= '2026-07-13') & (idx <= '2026-09-16')
        assert (ends[fit] < idx[test].min()).all()
        if test.sum() < 30:
            skipped[b] = dict(reason='insufficient validation target coverage', test=int(test.sum()))
            continue
        predictions_by_model = {}
        for name, x in [('own', own), ('base', base), ('leader', np.column_stack([base, l])),
                        ('leader_market', np.column_stack([market, l])), ('fixed', np.column_stack([base, fixed]))]:
            test_good = test & np.isfinite(x).all(axis=1)
            p = fit_predict(base[fit], y[fit], base[test])
            p[np.isfinite(x[test]).all(axis=1)] = fit_predict(x[fit], y[fit], x[test_good])
            predictions_by_model[name] = p
        for k, t in enumerate(np.where(test)[0]):
            predictions.append(dict(date=idx[t], label_end=ends.iloc[t], symbol=b, leader_symbol=spec['leader'],
                                    lag=spec['lag'], target=float(y[t]), leader_available=bool(np.isfinite(l[t]).all()),
                                    **{name: float(p[k]) for name, p in predictions_by_model.items()}))
    dump('skipped.json', skipped)
    df = pd.DataFrame(predictions)
    df.to_parquet(HERE / 'predictions.parquet', index=False, compression='zstd', compression_level=7)
    for model in ['own', 'base', 'leader', 'leader_market', 'fixed']:
        df[f'err_{model}'] = abs(df.target - df[model])
    df['improvement'] = df.err_base - df.err_leader
    matrix = df.pivot(index='date', columns='symbol', values='improvement').sort_index()
    means, ci, p, _ = bootstrap_matrix(matrix, 10)
    adj = holm(p)
    per_stock = []
    for j, b in enumerate(matrix.columns):
        g = df[df.symbol.eq(b)]
        early = g[g.date <= '2026-08-14'].improvement.mean()
        late = g[g.date >= '2026-08-17'].improvement.mean()
        per_stock.append(dict(symbol=b, **frozen[b], test_n=len(g),
                              base_mae=float(g.err_base.mean()), leader_mae=float(g.err_leader.mean()),
                              improvement_pp=float(means[j]), ci95_pp=ci[:, j].tolist(),
                              p_one_sided=float(p[j]), p_holm=float(adj[j]),
                              early_improvement_pp=float(early), late_improvement_pp=float(late)))
    dump('per_stock.json', per_stock)
    daily_errors = df.groupby('date')[[f'err_{m}' for m in ['own', 'base', 'leader', 'leader_market', 'fixed']]].mean()
    day_diff = (daily_errors.err_base - daily_errors.err_leader).to_frame('overall')
    overall_mean, overall_ci, overall_p, _ = bootstrap_matrix(day_diff, 10)
    _, ci5, _, _ = bootstrap_matrix(day_diff, 5)
    summary = dict(stocks=int(df.symbol.nunique()), rows=len(df), validation_days=int(df.date.nunique()),
                   validation_first=df.date.min(), validation_last=df.date.max(),
                   latest_label_end=df.label_end.max(),
                   total_candidates_tested=int(sum(x['candidate_pairs'] for x in frozen.values())),
                   mae_daily_equal_weight={m:float(daily_errors[f'err_{m}'].mean()) for m in ['own', 'base', 'leader', 'leader_market', 'fixed']},
                   improvement_pp=float(overall_mean[0]), ci95_block10_pp=overall_ci[:,0].tolist(),
                   ci95_block5_pp=ci5[:,0].tolist(), p_one_sided=float(overall_p[0]),
                   relative_mae_reduction=float(overall_mean[0] / daily_errors.err_base.mean()),
                   early_improvement_pp=float(day_diff.loc[:'2026-08-14'].overall.mean()),
                   late_improvement_pp=float(day_diff.loc['2026-08-17':].overall.mean()),
                   positive_stocks=int(sum(x['improvement_pp']>0 for x in per_stock)),
                   both_periods_positive=int(sum(x['early_improvement_pp']>0 and x['late_improvement_pp']>0 for x in per_stock)),
                   holm_pass_stocks=[x['symbol'] for x in per_stock if x['p_holm']<0.05 and x['early_improvement_pp']>0 and x['late_improvement_pp']>0],
                   leader_fallback_rows=int((~df.leader_available).sum()),
                   protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
