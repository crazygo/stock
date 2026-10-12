"""Option inputs affect only the payoff-weighted direction model; C/M unchanged."""
from pathlib import Path
import sys
import copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v35'))
from financial_learning import *
from financial_learning import columns as source_columns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from option_features import option_columns, load_raw, build_features, calendar

SOURCE = Path(__file__).resolve().parent
VARIANTS = ['options', 'delayed', 'baseline']
ROUTES = ['options_band', 'options_point', 'delayed_band', 'baseline_band',
          'source_v35', 'premarket_only', 'cash', 'double_cost']


def columns(clock, kind='baseline'):
    return source_columns(clock, 'full')


def q_columns(clock, kind):
    return columns(clock) + ([] if kind == 'baseline' else option_columns(2 if kind == 'options' else 5))


def fit_bases(panel, at):
    same = panel[panel.clock == panel.loc[at, 'clock']]
    dates, fb = training_dates(same, day_cutoff(at))
    linear, linear_info = fit_clock(same, dates['early'], columns(panel.loc[at, 'clock']))
    models, infos = {'baseline': linear}, {'baseline': linear_info}
    for kind in ['options', 'delayed']:
        model, info = copy.deepcopy(linear), copy.deepcopy(linear_info)
        cols = q_columns(panel.loc[at, 'clock'], kind)
        info['q_columns'] = cols
        if model is not None:
            rows = same.loc[pd.to_datetime(info['members'], utc=True)]
            weights = np.asarray(info['parent_weights'])
            cw = np.asarray(info['classification_weights'])
            scaler = StandardScaler().fit(rows[cols].to_numpy(), sample_weight=weights)
            Z = scaler.transform(rows[cols].to_numpy())
            model.update(q_columns=cols, q_scaler=scaler)
            model['heads'].pop('q', None)
            info['heads'].pop('q', None)
            info.update(q_center=scaler.mean_.tolist(), q_scale=scaler.scale_.tolist())
            y = (rows.D.to_numpy() > 0).astype(int)
            if len(np.unique(y[cw > 0])) == 2:
                h = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
                h.fit(Z, y, sample_weight=cw)
                assert h.n_iter_.max() < 2000
                model['heads']['q'] = h
                info['heads']['q'] = {'coefficients': h.coef_.tolist(), 'intercept': h.intercept_.tolist()}
        models[kind], infos[kind] = model, info
    out, raws = {}, {}
    for kind in VARIANTS:
        model = models[kind]
        if kind == 'baseline':
            raw = predict(model, same.loc[[at]])
        elif model is None or not same.loc[at, 'feature_ready']:
            raw = {}
        else:
            x = model['scaler'].transform(same.loc[[at], model['columns']].to_numpy())
            c = float(model['heads']['C'].predict(x)[0])
            m = max(0., float(model['heads']['M'].predict(x)[0]))
            q = model['q']
            if 'q' in model['heads']:
                z = model['q_scaler'].transform(same.loc[[at], model['q_columns']].to_numpy())
                q = float(model['heads']['q'].predict_proba(z)[0, 1])
            d = m * (2*q-1)
            raw = {'C': c, 'M': m, 'D': d, 'q': q, 'SOXL': c+d, 'SOXS': c-d}
        out[kind] = {'fit': infos[kind], 'feedback': fb, 'raw': raw}
        raws[kind] = np.array([raw[s] for s in core.ETFS]) if raw else None
    return models, out, raws


def common_history(histories):
    return {t: histories['baseline'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS)
            else None for t in sorted(histories['baseline'])}


def fit_calibration(panel, histories, reg, at):
    parents, fb = pool(reg, common_history(histories), at)
    ids = [t for p in parents for t in p['members']]
    n = len(parents)
    d = {'date': panel.loc[at, 'date'], 'cutoff': day_cutoff(at), 'feedback': fb,
         'parents': parents, 'members': ids, 'ready': n >= 60, 'variants': {}}
    if n < 60:
        return d
    dw = 2.**(-np.arange(n)[::-1]/63.)
    dw *= n/dw.sum()
    groups = np.repeat(np.arange(n), [len(p['members']) for p in parents])
    w = np.concatenate([np.repeat(dw[j]/len(p['members']), len(p['members'])) for j,p in enumerate(parents)])
    truth = panel.loc[ids, ['C', 'D']].to_numpy(float)
    Ytrue = np.column_stack([truth[:,0]+truth[:,1], truth[:,0]-truth[:,1]])
    counts, starts, seed = block_counts(n, at, 10)
    d.update(day_weights=dw, row_weights=w, row_groups=groups, block_starts=starts, seed=seed)
    for kind in VARIANTS:
        X = encode(panel, histories[kind], ids)
        Y = Ytrue-X[:,:2]
        d['variants'][kind] = {'point_model': fit_map(X,Y,w,'partial',n),
            'bootstrap_models': [fit_map(X,Y,w*c[groups],'partial',n) for c in counts]}
    return d


def events(at, date, clock, estimates, fb, old):
    primary = decide(estimates['options'], fb, True)
    cash = {'ready': True, 'action': 'CASH', 'allocation': 0.}
    out = {'options_band': primary, 'options_point': decide(estimates['options'],fb,False),
           'delayed_band': decide(estimates['delayed'],fb,True),
           'baseline_band': decide(estimates['baseline'],fb,True), 'source_v35': old,
           'premarket_only': primary if clock == CLOCKS[0] else cash, 'cash': cash, 'double_cost': primary}
    return {k: {**e,'at': at,'date': date} for k,e in out.items()}


def verified_source(phase, p):
    root = core.ROOT/p['source_runs'][phase]
    if phase == 'development':
        d = read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
        want = {str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),
                ('training_records_sha256','training'),('daily_records_sha256','daily')] for f,h in d[field].items()}
    else:
        d = read(core.ROOT/'research/soxl_soxs_14d5pct_v39/evidence/delivery_manifest.json')['runs']['financial_extension_20261013']
        assert sha(root/'extension_receipt.json') == d['top_level_sha256']['extension_receipt.json']
        want = read(root/'extension_receipt.json')['files_sha256']
    for f,h in want.items():
        assert sha(root/f) == h
    v = read(root/'verification.json')
    assert v['passed'] and v['results_sha256'] == sha(root/'results.json')
    return root, want


def issue(panel, at, raws, daily):
    out = {}
    for kind in VARIANTS:
        raw = raws[kind]
        ready = raw is not None and panel.loc[at,'feature_ready'] and daily['ready']
        info = {'ready': bool(ready), 'point': None, 'lower': None}
        if ready:
            x = encode(panel, {at:raw}, [at])[0]
            m = daily['variants'][kind]
            point = raw+forecast(m['point_model'],x)[0]
            draws = np.array([raw+forecast(z,x)[0] for z in m['bootstrap_models']])
            radius = max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind] = info
    return out
