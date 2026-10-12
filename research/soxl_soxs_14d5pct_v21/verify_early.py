"""Independent training equations, chronological admission and capital audit."""
import argparse
import gzip
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from early_training import core, ROUTES, BASE_COLUMNS, REGIME_COLUMNS, restrict_market
from verify_continuation import read, labels, key, near
from verify_feedback_v2 import terminal_audit, verify_asof_replays, independent_weights
from verify_allocation import book_check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_equations(panel, ids, at, model, info):
    w = 2. ** (-np.arange(len(ids))[::-1] / 126.)
    if len(w):
        w *= len(w) / w.sum()
    np.testing.assert_allclose(info['parent_weights'], w, atol=1e-12, rtol=1e-10)
    assert info['ready'] == (len(ids) >= 120)
    if len(ids) < 120:
        assert model is None
        return {}, 0
    columns = BASE_COLUMNS + REGIME_COLUMNS
    X = panel.loc[ids, columns].to_numpy(dtype=float)
    C = panel.loc[ids, 'C'].to_numpy(dtype=float); D = panel.loc[ids, 'D'].to_numpy(dtype=float); M = abs(D)
    center = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X-center)**2, axis=0, weights=w)); scale[scale < 1e-12] = 1.
    np.testing.assert_allclose(model['scaler'].mean_, center, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model['scaler'].scale_, scale, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(info['center'], center, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(info['scale'], scale, atol=1e-10, rtol=1e-10)
    Z = (X-center)/scale; heads = model['heads']; checks = 0
    for target, y in [('C', C), ('M', M)]:
        # Solve weighted ridge normal equations independently of sklearn SVD.
        zmean = np.average(Z, axis=0, weights=w); ymean = np.average(y, weights=w)
        A = Z-zmean
        coefficient = np.linalg.solve(A.T @ (w[:, None]*A) + 100*np.eye(len(columns)), A.T @ (w*(y-ymean)))
        intercept = ymean-zmean@coefficient
        np.testing.assert_allclose(heads[target].coef_, coefficient, atol=1e-10, rtol=1e-8)
        near(heads[target].intercept_, intercept)
        np.testing.assert_allclose(info['heads'][target]['coefficients'], coefficient, atol=1e-10, rtol=1e-8)
        near(info['heads'][target]['intercept'], intercept); checks += 1
    cw = w*M
    if cw.sum() > 0:
        cw *= len(w)/cw.sum()
    np.testing.assert_allclose(info['classification_weights'], cw, atol=1e-10, rtol=1e-10)
    fallback = np.average(D > 0, weights=cw) if cw.sum() > 0 else .5
    near(model['q'], fallback); near(info['fallback_q'], fallback)
    if len(np.unique((D[cw > 0] > 0).astype(int))) == 2:
        h = heads['q']; b = h.coef_[0]; a = h.intercept_[0]
        residual = cw*(expit(Z@b+a)-(D > 0))
        gradient = Z.T@residual+b/.1
        assert max(np.max(abs(gradient)), abs(residual.sum())) < 1e-6*max(1, len(w))
        np.testing.assert_allclose(info['heads']['q']['coefficients'], h.coef_, atol=1e-12)
        np.testing.assert_allclose(info['heads']['q']['intercept'], h.intercept_, atol=1e-12)
        checks += 1
    else:
        assert 'q' not in heads
    if not panel.loc[at, 'feature_ready']:
        return {}, checks
    query = (panel.loc[at, columns].to_numpy(dtype=float)-center)/scale
    c = float(query@heads['C'].coef_+heads['C'].intercept_)
    m = max(0., float(query@heads['M'].coef_+heads['M'].intercept_))
    q = float(expit(query@heads['q'].coef_[0]+heads['q'].intercept_[0])) if 'q' in heads else fallback
    d = m*(2*q-1)
    return {'C': c, 'D': d, 'M': m, 'q': q, 'SOXL': c+d, 'SOXS': c-d}, checks


def risk_equations(saved, forecast, at, cutoff, risk_vectors, risk_returns, risk_releases):
    event, fit = saved['event'], saved['risk']
    if not forecast['ready'] or (at, forecast['top_symbol']) not in risk_vectors:
        assert not event['ready'] and event['allocation'] == 0 and event['action'] == 'CASH' and fit is None
        return 0
    symbol = forecast['top_symbol']
    ids = [t for t in sorted(risk_releases) if t < cutoff and risk_releases[t] < cutoff][-120:]
    assert list(pd.to_datetime(fit['members'], utc=True)) == ids
    assert event['top_symbol'] == symbol
    if len(ids) < 60:
        assert not event['ready'] and event['allocation'] == 0
        return 0
    X = np.array([risk_vectors[t, symbol] for t in ids])
    prior = independent_weights(X, risk_vectors[at, symbol], True)
    np.testing.assert_allclose(fit['prior']['weights'], prior, atol=1e-12, rtol=1e-10)
    if 1/(prior@prior) < 20:
        assert not event['ready'] and event['allocation'] == 0
        return 0
    returns = np.array([risk_returns[t, symbol] for t in ids]); x = np.log1p(returns)
    target = forecast['adjusted'][symbol]; mask = prior > 0
    valid = (min(x[mask])+1e-12 < target < max(x[mask])-1e-12) if np.ptp(x[mask]) > 1e-12 else abs(prior@x-target) <= 1e-12
    assert fit['anchor']['valid'] == valid
    if not valid:
        assert not event['ready'] and event['allocation'] == 0
        return 0
    lp = np.log(prior[mask])
    def posterior(value):
        z = lp+value*x[mask]
        return np.exp(z-logsumexp(z))
    lam = brentq(lambda v: posterior(v)@x[mask]-target, -1e6, 1e6, xtol=1e-12) if np.ptp(x[mask]) > 1e-12 else 0.
    q = np.zeros_like(prior); q[mask] = posterior(lam)
    np.testing.assert_allclose(fit['distribution']['weights'], q, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(fit['distribution']['returns'], returns, atol=1e-12)
    near(q@x, target)
    if 1/(q@q) < 20:
        assert not event['ready'] and event['allocation'] == 0
        return 1
    value = float(q@(((1+returns)**-4-1)/-4)); fraction = float(value > 1e-12)
    assert event['ready'] and event['allocation'] == fraction and event['action'] == (symbol if fraction else 'CASH')
    return 1


def verify(run):
    run = Path(run); reg = read(run/'registration.json'); manifest = read(run/'input_manifest.json')
    for name, h in reg['source_sha256'].items():
        assert sha(core.ROOT/name) == h
    source = Path(manifest['source_run'])
    for name, h in manifest['source_files_sha256'].items():
        assert sha(source/name) == h
    for name, h in manifest['bank_files_sha256'].items():
        assert sha(run/name) == h
    original = core.load_market(); assert original.manifest == manifest['full_market']
    market = restrict_market(original, pd.Timestamp(manifest['observation_cutoff_exclusive']))
    panel = pd.read_parquet(run/'panel.parquet'); features = pd.read_parquet(run/'features.parquet')
    old_panel = pd.read_parquet(source/'panel.parquet')
    pd.testing.assert_frame_equal(panel.drop(columns='terminal_release'), old_panel)
    pd.testing.assert_frame_equal(features, pd.read_parquet(source/'features.parquet'))
    base = labels(source/'baseline_labels.json'); stress = labels(source/'stress_labels.json')
    old_histories = read(source/'predictions.json'); old_events = old_histories['regret_regime']
    releases = {}; terminals = {}; receipts = {pd.Timestamp(r['at']): r for r in read(run/'terminal_receipts.json')}
    for at, row in panel[panel.feature_ready].iterrows():
        pair = [base[key(at, s)] for s in core.ETFS]
        target = np.log1p([p.get('net_return', np.nan) for p in pair]); c, d = (target[0]+target[1])/2, (target[0]-target[1])/2
        np.testing.assert_allclose(row[['C','D','M']].to_numpy(dtype=float), [c, d, abs(d)], equal_nan=True)
        if all(p['complete_window'] and p['label_available_at'] is not None for p in pair):
            releases[at] = max(p['label_available_at'] for p in pair)
        observed = [terminal_audit(market, p) for p in pair]
        if all(t is not None for t in observed):
            terminals[at] = max(observed)
            assert panel.loc[at, 'terminal_release'] == terminals[at] == pd.Timestamp(receipts[at]['terminal_release'])
        else:
            assert pd.isna(panel.loc[at, 'terminal_release']) and receipts[at]['terminal_release'] is None
    risks = pd.read_parquet(run/'distribution_bank.parquet')
    risk_vectors = {}; risk_returns = {}; risk_releases = {}
    for e in old_events:
        if not e['ready']:
            continue
        at = pd.Timestamp(e['at']); row = panel.loc[at]
        if at in releases:
            risk_releases[at] = releases[at]
        for symbol, sign in [('SOXL', 1), ('SOXS', -1)]:
            vals = [e['adjusted'][symbol], sign*e['adjusted']['D'], sign*row.relative_5,
                    sign*row.breadth_5, sign*row.volume_direction_diffusion, row.log_sector_volatility]
            risk_vectors[at, symbol] = np.array(vals); risk_returns[at, symbol] = base[key(at, symbol)].get('net_return', np.nan)
            np.testing.assert_allclose(risks.loc[(at, symbol), ['score', 'advantage', 'relative_5', 'breadth_5',
                'volume_direction_diffusion', 'log_sector_volatility']].to_numpy(dtype=float), vals, atol=1e-12)
    predictions = read(run/'predictions.json'); histories = {'early': [], 'full': []}
    members = equations = calibrations = entropy = 0
    for i, e in enumerate(old_events):
        at = pd.Timestamp(e['at']); day = pd.Timestamp(e['date']); cutoff = day.tz_localize(core.ET).tz_convert('UTC')
        record = read(run/'training'/(str(day.date())+'.json')); assert pd.Timestamp(record['cutoff']) == cutoff
        prior_dates = list(panel[panel.feature_ready & (panel.index < cutoff) & (panel.date < day)].index)
        early_ids = []
        for t in prior_dates:
            if t not in terminals or terminals[t] >= cutoff:
                break
            early_ids.append(t)
        full_ids = [t for t in sorted(releases) if t < cutoff and releases[t] < cutoff]
        ids = {'early': early_ids, 'full': full_ids}
        expired = len(early_ids) < len(prior_dates) and panel.loc[prior_dates[len(early_ids)], 'date']+pd.Timedelta(days=14) <= day
        fb = record['feedback']
        assert fb['expired_unresolved_prefix'] == bool(expired)
        assert fb['prior_dates'] == len(prior_dates) and fb['selected_early_dates'] == len(early_ids) and fb['selected_full_dates'] == len(full_ids)
        assert fb['known_but_waiting'] == sum(t in terminals and terminals[t] < cutoff for t in prior_dates[len(early_ids)+1:])
        assert list(pd.to_datetime(record['fit']['members'], utc=True)) == early_ids
        assert list(pd.to_datetime(record['fit']['dates'])) == list(panel.loc[early_ids, 'date'])
        with gzip.open(run/'models'/(str(day.date())+'.pkl.gz'), 'rb') as f:
            model = pickle.load(f)
        raw, count = model_equations(panel, early_ids, at, model, record['fit']); equations += count
        for name, (base_mode, cal_mode, method) in ROUTES.items():
            saved = record['forecasts'][name]; event = saved['event']; cal = saved['calibration']
            current = raw if base_mode == 'early' else e['raw']
            assert set(event['raw']) == set(current)
            for k in current:
                near(event['raw'][k], current[k])
            old = [h for h in histories[base_mode] if h['at'] in set(ids[cal_mode]) and h['raw']][-120:]
            assert list(pd.to_datetime(cal['members'], utc=True)) == [h['at'] for h in old]
            assert cal['ready'] == (len(old) >= 60)
            values = {}
            for target in ['C', 'D']:
                if not old:
                    continue
                x = np.array([h['raw'][target] for h in old]); y = panel.loc[[h['at'] for h in old], target].to_numpy(dtype=float)
                center = x.mean(); scale = x.std() if x.std() >= 1e-12 else 1.
                z = (x-center)/scale; intercept = y.mean(); coefficient = z@(y-intercept)/(z@z+20.)
                for k, v in {'center': center, 'scale': scale, 'intercept': intercept, 'coefficient': coefficient,
                             'score_min': x.min(), 'score_max': x.max()}.items():
                    near(cal['maps'][target][k], v)
                if current:
                    values[target] = current[target] if method == 'raw' else intercept+coefficient*(current[target]-center)/scale
            ready = bool(current) and cal['ready'] and (name == 'full_calibrated' or not expired)
            assert event['ready'] == ready
            if ready:
                c, d = values['C'], values['D']; top = 'SOXL' if d >= 0 else 'SOXS'
                for k, v in {'C': c, 'D': d, 'SOXL': c+d, 'SOXS': c-d}.items():
                    near(event['adjusted'][k], v)
                assert event['top_symbol'] == top and event['action'] == (top if c+abs(d) > 0 else 'CASH')
            else:
                assert event['action'] == 'CASH' and event['adjusted'] is None
            if name == 'full_calibrated':
                for k in ['ready', 'action', 'top_symbol', 'adjusted']:
                    assert event[k] == e[k]
            saved_capital = record['capital'][name]
            assert saved_capital['event'] == predictions[name][i]
            entropy += risk_equations(saved_capital, event, at, cutoff, risk_vectors, risk_returns, risk_releases)
            calibrations += 1
        primary = record['forecasts']['early_base_cal']['event']; u = predictions['early_no_risk'][i]
        take = primary['ready'] and primary['action'] != 'CASH'
        assert u['action'] == (primary['top_symbol'] if take else 'CASH') and u['allocation'] == float(take)
        for name, old in [('source_full', 'regret_regime'), ('source_raw_full', 'same_primary_raw')]:
            assert predictions[name][i]['action'] == old_histories[old][i]['action']
        assert predictions['cash'][i]['action'] == 'CASH' and predictions['double_cost'][i] == predictions['early_base_cal'][i]
        histories['early'].append({'at': at, 'raw': record['forecasts']['early_base_cal']['event']['raw']})
        histories['full'].append({'at': at, 'raw': e['raw']}); members += 1
        if (i+1) % 75 == 0:
            print('verified through', day.date(), flush=True)
    asof = verify_asof_replays(market, base, terminals)
    result = read(run/'results.json'); frozen = read(run/'decisions_frozen.json')['routes']
    p = reg['preregistration']['protocol']; settings = p[reg['phase']]; books = []
    months = settings.get('months', settings.get('regression_months'))
    for name, stats in result['routes'].items():
        wanted = [e for e in predictions[name] if settings['evaluation_start'] <= e['date'][:10] <= settings['evaluation_end']]
        assert frozen[name] == wanted
        books.append(book_check(run, name, market, features, stress if name == 'double_cost' else base,
                                wanted, 2 if name == 'double_cost' else 1, stats, months))
    control = core.ROOT / ('.cache/soxl_soxs_14d5pct_v16/runs/anchored_20261012' if reg['phase'] == 'development'
                           else '.cache/soxl_soxs_14d5pct_v17/runs/cases_20261012')
    for name, old_run, old_name in [('full_calibrated', control, 'anchor_local_full'),
                                  ('source_full', source, 'regret_regime'), ('source_raw_full', source, 'same_primary_raw')]:
        for suffix in ['equity', 'trades']:
            a, b = [pd.read_csv(pth/(n+'_'+suffix+'.csv')) for pth, n in [(run, name), (old_run, old_name)]]
            pd.testing.assert_frame_equal(a[b.columns], b, atol=1e-12, rtol=1e-12)
    receipt = {'passed': True, 'phase': reg['phase'], 'daily_training_membership_checks': members,
        'independent_model_equations': equations, 'calibration_and_decision_checks': calibrations,
        'independent_entropy_roots': entropy, 'joint_terminal_receipts': len(terminals),
        'truncated_asof_outcome_replays': asof, 'capital_books': books,
        'old_forecasts_and_ledgers_exact': True, 'results_sha256': sha(run/'results.json'),
        'engineering_verification_is_not_return_acceptance': True}
    core.dump(run/'verification.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
