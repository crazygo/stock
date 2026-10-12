"""Independent maturity, convex optimality, entropy and capital checks."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from resolved_feedback import core, ROUTES, STATE_COLUMNS, restrict_market
from risk import risk_label
from verify_continuation import read, labels, key, near
from verify_allocation import book_check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_weights(X, query, local):
    n = len(X); b = np.array([2 ** ((i - n + 1) / 63) for i in range(n)])
    b /= sum(b); w = b.copy()
    if local:
        center = sum(b[i] * X[i] for i in range(n))
        variance = sum(b[i] * (X[i] - center) ** 2 for i in range(n))
        scale = np.sqrt(variance); scale[scale < 1e-12] = 1.
        distance = np.sum(((X - query) / scale) ** 2, axis=1) / X.shape[1]
        bandwidth = max(sorted(distance)[39], 1e-12)
        w *= np.exp(-distance / (2 * bandwidth))
    return w / sum(w)


def terminal_audit(market, label):
    if label.get('exit_at') is None or label.get('reason') not in ['target', 'risk_stop', 'expiry'] or label.get('net_return') is None or not np.isfinite(label.get('net_return', np.nan)):
        return None
    a, b = label['entry_at'], label['exit_at']
    f = market.bars[label['symbol']]
    path = f[(f.start_at >= a) & (f.end_at <= b) & f.session_type.eq('regular')]
    starts = []
    days = market.sessions[(market.sessions >= label['entry_date']) & (market.sessions <= b.tz_convert(core.ET).tz_localize(None).normalize())]
    for d in days:
        lo = max(a, pd.Timestamp(market.calendar[d]['open_at']))
        hi = min(b, pd.Timestamp(market.calendar[d]['close_at']))
        starts.extend(pd.date_range(lo, hi, freq='5min', inclusive='left'))
    if not starts or list(path.start_at) != starts or path.end_at.iloc[-1] != b:
        return None
    if not path.end_at.eq(path.start_at + pd.Timedelta(minutes=5)).all() or path.available_at.isna().any() or not path.available_at.ge(path.end_at).all():
        return None
    release = max(path.available_at)
    if label.get('dividends'):
        if not label.get('complete_window') or label.get('label_available_at') is None:
            return None
        release = max(release, label['label_available_at'])
    return release


def verify_asof_replays(market, base, terminal_releases):
    groups = {}
    for at, release in terminal_releases.items():
        dc = release.tz_convert(core.ET).normalize() + pd.Timedelta(days=1)
        groups.setdefault(dc, []).append(at)
    checks = 0
    for dc, dates in sorted(groups.items()):
        limited = restrict_market(market, dc)
        assert all((f.end_at < dc).all() and (f.available_at < dc).all() for f in limited.bars.values())
        for at in dates:
            for symbol in core.ETFS:
                old = base[key(at, symbol)]
                fresh = risk_label(limited, symbol, at, {'id': 'fixed_stop6', 'stop_fraction': .06, 'breakeven_trigger': None})
                assert fresh['exit_at'] == old['exit_at'] and fresh['reason'] == old['reason'] and fresh['prefix_complete']
                near(fresh['net_return'], old['net_return']); near(fresh['exit_fill_economic'], old['exit_fill_economic'])
                checks += 1
    return checks


def verify(run):
    amendment = read(Path(__file__).parent / 'evidence/verification_amendment.json')
    assert sha(Path(__file__)) == amendment['corrected_verifier_sha256']
    assert sha(Path(__file__).parent / 'verify_feedback.py') == amendment['original_verifier_sha256']
    run = Path(run); reg = read(run / 'registration.json'); manifest = read(run / 'input_manifest.json')
    for f, h in reg['source_sha256'].items():
        assert sha(core.ROOT / f) == h
    source = Path(manifest['source_run'])
    for f, h in manifest['source_files_sha256'].items():
        assert sha(source / f) == h
    for f, h in manifest['bank_files_sha256'].items():
        assert sha(run / f) == h
    p = reg['preregistration']['protocol']; settings = p[reg['phase']]
    cutoff = pd.Timestamp(manifest['observation_cutoff_exclusive'])
    original = core.load_market(); assert original.manifest == manifest['full_market']
    market = restrict_market(original, cutoff)
    base = labels(source / 'baseline_labels.json'); stress = labels(source / 'stress_labels.json')
    events = read(source / 'predictions.json')['regret_regime']
    original_histories = read(source / 'predictions.json')
    panel = pd.read_parquet(source / 'panel.parquet'); features = pd.read_parquet(run / 'features.parquet')
    experts = pd.read_parquet(run / 'expert_bank.parquet'); risks = pd.read_parquet(run / 'distribution_bank.parquet')
    releases = {}; terminals = {}; vectors = {}; targets = {}; risk_vectors = {}; returns = {}
    receipts = {pd.Timestamp(x['at']): x for x in read(run / 'terminal_receipts.json')}
    for e in events:
        if not e['ready']:
            continue
        at = pd.Timestamp(e['at']); pair = [base[key(at, s)] for s in core.ETFS]; row = panel.loc[at]
        if all(l['complete_window'] and l['label_available_at'] is not None for l in pair):
            assert all(l['label_available_at'] >= l['expiry_at'] + pd.Timedelta(seconds=1) for l in pair)
            releases[at] = max(l['label_available_at'] for l in pair)
        observed = [terminal_audit(market, l) for l in pair]
        receipt = receipts[at]
        if all(t is not None for t in observed):
            terminals[at] = max(observed)
            assert pd.Timestamp(receipt['terminal_release']) == terminals[at] == experts.loc[at, 'terminal_release']
        else:
            assert receipt['terminal_release'] is None and pd.isna(experts.loc[at, 'terminal_release'])
        for symbol, release in zip(core.ETFS, observed):
            side = receipt['sides'][symbol]
            assert side['known'] == (release is not None)
            if release is not None: assert pd.Timestamp(side['release']) == release
        actual = np.log1p([l.get('net_return', np.nan) for l in pair])
        targets[at] = np.array([(actual[0] + actual[1]) / 2, (actual[0] - actual[1]) / 2])
        vectors[at] = row[STATE_COLUMNS].to_numpy(dtype=float)
        ex = experts.loc[at]
        np.testing.assert_allclose(ex[STATE_COLUMNS].to_numpy(dtype=float), vectors[at], atol=1e-12)
        for j, k in enumerate(['C', 'D']):
            near(ex['raw_' + k], e['raw'][k]); near(ex['cal_' + k], e['adjusted'][k])
            np.testing.assert_allclose(ex['actual_' + k], targets[at][j], equal_nan=True, atol=1e-12)
        for s, sign in [('SOXL', 1), ('SOXS', -1)]:
            risk_vectors[at, s] = np.array([e['adjusted'][s], sign * e['adjusted']['D'], sign * row.relative_5,
                sign * row.breadth_5, sign * row.volume_direction_diffusion, row.log_sector_volatility])
            returns[at, s] = base[key(at, s)].get('net_return', np.nan)
            np.testing.assert_allclose(risks.loc[(at, s), ['score', 'advantage', 'relative_5', 'breadth_5',
                'volume_direction_diffusion', 'log_sector_volatility']].to_numpy(dtype=float), risk_vectors[at, s], atol=1e-12)
    by_at = {pd.Timestamp(e['at']): e for e in events}
    predictions = read(run / 'predictions.json'); frozen = read(run / 'decisions_frozen.json')['routes']
    membership = optimality = entropy = 0
    for i, e in enumerate(events):
        at = pd.Timestamp(e['at']); date = pd.Timestamp(e['date']); dc = date.tz_localize(core.ET).tz_convert('UTC')
        record = read(run / 'training' / (str(date.date()) + '.json'))
        assert pd.Timestamp(record['cutoff']) == dc
        full_ids = [t for t in sorted(releases) if t < dc and releases[t] < dc][-120:] if e['ready'] else []
        prior_dates = [t for t in sorted(vectors) if t < dc]
        available = [t for t in prior_dates if t in terminals and terminals[t] < dc]
        prefix = []
        for t in prior_dates:
            if t not in terminals or terminals[t] >= dc: break
            prefix.append(t)
        expired = len(prefix) < len(prior_dates) and pd.Timestamp(by_at[prior_dates[len(prefix)]]['date']) + pd.Timedelta(days=14) <= date
        for name, (pool_mode, mode, method) in ROUTES.items():
            ids = (full_ids if pool_mode == 'full' else available[-120:] if pool_mode == 'available' else prefix[-120:]) if e['ready'] else []
            saved = record['routes'][name]; blend = saved['blend']; event = predictions[name][i]
            assert event == saved['event'] and list(pd.to_datetime(blend['members'], utc=True)) == ids
            metadata = blend['feedback']
            assert metadata['prior_dates'] == len(prior_dates) and metadata['terminal_known_dates'] == len(available)
            if pool_mode == 'frontier':
                assert metadata['expired_unresolved_prefix'] == bool(expired)
                assert metadata['known_but_waiting_behind_pending'] == len(set(available) - set(prefix))
                if expired:
                    assert not blend['ready'] and not event['ready'] and event['action'] == 'CASH'
                    continue
            membership += 1
            if len(ids) < 60:
                assert not blend['ready'] and not event['ready'] and event['action'] == 'CASH'
                continue
            X = np.array([vectors[t] for t in ids]); w = independent_weights(X, vectors[at], mode == 'state')
            np.testing.assert_allclose(blend['weights']['weights'], w, atol=1e-12, rtol=1e-10)
            neff = 1 / (w @ w); near(blend['weights']['effective_dates'], neff)
            if neff < 20 and method not in ['raw', 'calibrated', 'equal']:
                assert not blend['ready'] and event['action'] == 'CASH'; continue
            adjusted = {}
            for j, k in enumerate(['C', 'D']):
                raw = np.array([by_at[t]['raw'][k] for t in ids]); cal = np.array([by_at[t]['adjusted'][k] for t in ids])
                actual = np.array([targets[t][j] for t in ids]); delta = cal - raw
                h = blend['heads'][k]; alpha = h['alpha_calibrated']; assert 0 <= alpha <= 1
                curvature = sum(w * delta ** 2); penalty = 2 / neff * curvature
                near(h['curvature'], curvature); near(h['penalty'], penalty)
                raw_loss = sum(w * (raw - actual) ** 2); cal_loss = sum(w * (cal - actual) ** 2)
                near(h['mse_raw'], raw_loss); near(h['mse_calibrated'], cal_loss)
                loss = sum(w * (raw + alpha * delta - actual) ** 2)
                near(h['mse_blended'], loss); near(h['objective'], loss + penalty * (alpha - .5) ** 2)
                if method == 'blend':
                    if curvature <= 1e-16:
                        assert alpha == .5
                    else:
                        gradient = sum(w * delta * (raw + alpha * delta - actual)) + penalty * (alpha - .5)
                        assert (abs(gradient) < 1e-12 if 0 < alpha < 1 else gradient >= -1e-12 if alpha == 0 else gradient <= 1e-12)
                    optimality += 1
                elif method == 'hard':
                    assert alpha == (.5 if abs(raw_loss - cal_loss) <= 1e-12 else float(cal_loss < raw_loss))
                else:
                    assert alpha == {'raw': 0., 'calibrated': 1., 'equal': .5}[method]
                adjusted[k] = (1 - alpha) * e['raw'][k] + alpha * e['adjusted'][k]
                near(blend['adjusted'][k], adjusted[k])
            top = 'SOXL' if adjusted['D'] >= 0 else 'SOXS'
            assert blend['top_symbol'] == event['top_symbol'] == top
            for s, sign in [('SOXL', 1), ('SOXS', -1)]:
                near(event['adjusted'][s], adjusted['C'] + sign * adjusted['D'])
            risk = saved['risk']; assert list(pd.to_datetime(risk['members'], utc=True)) == full_ids
            if len(full_ids) < 60:
                assert not event['ready'] and event['allocation'] == 0.; continue
            Z = np.array([risk_vectors[t, top] for t in full_ids])
            prior = independent_weights(Z, risk_vectors[at, top], True)
            np.testing.assert_allclose(risk['prior']['weights'], prior, atol=1e-12, rtol=1e-10)
            if 1 / (prior @ prior) < 20:
                assert not event['ready'] and event['allocation'] == 0.; continue
            r = np.array([returns[t, top] for t in full_ids]); x = np.log1p(r); target = event['adjusted'][top]
            a = risk['anchor']; mask = prior > 0
            valid = (min(x[mask]) + 1e-12 < target < max(x[mask]) - 1e-12) if np.ptp(x[mask]) > 1e-12 else abs(prior @ x - target) <= 1e-12
            assert a['valid'] == valid
            if not valid:
                assert not event['ready'] and event['allocation'] == 0.; continue
            lp = np.log(prior[mask])
            def posterior(lam):
                z = lp + lam * x[mask]
                return np.exp(z - logsumexp(z))
            lam = brentq(lambda value: posterior(value) @ x[mask] - target, -1e6, 1e6, xtol=1e-12) if np.ptp(x[mask]) > 1e-12 else 0.
            q = np.zeros_like(prior); q[mask] = posterior(lam)
            np.testing.assert_allclose(risk['distribution']['weights'], q, atol=1e-10, rtol=1e-9)
            near(q @ x, target); entropy += 1
            if 1 / (q @ q) < 20:
                assert not event['ready'] and event['allocation'] == 0.; continue
            value = float(q @ (((1 + r) ** -4 - 1) / -4))
            allocation = float(value > 1e-12)
            assert event['ready'] and event['allocation'] == allocation
            assert event['action'] == (top if allocation else 'CASH')
        primary = record['routes'][p['primary']]['blend']; unfiltered = predictions['frontier_no_risk'][i]
        take = primary['ready'] and max(primary['adjusted'][s] for s in core.ETFS) > 0
        assert unfiltered['action'] == (primary['top_symbol'] if take else 'CASH') and unfiltered['allocation'] == float(take)
        for name, old in [('source_full', 'regret_regime'), ('source_raw_full', 'same_primary_raw')]:
            assert predictions[name][i]['action'] == original_histories[old][i]['action']
        assert predictions['cash'][i]['action'] == 'CASH' and predictions['double_cost'][i] == predictions[p['primary']][i]
    results = read(run / 'results.json'); books = []
    months = settings.get('months', settings.get('regression_months'))
    for name, stats in results['routes'].items():
        wanted = [e for e in predictions[name] if settings['evaluation_start'] <= e['date'][:10] <= settings['evaluation_end']]
        assert wanted == frozen[name]
        books.append(book_check(run, name, market, features, stress if name == 'double_cost' else base,
                                wanted, 2 if name == 'double_cost' else 1, stats, months))
    control_run = core.ROOT / ('.cache/soxl_soxs_14d5pct_v16/runs/anchored_20261012' if reg['phase'] == 'development'
                               else '.cache/soxl_soxs_14d5pct_v17/runs/cases_20261012')
    selection_run = core.ROOT / ('.cache/soxl_soxs_14d5pct_v18/runs/blend_20261012' if reg['phase'] == 'development'
                                 else '.cache/soxl_soxs_14d5pct_v19/runs/cases_20261012')
    for name, old_run, old_name in [('full_window_hard', selection_run, 'state_hard'),
                                   ('raw_fixed', selection_run, 'raw_fixed'),
                                   ('calibrated_fixed', control_run, 'anchor_local_full'),
                                   ('source_full', source, 'regret_regime'), ('source_raw_full', source, 'same_primary_raw')]:
        for suffix in ['equity', 'trades']:
            now = pd.read_csv(run / (name + '_' + suffix + '.csv'))
            before = pd.read_csv(old_run / (old_name + '_' + suffix + '.csv'))
            pd.testing.assert_frame_equal(now[before.columns], before, atol=1e-12, rtol=1e-12)
    asof_checks = verify_asof_replays(market, base, terminals)
    report = {'passed': True, 'phase': reg['phase'], 'expert_dates': len(experts), 'joint_terminal_receipts': len(terminals), 'truncated_asof_outcome_replays': asof_checks,
        'daily_maturity_decision_checks': membership, 'independent_convex_optimality_checks': optimality,
        'independent_entropy_root_checks': entropy, 'capital_books': books,
        'full_window_selection_and_fixed_forecast_controls_exact': True, 'source_labels_and_market_unchanged': True,
        'verification_amendment_sha256': sha(Path(__file__).parent / 'evidence/verification_amendment.json'), 'results_sha256': sha(run / 'results.json'), 'engineering_verification_is_not_return_acceptance': True}
    core.dump(run / 'verification.json', report); print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
