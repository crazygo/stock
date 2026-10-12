"""Independent as-of memberships, model refits, calibration, fills and capital audit."""
import argparse
import gzip
import hashlib
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from entry_value import core, restrict_market, ENTRY_COLUMNS, LEARNERS
from verify_continuation import read, labels, key, near, raw_exit, independent_state
from verify_uncertainty import book_check


def check_probability_cal(cal, old, lab, valid):
    past = [e for e in old if pd.Timestamp(e['date']) in valid and e['raw']][-60:]
    assert list(pd.to_datetime(cal['probability_dates'])) == [pd.Timestamp(e['date']) for e in past]
    assert cal['probability_ready'] == (len(past) >= 40)
    for s in core.ETFS:
        residuals = [lab[key(e['at'], s)]['target_hit'] - e['raw'][s]['p'] for e in past]
        if residuals:
            near(cal['probability_correction'][s], np.mean(residuals))
        else:
            assert cal['probability_correction'][s] is None
    top = [e for e in old if pd.Timestamp(e['date']) in valid and e['top_symbol'] is not None and e['score_ready']][-60:]
    assert list(pd.to_datetime(cal['top_dates'])) == [pd.Timestamp(e['date']) for e in top]
    assert cal['top_members'] == [key(e['at'], e['top_symbol']) for e in top]
    assert cal['top_ready'] == (len(top) >= 40)
    if top:
        near(cal['top_correction'], np.mean([np.log1p(lab[key(e['at'], e['top_symbol'])]['net_return']) - e['scores'][e['top_symbol']] for e in top]))
    else:
        assert cal['top_correction'] is None


def verify(run):
    run = Path(run); reg = read(run / 'registration.json'); p = reg['preregistration']['protocol']
    manifest = read(run / 'input_manifest.json'); cutoff = pd.Timestamp(manifest['observation_cutoff_exclusive'])
    for file, digest in reg['source_sha256'].items():
        assert hashlib.sha256((core.ROOT / file).read_bytes()).hexdigest() == digest
    source = Path(manifest['source_feature_file'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_feature_sha256']
    original = core.load_market(); assert original.manifest == manifest['full_market']
    market = restrict_market(original, cutoff)
    for s, f in market.bars.items():
        assert (f.available_at < cutoff).all() and (f.end_at < cutoff).all()
    features = pd.read_parquet(run / 'features.parquet')
    assert (features.index < cutoff).all() and features.clock.eq('09:25:30').all()
    base = labels(run / 'baseline_labels.json'); stress = labels(run / 'stress_labels.json')
    releases = {}
    for at, row in features.iterrows():
        pair = [base[key(at, s)] for s in core.ETFS]
        if all(x['complete_window'] and x.get('label_available_at') is not None and np.isfinite(x.get('net_return', np.nan)) for x in pair):
            for x in pair:
                assert x['label_available_at'] >= x['expiry_at'] + pd.Timedelta(seconds=1)
            releases[row.date] = max(x['label_available_at'] for x in pair)
    histories = {m: [] for m in LEARNERS}; all_events = read(run / 'predictions.json')
    memberships = forecasts = refits = 0
    params = dict(max_iter=80, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=.05,
                  l2_regularization=5, early_stopping=False, random_state=814)
    anchors = ['2025-09-02', '2025-12-01', '2026-03-02']
    with threadpool_limits(limits=1):
        for i, (at, row) in enumerate(features.iterrows()):
            record = read(run / 'training' / (str(row.date.date()) + '.json'))
            day_cutoff = row.date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(record['cutoff']) == day_cutoff
            with gzip.open(run / 'models' / (str(row.date.date()) + '.pkl.gz'), 'rb') as f:
                models = pickle.load(f)
            for method in LEARNERS:
                saved = record['routes'][method]; info = saved['fit']; cal = saved['calibration']; event = saved['event']
                assert event == all_events[method][i] and pd.Timestamp(event['at']) == at
                dates = [d for d, r in sorted(releases.items()) if r < day_cutoff and d.tz_localize(core.ET) < day_cutoff]
                if method == 'two_part_252': dates = dates[-252:]
                members = [(t, s) for t, r in features[features.date.isin(dates)].iterrows() for s in core.ETFS]
                assert info['members'] == [key(t, s) for t, s in members]
                assert list(pd.to_datetime(info['dates'])) == dates
                assert info['ready'] == (len(dates) >= 120)
                if dates: assert pd.Timestamp(saved['latest_label_release']) == max(releases[d] for d in dates)
                check_probability_cal(cal, histories[method], base, set(dates)); memberships += 1
                model = models[method]
                if model is None:
                    assert not info['ready'] and not event['raw'] and event['action'] == 'CASH'
                    histories[method].append(event); continue
                near(info['weight_sum'], len(dates))
                X = np.array([[independent_state(features.loc[t], s)[c] for c in ENTRY_COLUMNS] for t, s in members])
                y = np.array([np.log1p(base[key(t, s)]['net_return']) for t, s in members])
                hit = np.array([base[key(t, s)]['target_hit'] for t, s in members])
                current = np.array([[independent_state(row, s)[c] for c in ENTRY_COLUMNS] for s in core.ETFS])
                for j, s in enumerate(core.ETFS):
                    mask = np.array([sym == s for _, sym in members]); win = y[mask & (hit == 1)]; lose = y[mask & (hit == 0)]
                    means = model['means'][s]
                    for field, value in [('p', hit[mask].mean()), ('log_return', y[mask].mean()),
                                         ('win', win.mean() if len(win) else 0.), ('failure', lose.mean() if len(lose) else 0.)]:
                        near(means[field], value); near(info['means'][s][field], value)
                    expected = {'p': model['classifier'].predict_proba(current)[j, 1] if model['classifier'] else means['p'],
                        'failure': model['failure'].predict(current)[j] if model['failure'] else means['failure'],
                        'direct': model['direct'].predict(current)[j] if model['direct'] else means['log_return'], 'win': means['win']}
                    for field, value in expected.items(): near(event['raw'][s][field], value)
                    forecasts += 1
                if method.startswith('two_part'):
                    fail_dates = sorted({features.loc[t, 'date'] for (t, _), h in zip(members, hit) if not h})
                    assert list(pd.to_datetime(info['failure_dates'])) == fail_dates
                    assert info['failure_members'] == [key(t, s) for (t, s), h in zip(members, hit) if not h]
                    assert info['failure_model_ready'] == (len(fail_dates) >= 40)
                if str(row.date.date()) in anchors and method != 'mean_expanding':
                    if method == 'direct_expanding':
                        fit = HistGradientBoostingRegressor(**params).fit(X, y, sample_weight=np.full(len(y), .5))
                        assert np.allclose(fit.predict(current), model['direct'].predict(current), atol=1e-12, rtol=1e-12)
                    else:
                        fit = HistGradientBoostingClassifier(**params).fit(X, hit, sample_weight=np.full(len(y), .5))
                        assert np.allclose(fit.predict_proba(current), model['classifier'].predict_proba(current), atol=1e-12, rtol=1e-12)
                        fit = HistGradientBoostingRegressor(**params).fit(X[hit == 0], y[hit == 0], sample_weight=np.full((hit == 0).sum(), .5))
                        assert np.allclose(fit.predict(current), model['failure'].predict(current), atol=1e-12, rtol=1e-12)
                    refits += 1
                scores = {}
                for s, raw in event['raw'].items():
                    prob = np.clip(raw['p'] + (cal['probability_correction'][s] or 0.), .001, .999)
                    near(event['probabilities'][s], prob)
                    scores[s] = prob * raw['win'] + (1 - prob) * raw['failure'] if method.startswith('two_part') else raw['direct']
                    near(event['scores'][s], scores[s])
                top = max(core.ETFS, key=lambda s: scores[s]); assert event['top_symbol'] == top
                score_ready = cal['probability_ready'] if method.startswith('two_part') else True
                assert event['score_ready'] == score_ready
                adjusted = scores[top] + cal['top_correction'] if cal['top_correction'] is not None else None
                if adjusted is not None: near(event['adjusted'], adjusted)
                ready = score_ready and cal['top_ready']; assert event['ready'] == ready
                assert event['action'] == (top if ready and adjusted > 0 else 'CASH')
                histories[method].append(event)
            if (i + 1) % 75 == 0: print('verified forecasts through', row.date.date(), flush=True)
    exits = 0
    for lab, multiple in [(base, 1), (stress, 2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label:
                raw_exit(market, label, None, multiple); exits += 1
    result = read(run / 'results.json'); frozen = read(run / 'decisions_frozen.json'); books = []
    settings = p[reg['phase']]; months = settings.get('months', settings.get('regression_months'))
    for name, stats in result['routes'].items():
        actions = {pd.Timestamp(e['at']): e['action'] for e in frozen['actions'][name]}
        if name in LEARNERS:
            assert all(actions[pd.Timestamp(e['at'])] == e['action'] for e in histories[name] if pd.Timestamp(e['at']) in actions)
        elif name == 'double_cost': assert frozen['actions'][name] == frozen['actions'][p['primary']]
        else:
            for at, action in actions.items():
                expected = {'always_soxl': 'SOXL', 'always_soxs': 'SOXS', 'cash': 'CASH'}.get(name)
                if name == 'trend20':
                    r = features.loc[at, 'SOXX_ret20']; expected = 'SOXL' if r > 0 else 'SOXS' if r < 0 else 'CASH'
                assert action == expected
        books.append(book_check(run, name, market, features, stress if name == 'double_cost' else base,
                                actions, 2 if name == 'double_cost' else 1, stats, months))
    verification = {'passed': True, 'phase': reg['phase'], 'training_membership_checks': memberships,
        'candidate_forecast_checks': forecasts, 'anchor_model_bundle_refits': refits, 'anchors': anchors,
        'raw_exit_checks': exits, 'capital_books': books, 'observation_cutoff_exclusive': cutoff,
        'results_sha256': hashlib.sha256((run / 'results.json').read_bytes()).hexdigest(),
        'probability_and_selected_value_calibration_separate': True, 'all_inputs_before_cutoff': True,
        'engineering_verification_is_not_effect_acceptance': True}
    core.dump(run / 'verification.json', verification); print(verification, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    verify(p.parse_args().run)
