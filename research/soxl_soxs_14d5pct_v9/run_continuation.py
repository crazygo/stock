"""Run the registered daily continuation experiment, retaining every attempt."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from continuation import (core, RULE, ENTRY_COLUMNS, MANAGER_COLUMNS, label_key, entry_state,
    baseline_labels, joint_releases, eligible_dates, alive, review_table, fit, calibration,
    predict, choose_entry, managed_exit, day_cutoff)
from risk import direction
from risk_ledger import replay
from study import save_replay
from run_intraday import score
from run_breadth import paired_uncertainty

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_entries(features, labels, dates):
    return [{'id': label_key(at, symbol), 'entry_date': row.date, 'state': entry_state(row, symbol),
             'target': float(np.log1p(labels[label_key(at, symbol)]['net_return']))}
            for at, row in features[features.date.isin(dates)].iterrows() for symbol in core.ETFS]


def selected_history(history, labels):
    return [{'id': label_key(r['at'], r['top_symbol']), 'entry_date': r['entry_date'],
             'target': float(np.log1p(labels[label_key(r['at'], r['top_symbol'])]['net_return'])),
             'raw': r['raw'][r['top_symbol']]}
            for r in history if r['top_symbol'] is not None]


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; keep all attempts')
    for sub in ['', 'training', 'models', 'executed_sources']:
        (output / sub).mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    prereg = json.loads((SOURCE / 'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    source = core.ROOT / protocol['source_features']
    manifest = json.loads((source.parent / 'input_manifest.json').read_text())
    assert manifest['feature_sha256'] == sha(source)
    for name in ['verification', 'input_verification', 'numerical_audit']:
        assert json.loads((source.parent / (name + '.json')).read_text())['passed']
    market = core.load_market(); assert market.manifest == manifest['market']
    features = pd.read_parquet(source)
    features = features[features.clock.eq(protocol['analysis_clock_et'])]
    assert not features.date.duplicated().any()
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'input_manifest.json', {'market': market.manifest, 'source_feature_file': str(source),
        'source_feature_sha256': sha(source), 'source_input_manifest_sha256': sha(source.parent / 'input_manifest.json'),
        'features_sha256': sha(output / 'features.parquet'), 'rows': len(features),
        'entry_columns': ENTRY_COLUMNS, 'manager_columns': MANAGER_COLUMNS})
    files = list(SOURCE.glob('*.py')) + [SOURCE / 'protocol.json']
    files += [SOURCE.parent / f'soxl_soxs_14d5pct_{v}' / n for v, n in [
        ('v2', 'core.py'), ('v2', 'study.py'), ('v3', 'risk.py'), ('v3', 'risk_ledger.py'),
        ('v4', 'walkforward.py'), ('v4', 'daily_study.py'), ('v6', 'path_engine.py'),
        ('v6', 'run_intraday.py'), ('v7', 'breadth.py'), ('v7', 'run_breadth.py')]]
    core.dump(output / 'registration.json', {'started_at': datetime.now(timezone.utc).isoformat(),
        'preregistration': prereg, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {str(p.relative_to(core.ROOT)): sha(p) for p in files}})
    for path in files:
        dest = output / 'executed_sources' / path.parent.name / path.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(path.read_bytes())
    baseline = baseline_labels(market, features)
    stress_base = baseline_labels(market, features, 2)
    releases = joint_releases(features, baseline)
    reviews = review_table(market, features, baseline)
    stress_reviews = review_table(market, features, stress_base, 2)
    core.dump(output / 'baseline_labels.json', list(baseline.values()))
    core.dump(output / 'stress_baseline_labels.json', list(stress_base.values()))
    core.dump(output / 'review_states.json', reviews)
    core.dump(output / 'stress_review_states.json', stress_reviews)
    core.dump(output / 'releases.json', [{'date': d, 'release': r} for d, r in releases.items()])
    print('prepared', len(features), 'dates;', len(reviews), 'review states', flush=True)
    managed = dict(baseline); stressed = dict(stress_base)
    histories = {name: [] for name in ['managed', 'baseline']}
    manager_history = []; manager_events = []; registry = []
    by_time = {k: {} for k in ['base', 'stress']}
    for name, rows in [('base', reviews), ('stress', stress_reviews)]:
        for row in rows:
            by_time[name].setdefault(row['at'], []).append(row)
    with threadpool_limits(limits=1):
        for i, (at, row) in enumerate(features.iterrows()):
            cutoff = day_cutoff(at); dates = eligible_dates(releases, cutoff, protocol)
            record = {'date': row.date, 'at': at, 'cutoff': cutoff, 'eligible_dates': dates,
                      'latest_label_release': max((releases[d] for d in dates), default=None),
                      'fits': {}, 'calibrations': {}, 'entry': {}, 'manager': []}
            models = {}
            for name, labels in [('managed', managed), ('baseline', baseline)]:
                models[name], record['fits'][name] = fit(training_entries(features, labels, dates), ENTRY_COLUMNS, protocol)
                record['calibrations'][name] = calibration(selected_history(histories[name], labels), dates, protocol)
            training_reviews = [r for r in reviews if r['entry_date'] in dates and
                                r['state'] is not None and np.isfinite(r['target'])]
            models['manager'], record['fits']['manager'] = fit(training_reviews, MANAGER_COLUMNS, protocol)
            manager_cal = calibration(manager_history, dates, protocol)
            record['calibrations']['manager'] = manager_cal
            with gzip.open(output / 'models' / (str(row.date.date()) + '.pkl.gz'), 'wb') as f:
                pickle.dump(models, f)
            # Apply a manager only to episodes whose prior observed path remains alive.
            # The stress path uses its own fill, barriers and surviving state; never its labels for training.
            intents = []
            for name, labels in [('base', managed), ('stress', stressed)]:
                for review in by_time[name].get(at, []):
                    if not alive(labels[review['key']], at):
                        continue
                    raw = (predict(models['manager'], review['state'], MANAGER_COLUMNS)
                           if models['manager'] is not None and review['state'] is not None else None)
                    adjusted = raw + manager_cal['correction'] if raw is not None and manager_cal['correction'] is not None else None
                    ready = raw is not None and manager_cal['ready']
                    event = {'id': review['id'], 'key': review['key'], 'cost_path': name, 'at': at,
                             'entry_date': review['entry_date'], 'raw': raw, 'adjusted': adjusted,
                             'ready': ready, 'action': 'EXIT' if ready and adjusted <= 0 else 'HOLD'}
                    manager_events.append(event); record['manager'].append(event)
                    if name == 'base' and raw is not None:
                        manager_history.append({**event, 'target': review['target']})
                    if event['action'] == 'EXIT':
                        intents.append((labels, review))
            # Entry forecasts are emitted before processing any future Open execution.
            for name in histories:
                raw = {s: predict(models[name], entry_state(row, s), ENTRY_COLUMNS) for s in core.ETFS} if models[name] else {}
                event = {'at': at, 'entry_date': row.date, **choose_entry(raw, record['calibrations'][name])}
                histories[name].append(event); record['entry'][name] = event
            for labels, review in intents:
                labels[review['key']] = managed_exit(labels[review['key']], review['sale'], at)
            core.dump(output / 'training' / (str(row.date.date()) + '.json'), record)
            registry.append({'date': row.date, 'cutoff': cutoff, 'latest_label_release': record['latest_label_release'],
                'training_dates': {n: len(v['dates']) for n, v in record['fits'].items()},
                'calibration_dates': {n: len(v['dates']) for n, v in record['calibrations'].items()},
                'entry_action': record['entry']['managed']['action'], 'manager_reviews': len(record['manager'])})
            if (i + 1) % 50 == 0:
                print('trained through', row.date.date(), flush=True)
    core.dump(output / 'training_registry.json', registry)
    core.dump(output / 'managed_labels.json', list(managed.values()))
    core.dump(output / 'stress_managed_labels.json', list(stressed.values()))
    core.dump(output / 'manager_predictions.json', manager_events)
    for name, events in histories.items():
        core.dump(output / (name + '_entry_predictions.json'), events)
    start, end = protocol['evaluation_start'], protocol['evaluation_end']
    test = features[(features.date >= start) & (features.date <= end)]
    primary = protocol['primary']; primary_actions = {r['at']: r['action'] for r in histories['managed']}
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'new_strategy_case_results_seen_before_decisions_frozen': False, 'previous_versions_same_history_exposed': True,
        'entry': [r for r in histories['managed'] if r['at'] in test.index],
        'manager_predictions_sha256': sha(output / 'manager_predictions.json')})
    definitions = [(primary, managed, primary_actions, 1),
        (protocol['controls'][0], baseline, primary_actions, 1),
        (protocol['controls'][1], baseline, {r['at']: r['action'] for r in histories['baseline']}, 1),
        (protocol['controls'][2], baseline, {at: direction(row, 'trend20') for at, row in test.iterrows()}, 1),
        ('double_cost', stressed, primary_actions, 2)]
    results = {'primary_name': primary, 'routes': {}}; books = {}
    for name, labels, actions, multiple in definitions:
        book = replay(market, test, list(labels.values()), actions, start, end, cost_multiple=multiple)
        books[name] = book; save_replay(output, name, book)
        result = score(book, protocol['regression_months'])
        trades = book[1]
        done = trades[(pd.to_datetime(trades.exit_at, utc=True) <= pd.Timestamp(end, tz=core.ET) + pd.Timedelta(days=1))] if len(trades) else trades
        result['continuation_exits'] = int(done.reason.eq('continuation_exit').sum()) if len(done) else 0
        results['routes'][name] = result
    results['primary'] = results['routes'][primary]
    results['paired_exit_increment'] = paired_uncertainty(books[primary][0], books[protocol['controls'][0]][0])
    required = market.sessions[(market.sessions >= start) & (market.sessions <= end)]
    covered = pd.DatetimeIndex(test.date).equals(required)
    results['acceptance'] = {'passed': bool(covered and results['primary']['primary_pass']),
        'complete_premarket_coverage': covered, 'expected_days': len(required), 'observed_days': len(test),
        'new_buy_eligible': False, 'live_entry_not_replaced': True,
        'independent_forward_validation_passed': False, 'future_nonnegative_month_guarantee': False}
    core.dump(output / 'results.json', results)
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
