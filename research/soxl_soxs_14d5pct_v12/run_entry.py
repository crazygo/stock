"""Entry information study: matched contracts, daily learning, pre-May observations."""
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
from entry_value import (core, restrict_market, development_gate, ENTRY_COLUMNS, LEARNERS,
    baseline_labels, joint_releases, day_cutoff, eligible, examples, fit, forecast, calibrate, decide)
from risk import direction
from risk_ledger import replay
from study import save_replay
from run_intraday import score
from run_breadth import paired_uncertainty

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_cases(development, protocol):
    run = Path(development)
    registration = json.loads((run / 'registration.json').read_text())
    verification = json.loads((run / 'verification.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    assert registration['phase'] == 'development' and registration['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    assert verification['passed'] and verification['results_sha256'] == sha(run / 'results.json')
    gate = development_gate(results['primary'], results['coverage']['complete'], protocol)
    if not gate['passed']:
        raise ValueError('Pre-case development gate failed; case replay is forbidden')
    return {'run': str(run.resolve()), 'results_sha256': sha(run / 'results.json'), 'gate': gate}


def run(output, case_after=None):
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    prereg = json.loads((SOURCE / 'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    authorization = authorize_cases(case_after, protocol) if case_after else None
    phase = 'cases' if case_after else 'development'; settings = protocol[phase]
    start, end = settings['evaluation_start'], settings['evaluation_end']
    cutoff = pd.Timestamp(end, tz=core.ET) + pd.Timedelta(days=1)
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['', 'models', 'training', 'executed_sources']:
        (output / sub).mkdir(parents=True, exist_ok=True)
    source = core.ROOT / protocol['source_features']; manifest = json.loads((source.parent / 'input_manifest.json').read_text())
    assert manifest['feature_sha256'] == sha(source)
    for name in ['verification', 'input_verification', 'numerical_audit']:
        assert json.loads((source.parent / (name + '.json')).read_text())['passed']
    original = core.load_market(); assert original.manifest == manifest['market']
    market = restrict_market(original, cutoff)
    features = pd.read_parquet(source)
    features = features[features.clock.eq(protocol['analysis_clock_et']) & (features.index < cutoff)]
    assert not features.date.duplicated().any()
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    files = list(SOURCE.glob('*.py')) + [SOURCE / 'protocol.json']
    files += [SOURCE.parent / f'soxl_soxs_14d5pct_{v}' / n for v, n in [
        ('v2','core.py'),('v2','study.py'),('v3','risk.py'),('v3','risk_ledger.py'),
        ('v4','walkforward.py'),('v4','daily_study.py'),('v6','path_engine.py'),('v6','run_intraday.py'),
        ('v7','breadth.py'),('v7','run_breadth.py'),('v9','continuation.py'),('v10','micro_manager.py'),('v11','uncertainty.py')]]
    core.dump(output / 'registration.json', {'phase': phase, 'started_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(SOURCE / 'protocol.json'), 'preregistration': prereg, 'authorization': authorization,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {str(p.relative_to(core.ROOT)): sha(p) for p in files}})
    for p in files:
        dest = output / 'executed_sources' / p.parent.name / p.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(p.read_bytes())
    baseline = baseline_labels(market, features); stress = baseline_labels(market, features, 2)
    releases = joint_releases(features, baseline)
    core.dump(output / 'baseline_labels.json', list(baseline.values()))
    core.dump(output / 'stress_labels.json', list(stress.values()))
    core.dump(output / 'input_manifest.json', {'full_market': original.manifest, 'observation_cutoff_exclusive': cutoff,
        'source_feature_file': str(source), 'source_feature_sha256': sha(source),
        'feature_sha256': sha(output / 'features.parquet'), 'entry_columns': ENTRY_COLUMNS,
        'restricted_market': {s: {'rows': len(f), 'last_bar_end': f.end_at.max(), 'last_available_at': f.available_at.max()}
                              for s, f in market.bars.items()}})
    histories = {m: [] for m in LEARNERS}; registry = []
    print('prepared', len(features), 'dates;', len(releases), 'joint mature dates', flush=True)
    with threadpool_limits(limits=1):
        for i, (at, row) in enumerate(features.iterrows()):
            record = {'date': row.date, 'at': at, 'cutoff': day_cutoff(at), 'routes': {}}; models = {}
            for method in LEARNERS:
                dates = eligible(releases, record['cutoff'], rolling=method == 'two_part_252')
                rows = examples(features, baseline, dates)
                model, info = fit(rows, method, protocol); models[method] = model
                cal = calibrate(histories[method], baseline, dates, protocol)
                event = {'at': at, 'date': row.date, **decide(forecast(model, row), cal, method)}
                histories[method].append(event)
                record['routes'][method] = {'fit': info, 'calibration': cal, 'event': event,
                    'latest_label_release': max((releases[d] for d in dates), default=None)}
            with gzip.open(output / 'models' / (str(row.date.date()) + '.pkl.gz'), 'wb') as f:
                pickle.dump(models, f)
            core.dump(output / 'training' / (str(row.date.date()) + '.json'), record)
            registry.append({'date': row.date, 'cutoff': record['cutoff'], 'routes': {
                m: {'training_dates': len(v['fit']['dates']), 'first_training_date': min(v['fit']['dates'], default=None),
                    'last_training_date': max(v['fit']['dates'], default=None), 'latest_label_release': v['latest_label_release'],
                    'probability_dates': len(v['calibration']['probability_dates']), 'top_dates': len(v['calibration']['top_dates']),
                    'action': v['event']['action']} for m, v in record['routes'].items()}})
            if (i + 1) % 50 == 0:
                print('trained through', row.date.date(), flush=True)
    core.dump(output / 'predictions.json', histories); core.dump(output / 'training_registry.json', registry)
    test = features[(features.date >= start) & (features.date <= end)]
    actions = {m: {r['at']: r['action'] for r in history} for m, history in histories.items()}
    actions.update(always_soxl={at: 'SOXL' for at in test.index}, always_soxs={at: 'SOXS' for at in test.index},
        trend20={at: direction(row, 'trend20') for at, row in test.iterrows()}, cash={at: 'CASH' for at in test.index})
    actions['double_cost'] = actions[protocol['primary']]
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
        'new_route_case_results_seen_before_decisions_frozen': False, 'development_interval_previously_exposed': True,
        'prediction_sha256': sha(output / 'predictions.json'),
        'actions': {n: [{'at': t, 'action': a[t]} for t in test.index] for n, a in actions.items()}})
    results = {'phase': phase, 'primary_name': protocol['primary'], 'routes': {}}; books = {}
    months = settings['months'] if phase == 'development' else settings['regression_months']
    for name, chosen in actions.items():
        multiple = 2 if name == 'double_cost' else 1
        book = replay(market, test, list((stress if multiple == 2 else baseline).values()), chosen, start, end, cost_multiple=multiple)
        books[name] = book; save_replay(output, name, book); results['routes'][name] = score(book, months)
    required = market.sessions[(market.sessions >= start) & (market.sessions <= end)]
    covered = pd.DatetimeIndex(test.date).equals(required)
    results['coverage'] = {'complete': covered, 'expected_days': len(required), 'observed_days': len(test)}
    results['primary'] = results['routes'][protocol['primary']]
    results['paired_increments'] = {n: paired_uncertainty(books[protocol['primary']][0], books[n][0])
                                   for n in ['direct_expanding', 'mean_expanding', 'two_part_252', 'trend20']}
    if phase == 'development':
        gate = development_gate(results['primary'], covered, protocol); results['development_gate'] = gate
        results['case_regression'] = {'executed': False, 'eligible_after_verification': gate['passed']}
    else:
        results['case_acceptance'] = {'passed': bool(covered and results['primary']['primary_pass'])}
    results.update(goal_complete=False, new_buy_eligible=False, future_nonnegative_month_guarantee=False)
    core.dump(output / 'results.json', results)
    print(json.dumps({n: {'mean': s['primary_average'], 'return': s['continuous_return'], 'pass': s['primary_pass']}
                      for n, s in results['routes'].items()}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--case-after', type=Path)
    args = parser.parse_args(); run(args.output, args.case_after)
