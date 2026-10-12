"""Freeze daily early-training forecasts before replaying capital outcomes."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import pickle

import pandas as pd
from threadpoolctl import threadpool_limits
from early_training import (core, ROUTES, terminal_panel, forecast_routes, capital_decision,
                            bank, restrict_market, development_gate, day_cutoff)
from verify_continuation import read, labels
from allocation_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_cases(run, protocol):
    run = Path(run); reg = read(run / 'registration.json')
    result = read(run / 'results.json'); verification = read(run / 'verification.json')
    assert reg['phase'] == 'development' and reg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    assert verification['passed'] and verification['results_sha256'] == sha(run / 'results.json')
    gate = development_gate(result['primary'], result['coverage']['complete'], protocol)
    if not gate['passed']:
        raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run': str(run.resolve()), 'results_sha256': sha(run / 'results.json'), 'gate': gate}


def run(output, case_after=None):
    p = read(SOURCE / 'protocol.json'); prereg = read(SOURCE / 'evidence/preregistration.json')
    assert prereg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    for name, digest in prereg['code_sha256'].items():
        assert sha(core.ROOT / name) == digest
    authorization = authorize_cases(case_after, p) if case_after else None
    phase = 'cases' if case_after else 'development'; settings = p[phase]
    start, end = settings['evaluation_start'], settings['evaluation_end']
    cutoff = pd.Timestamp(end, tz=core.ET) + pd.Timedelta(days=1)
    source = core.ROOT / p['source_runs'][phase]
    delivery_path = (SOURCE.parent / 'soxl_soxs_14d5pct_v14/evidence/delivery_manifest.json'
                     if phase == 'development' else source / 'delivery_manifest.json')
    delivery = read(delivery_path)
    needed = ['features.parquet', 'panel.parquet', 'predictions.json', 'baseline_labels.json',
              'stress_labels.json', 'results.json', 'verification.json', 'input_manifest.json', 'registration.json']
    for name in needed:
        assert sha(source / name) == delivery['cache_outputs_sha256'][name]
    src_manifest = read(source / 'input_manifest.json'); src_verify = read(source / 'verification.json')
    assert src_verify['passed'] and src_verify['results_sha256'] == sha(source / 'results.json')
    assert pd.Timestamp(src_manifest['observation_cutoff_exclusive']) == cutoff
    original = core.load_market(); assert original.manifest == src_manifest['full_market']
    market = restrict_market(original, cutoff)
    features = pd.read_parquet(source / 'features.parquet'); source_panel = pd.read_parquet(source / 'panel.parquet')
    histories = read(source / 'predictions.json'); events = histories['regret_regime']
    base = labels(source / 'baseline_labels.json'); stress = labels(source / 'stress_labels.json')
    panel, receipts = terminal_panel(market, source_panel, base); risks = bank(source_panel, events)
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; retain every attempt')
    for sub in ['', 'training', 'models', 'executed_sources']:
        (output / sub).mkdir(parents=True, exist_ok=True)
    core.dump(output / 'registration.json', {'phase': phase, 'started_at': datetime.now(timezone.utc).isoformat(),
        'preregistration': prereg, 'protocol_sha256': sha(SOURCE / 'protocol.json'), 'authorization': authorization,
        'source_sha256': prereg['code_sha256']})
    for name in prereg['code_sha256']:
        f = core.ROOT / name; dest = output / 'executed_sources' / f.parent.name / f.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(f.read_bytes())
    core.dump(output / 'terminal_receipts.json', receipts)
    for name, frame in [('features', features), ('panel', panel), ('distribution_bank', risks)]:
        frame.to_parquet(output / (name + '.parquet'), compression='zstd', compression_level=7)
    core.dump(output / 'input_manifest.json', {'source_run': str(source.resolve()),
        'source_files_sha256': {f: sha(source / f) for f in needed}, 'source_delivery_sha256': sha(delivery_path),
        'full_market': original.manifest, 'observation_cutoff_exclusive': cutoff,
        'bank_files_sha256': {n: sha(output / n) for n in ['features.parquet', 'panel.parquet', 'distribution_bank.parquet', 'terminal_receipts.json']}})
    predictions = {n: [] for n in [*ROUTES, 'early_no_risk', 'source_full', 'source_raw_full', 'cash']}
    early_history = []; full_history = []; registry = []
    with threadpool_limits(limits=1):
        for i, e in enumerate(events):
            at = pd.Timestamp(e['at']); date = pd.Timestamp(e['date']); dc = day_cutoff(at)
            model, record, raw = forecast_routes(panel, at, e, early_history, full_history)
            record.update(at=at, date=date, cutoff=dc, capital={})
            for name in ROUTES:
                forecast = record['forecasts'][name]['event']
                choice = capital_decision(risks, dc, at, forecast)
                event = {'at': at, 'date': date, **{k: v for k, v in choice.items() if k != 'fit'},
                         'adjusted': forecast['adjusted']}
                predictions[name].append(event); record['capital'][name] = {'risk': choice['fit'], 'event': event}
            primary = record['forecasts'][p['primary']]['event']
            buy = primary['ready'] and primary['action'] != 'CASH'
            predictions['early_no_risk'].append({'at': at, 'date': date, 'ready': primary['ready'],
                'action': primary['top_symbol'] if buy else 'CASH', 'allocation': float(buy)})
            for name, old in [('source_full', 'regret_regime'), ('source_raw_full', 'same_primary_raw')]:
                saved = histories[old][i]; buy = saved['action'] != 'CASH'
                predictions[name].append({'at': at, 'date': date, 'ready': saved['ready'],
                                         'action': saved['action'], 'allocation': float(buy)})
            predictions['cash'].append({'at': at, 'date': date, 'ready': True, 'action': 'CASH', 'allocation': 0.})
            early_history.append({'at': at, 'date': date, 'raw': raw})
            full_history.append({'at': at, 'date': date, 'raw': e['raw']})
            core.dump(output / 'training' / (str(date.date()) + '.json'), record)
            with gzip.open(output / 'models' / (str(date.date()) + '.pkl.gz'), 'wb') as f:
                pickle.dump(model, f)
            registry.append({'date': date, 'feedback': record['feedback'], 'fit_dates': len(record['fit']['dates']),
                'routes': {n: {'calibration_dates': len(v['calibration']['dates']),
                    'calibration_coefficients': {k: m['coefficient'] for k, m in v['calibration']['maps'].items()},
                    'action': record['capital'][n]['event']['action']} for n, v in record['forecasts'].items()}})
            if (i+1) % 75 == 0:
                print('trained through', date.date(), flush=True)
    predictions['double_cost'] = predictions[p['primary']]
    core.dump(output / 'predictions.json', predictions); core.dump(output / 'training_registry.json', registry)
    test = features[(features.date >= start) & (features.date <= end)]
    frozen = {n: [e for e in h if start <= str(e['date'].date()) <= end] for n, h in predictions.items()}
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
        'new_v21_capital_outcomes_seen': False, 'older_versions_exposed_intervals': True,
        'predictions_sha256': sha(output / 'predictions.json'), 'routes': frozen})
    books = {}; results = {'phase': phase, 'primary_name': p['primary'], 'routes': {}}
    months = settings.get('months', settings.get('regression_months'))
    for name, history in frozen.items():
        multiple = 2 if name == 'double_cost' else 1
        actions = {e['at']: e['action'] for e in history}; allocations = {e['at']: e['allocation'] for e in history}
        book = replay(market, test, list((stress if multiple == 2 else base).values()), actions, start, end,
                      cost_multiple=multiple, allocations=allocations)
        books[name] = book; save_replay(output, name, book); results['routes'][name] = score(book, months)
    required = market.sessions[(market.sessions >= start) & (market.sessions <= end)]
    covered = pd.DatetimeIndex(test.date).equals(required)
    results['coverage'] = {'complete': covered, 'expected_days': len(required), 'observed_days': len(test)}
    results['primary'] = results['routes'][p['primary']]
    results['paired_increments'] = {n: paired_uncertainty(books[p['primary']][0], books[n][0])
                                   for n in predictions if n not in [p['primary'], 'cash', 'double_cost']}
    if phase == 'development':
        gate = development_gate(results['primary'], covered, p); results['development_gate'] = gate
        results['case_regression'] = {'executed': False, 'eligible_after_verification': gate['passed']}
    else:
        results['case_acceptance'] = {'passed': bool(covered and results['primary']['primary_pass'])}
    results.update(goal_complete=False, new_buy_eligible=False, future_nonnegative_month_guarantee=False)
    core.dump(output / 'results.json', results)
    print({n: {'mean': v['primary_average'], 'total': v['continuous_return']} for n, v in results['routes'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--case-after', type=Path)
    args = parser.parse_args(); run(args.output, args.case_after)
