"""Pre-register daily expert blending, freeze actions, then evaluate capital."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "soxl_soxs_14d5pct_v18"))
from selection import validate_selection

from adaptive_blend import (core, ROUTES, bank, expert_bank, matured_pool, forecast,
                            risk_decision, day_cutoff, restrict_market, development_gate)
from allocation_ledger import replay
from verify_continuation import read, labels
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    p = read(SOURCE / 'protocol.json'); prereg = read(SOURCE / 'evidence/preregistration.json')
    assert prereg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    for name, digest in prereg['code_sha256'].items():
        assert sha(core.ROOT / name) == digest
    authorization = validate_selection()
    phase = 'cases'; settings = p[phase]
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
    features = pd.read_parquet(source / 'features.parquet'); panel = pd.read_parquet(source / 'panel.parquet')
    assert (features.index < cutoff).all() and features.clock.eq('09:25:30').all()
    histories = read(source / 'predictions.json'); events = histories['regret_regime']
    experts = expert_bank(panel, events); risks = bank(panel, events)
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; retain all prior attempts')
    for sub in ['', 'training', 'executed_sources']:
        (output / sub).mkdir(parents=True, exist_ok=True)
    files = [*SOURCE.glob('*.py'), SOURCE / 'protocol.json', SOURCE.parent / 'soxl_soxs_14d5pct_v18/adaptive_blend.py', SOURCE.parent / 'soxl_soxs_14d5pct_v18/verify_blend.py']
    files += [SOURCE.parent / f'soxl_soxs_14d5pct_{v}' / n for v, n in [
        ('v2', 'core.py'), ('v2', 'study.py'), ('v3', 'risk.py'), ('v3', 'risk_ledger.py'),
        ('v6', 'run_intraday.py'), ('v7', 'run_breadth.py'), ('v11', 'uncertainty.py'),
        ('v14', 'paired_entry.py'), ('v15', 'allocation.py'), ('v15', 'allocation_ledger.py'),
        ('v15', 'verify_allocation.py'), ('v16', 'anchored_allocation.py')]]
    core.dump(output / 'registration.json', {'phase': phase, 'started_at': datetime.now(timezone.utc).isoformat(),
        'preregistration': prereg, 'protocol_sha256': sha(SOURCE / 'protocol.json'), 'authorization': authorization,
        'source_sha256': {str(f.relative_to(core.ROOT)): sha(f) for f in files}})
    for f in files:
        dest = output / 'executed_sources' / f.parent.name / f.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(f.read_bytes())
    for name, frame in [('features', features), ('expert_bank', experts), ('distribution_bank', risks)]:
        frame.to_parquet(output / (name + '.parquet'), compression='zstd', compression_level=7)
    core.dump(output / 'input_manifest.json', {'source_run': str(source.resolve()),
        'source_files_sha256': {f: sha(source / f) for f in needed}, 'source_delivery_sha256': sha(delivery_path),
        'full_market': original.manifest, 'observation_cutoff_exclusive': cutoff,
        'bank_files_sha256': {n: sha(output / n) for n in ['features.parquet', 'expert_bank.parquet', 'distribution_bank.parquet']}})
    predictions = {n: [] for n in [*ROUTES, 'blend_no_risk', 'source_full', 'source_raw_full', 'cash']}
    registry = []
    for e in events:
        at = pd.Timestamp(e['at']); date = pd.Timestamp(e['date']); dc = day_cutoff(at)
        pool = matured_pool(experts, dc) if e['ready'] else experts.iloc[:0]
        current = experts.loc[at] if e['ready'] else None
        record = {'at': at, 'date': date, 'cutoff': dc, 'routes': {}}
        for name, (mode, method) in ROUTES.items():
            blend = forecast(pool, current, e, mode, method)
            decision = risk_decision(risks, dc, at, blend)
            event = {'at': at, 'date': date, **{k: v for k, v in decision.items() if k != 'fit'},
                     'adjusted': blend['adjusted']}
            record['routes'][name] = {'blend': blend, 'risk': decision['fit'], 'event': event}
            predictions[name].append(event)
        primary = record['routes'][p['primary']]['blend']
        buy = primary['ready'] and max(primary['adjusted'][s] for s in core.ETFS) > 0
        predictions['blend_no_risk'].append({'at': at, 'date': date, 'ready': primary['ready'],
            'action': primary['top_symbol'] if buy else 'CASH', 'allocation': float(buy)})
        for name, old in [('source_full', 'regret_regime'), ('source_raw_full', 'same_primary_raw')]:
            saved = histories[old][len(predictions[name])]
            buy = saved['action'] != 'CASH'
            predictions[name].append({'at': at, 'date': date, 'ready': saved['ready'],
                                     'action': saved['action'], 'allocation': float(buy)})
        predictions['cash'].append({'at': at, 'date': date, 'ready': True, 'action': 'CASH', 'allocation': 0.})
        core.dump(output / 'training' / (str(date.date()) + '.json'), record)
        registry.append({'date': date, 'cutoff': dc, 'history_dates': len(pool),
            'latest_label_release': pool.release.max() if len(pool) else None,
            'routes': {n: {'ready': v['blend']['ready'], 'alpha_calibrated': {k: h['alpha_calibrated'] for k, h in v['blend']['heads'].items()},
                          'action': v['event']['action'], 'reason': v['event']['reason']} for n, v in record['routes'].items()}})
    predictions['double_cost'] = predictions[p['primary']]
    core.dump(output / 'predictions.json', predictions); core.dump(output / 'training_registry.json', registry)
    test = features[(features.date >= start) & (features.date <= end)]
    frozen = {n: [e for e in h if start <= str(e['date'].date()) <= end] for n, h in predictions.items()}
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
        'new_v19_capital_outcomes_seen': False, 'older_versions_exposed_intervals': True,
        'predictions_sha256': sha(output / 'predictions.json'), 'routes': frozen})
    base = labels(source / 'baseline_labels.json'); stress = labels(source / 'stress_labels.json')
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
    print(json.dumps({n: {'mean': v['primary_average'], 'return': v['continuous_return'], 'pass': v['primary_pass']}
                      for n, v in results['routes'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
