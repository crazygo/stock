"""Post-result read-only audit of prefixes and diagnostics; never changes actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from resolved_feedback import core
from verify_continuation import key, labels, read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_diagnostics(run):
    run = Path(run)
    manifest = read(run / 'input_manifest.json')
    source = Path(manifest['source_run'])
    base = labels(source / 'baseline_labels.json')
    frozen = read(run / 'decisions_frozen.json')['routes']
    diagnosis = read(run / 'diagnosis.json')
    cutoff = pd.Timestamp(manifest['observation_cutoff_exclusive'])
    locked = {n: sha(run / n) for n in ['predictions.json', 'decisions_frozen.json', 'results.json']}
    metrics = 0
    outcomes = {}
    for name, summary in diagnosis['routes'].items():
        rows = []
        unknown = []
        ages = []
        waiting = expired = 0
        for event in frozen[name]:
            at = pd.Timestamp(event['at'])
            fit = read(run / 'training' / (event['date'][:10] + '.json'))['routes'][name]['blend']
            feedback = fit['feedback']
            if feedback['last_selected'] is not None:
                ages.append((pd.Timestamp(event['date']) - pd.Timestamp(feedback['last_selected'])).days)
            waiting += feedback['known_but_waiting_behind_pending'] > 0
            expired += feedback['expired_unresolved_prefix']
            if not fit['ready']:
                continue
            pair = [base[key(at, s)] for s in core.ETFS]
            mature = all(p['complete_window'] and p['label_available_at'] is not None
                         and p['label_available_at'] < cutoff for p in pair)
            if not mature:
                unknown.append(event['date'][:10])
                continue
            actual = np.log1p([p['net_return'] for p in pair])
            chosen = int(fit['top_symbol'] == 'SOXS')
            predicted = fit['adjusted'][fit['top_symbol']]
            direction = (actual[0] - actual[1]) / 2
            rows.append({'at': event['at'], 'predicted': predicted, 'actual': actual[chosen],
                         'error_D': (fit['adjusted']['D'] - direction) ** 2,
                         'regret': max(actual) - actual[chosen],
                         'correct': chosen == int(actual[1] > actual[0]),
                         'allowed': event['allocation'] > 0, 'symbol': fit['top_symbol']})
        assert unknown == summary['unknown_dates']
        expected_feedback = {'median_latest_training_date_age_calendar_days': float(np.median(ages)),
            'maximum_latest_training_date_age_calendar_days': max(ages),
            'dates_with_known_results_waiting_behind_pending': int(waiting),
            'expired_unresolved_prefix_dates': int(expired)}
        assert expected_feedback == summary['feedback']
        for group, selected in [('mature_opportunities', rows),
                                ('allowed_mature_opportunities', [r for r in rows if r['allowed']])]:
            old = summary[group]
            assert len(selected) == old['dates']
            if not selected:
                continue
            expected = {'predicted_mean_log': np.mean([r['predicted'] for r in selected]),
                        'actual_mean_log': np.mean([r['actual'] for r in selected]),
                        'mean_squared_error': np.mean([(r['actual'] - r['predicted']) ** 2 for r in selected]),
                        'D_mean_squared_error': np.mean([r['error_D'] for r in selected]),
                        'mean_regret_log': np.mean([r['regret'] for r in selected]),
                        'direction_correct': sum(r['correct'] for r in selected)}
            for k, value in expected.items():
                np.testing.assert_allclose(old[k], value, atol=1e-12, rtol=1e-10)
                metrics += 1
        outcomes[name] = {r['at']: r for r in rows}
    primary = outcomes['terminal_frontier_hard']
    previous = outcomes['full_window_hard']
    common = sorted(primary.keys() & previous.keys())
    changed = [at for at in common if primary[at]['symbol'] != previous[at]['symbol']]
    comparison = {'common_mature_dates': len(common), 'changed_direction_dates': len(changed),
        'earlier_feedback_correct_on_changed_dates': sum(primary[at]['correct'] for at in changed),
        'old_full_window_correct_on_changed_dates': sum(previous[at]['correct'] for at in changed),
        'earlier_feedback_mean_log_on_changed_dates': float(np.mean([primary[at]['actual'] for at in changed])) if changed else None,
        'old_full_window_mean_log_on_changed_dates': float(np.mean([previous[at]['actual'] for at in changed])) if changed else None,
        'not_independent_trades_or_causal_capital_attribution': True}
    assert all(sha(run / n) == h for n, h in locked.items())
    receipt = {'passed': True, 'post_result_audit_only': True, 'raw_label_metrics_checked': metrics,
        'diagnosis_sha256': sha(run / 'diagnosis.json'), 'forecast_source_comparison': comparison,
        'frozen_files_unchanged': locked}
    core.dump(run / 'diagnosis_verification.json', receipt)
    return receipt


def verify(development, cases):
    development, cases = Path(development), Path(cases)
    left, right = [read(p / 'predictions.json') for p in [development, cases]]
    assert set(left) == set(right)
    counts = {}
    for name in left:
        a, b = [[e for e in h[name] if e['date'][:10] < '2026-05-01'] for h in [left, right]]
        assert a == b
        counts[name] = len(a)
    receipt = {'passed': True, 'all_routes_checked': True, 'event_counts': counts,
        'total_unchanged_events': sum(counts.values()),
        'development_predictions_sha256': sha(development / 'predictions.json'),
        'case_predictions_sha256': sha(cases / 'predictions.json')}
    core.dump(cases / 'historical_prefix_verification.json', receipt)
    print(json.dumps({'prefix': receipt,
        'development': check_diagnostics(development), 'cases': check_diagnostics(cases)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    verify(args.development, args.cases)
