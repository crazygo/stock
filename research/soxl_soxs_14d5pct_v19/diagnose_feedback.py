"""Post-result audit of feedback waiting time and cost-sensitive exits.

This does not release any label early or create a new trading rule. The source
contract still waits for both complete 14-day windows in all v18/v19 fits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(run):
    run = Path(run); source = Path(read(run / 'input_manifest.json')['source_run'])
    protected = [run / x for x in ['results.json', 'predictions.json', 'decisions_frozen.json']]
    before = {str(f): sha(f) for f in protected}
    records = read(source / 'baseline_labels.json'); groups = {}
    for r in records:
        if r['decision_at'][:10] < '2026-05-01':
            continue
        groups.setdefault(r['decision_at'], {})[r['symbol']] = r
    pairs = []; unresolved = []
    for at, pair in groups.items():
        if set(pair) != {'SOXL', 'SOXS'} or not all(
            r.get('prefix_complete') and r.get('exit_at') is not None
            and r.get('reason') in ['target', 'risk_stop', 'expiry']
            and r.get('net_return') is not None and np.isfinite(r['net_return'])
            for r in pair.values()
        ):
            unresolved.append(at); continue
        terminal = max(pd.Timestamp(r['exit_at']) for r in pair.values()) + pd.Timedelta(seconds=1)
        complete = all(r.get('complete_window') and r.get('label_available_at') for r in pair.values())
        release = max(pd.Timestamp(r['label_available_at']) for r in pair.values()) if complete else None
        pairs.append({'at': at, 'joint_terminal_bar_available': str(terminal),
            'registered_full_window_release': str(release) if release is not None else None,
            'waiting_calendar_days': (release - terminal).total_seconds() / 86400 if release is not None else None,
            'outcomes': {s: {'reason': r['reason'], 'exit_at': r['exit_at']} for s, r in pair.items()}})
    primary = pd.read_csv(run / 'state_hard_trades.csv')
    stress = pd.read_csv(run / 'double_cost_trades.csv')
    joined = primary.merge(stress, on='decision_at', suffixes=('_base', '_stress'))
    changed = joined[joined.reason_base != joined.reason_stress]
    cols = ['decision_at', 'symbol_base', 'reason_base', 'reason_stress', 'net_return_base',
            'net_return_stress', 'exit_at_base', 'exit_at_stress']
    registry = [r for r in read(run / 'training_registry.json') if r['date'] >= '2026-05-01']
    switching = [r['date'][:10] for r in registry if r['routes']['state_hard']['alpha_calibrated'].get('D') == 0]
    by_month = {}
    for month in sorted({r['date'][:7] for r in registry}):
        subset = [r for r in registry if r['date'].startswith(month)]
        by_month[month] = {'dates': len(subset), **{k + '_raw_dates': sum(
            r['routes']['state_hard']['alpha_calibrated'].get(k) == 0 for r in subset) for k in ['C', 'D']}}
    wait = [p['waiting_calendar_days'] for p in pairs if p['waiting_calendar_days'] is not None]
    report = {'post_result_diagnostic_only': True, 'new_predictions_or_earlier_training_labels_created': False,
        'interpretation': 'Terminal-bar metadata motivates checking a future protocol with truly observed joint outcomes. This diagnostic does not certify faster-release training; unresolved observations, prefix completeness, availability and informative resolution delays need separate treatment.',
        'source_label_sha256': sha(source / 'baseline_labels.json'), 'frozen_inputs_sha256': before,
        'candidate_dates': len(groups), 'both_terminal_prefixes_observed': len(pairs),
        'unresolved_dates_retained': unresolved, 'pairs': pairs,
        'complete_window_pairs_in_wait_summary': len(wait),
        'registered_release_wait_days_median': float(np.median(wait)),
        'registered_release_wait_days_range': [min(wait), max(wait)],
        'first_raw_D_selection_date': switching[0] if switching else None, 'head_switches_by_month': by_month,
        'matched_actual_trades': len(joined), 'cost_sensitive_reason_changes': changed[cols].to_dict('records')}
    assert before == {str(f): sha(f) for f in protected}
    (run / 'feedback_diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print({k: report[k] for k in ['candidate_dates', 'both_terminal_prefixes_observed',
        'registered_release_wait_days_median', 'registered_release_wait_days_range',
        'first_raw_D_selection_date', 'head_switches_by_month', 'cost_sensitive_reason_changes']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    diagnose(parser.parse_args().run)
