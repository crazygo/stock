"""Post-run attribution only: preserve decisions and identify calibration errors.

Written after the v17 case result. No thresholds, models or replacement trades
are fitted here. Overlapping daily opportunities are not capital returns.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(rows):
    f = pd.DataFrame(rows)
    if f.empty:
        return {'mature_dates': 0}
    out = {'mature_dates': len(f), 'direction_flips': int(f.flipped.sum()),
           'negative_D_calibration_dates': int(f.D_coefficient.lt(0).sum())}
    for name in ['raw', 'calibrated']:
        actual = f[name + '_actual_log']
        prediction = f[name + '_predicted_log']
        out[name] = {'SOXL_dates': int(f[name + '_SOXL'].sum()),
            'pair_direction_correct': int(f[name + '_correct'].sum()),
            'mean_predicted_log': float(prediction.mean()),
            'mean_actual_log': float(actual.mean()),
            'mean_regret_log': float((f.oracle_log - actual).mean()),
            'mean_squared_error': float(np.mean((prediction - actual) ** 2))}
    return out


def diagnose(run):
    run = Path(run)
    source = Path(read(run / 'input_manifest.json')['source_run'])
    reg = read(run / 'registration.json')
    settings = reg['preregistration']['protocol']['cases']
    cutoff = pd.Timestamp(settings['evaluation_end'], tz='America/New_York') + pd.Timedelta(days=1)
    guarded = [run / f for f in ['decisions_frozen.json', 'predictions.json', 'results.json']]
    guarded += [source / f for f in ['predictions.json', 'results.json', 'panel.parquet']]
    before = {str(f): sha(f) for f in guarded}
    panel = pd.read_parquet(source / 'panel.parquet')
    events = read(source / 'predictions.json')['regret_regime']
    registry = {e['date']: e for e in read(source / 'training_registry.json')}
    by_at = {pd.Timestamp(e['at']): e for e in events}
    rows = []
    unknown = []
    negative_slope_dates = []
    for e in events:
        day = e['date'][:10]
        if not settings['evaluation_start'] <= day <= settings['evaluation_end'] or not e['ready']:
            continue
        p = panel.loc[pd.Timestamp(e['at'])]
        coefficient = registry[e['date']]['routes']['regret_regime']['calibration_coefficients']['D']
        if coefficient < 0:
            negative_slope_dates.append(day)
        if pd.isna(p.release) or p.release >= cutoff:
            unknown.append(day)
            continue
        # Fixed zero-sign descriptions, not learned trading state thresholds.
        state = ('positive_relative_and_breadth' if p.relative_5 > 0 and p.breadth_5 > 0
                 else 'negative_relative_and_breadth' if p.relative_5 < 0 and p.breadth_5 < 0
                 else 'mixed_or_zero')
        row = {'date': day, 'month': day[:7], 'state': state,
               'flipped': (e['raw']['D'] >= 0) != (e['adjusted']['D'] >= 0),
               'D_coefficient': coefficient, 'oracle_log': float(p.C + abs(p.D))}
        for name, prediction in [('raw', e['raw']), ('calibrated', e['adjusted'])]:
            sign = 1 if prediction['D'] >= 0 else -1
            row.update({name + '_SOXL': sign == 1,
                name + '_correct': (sign == 1) == (p.D >= 0),
                name + '_predicted_log': prediction['C'] + abs(prediction['D']),
                name + '_actual_log': float(p.C + sign * p.D)})
        rows.append(row)
    trades = pd.read_csv(run / 'anchor_local_full_trades.csv')
    actual = []
    for t in trades.to_dict('records'):
        at = pd.Timestamp(t['decision_at']); p = panel.loc[at]; e = by_at[at]
        assert e['top_symbol'] == t['symbol'] and t['allocation'] == 1
        actual.append({'entry_date': t['entry_date'], 'symbol': t['symbol'],
            'reason': t['reason'], 'net_return': t['net_return'], 'exit_at': t['exit_at'],
            'raw_top': 'SOXL' if e['raw']['D'] >= 0 else 'SOXS',
            'raw_D': e['raw']['D'], 'calibrated_D': e['adjusted']['D'],
            'predicted_net_log': e['adjusted'][t['symbol']],
            'causal_relative_5': float(p.relative_5), 'causal_breadth_5': float(p.breadth_5),
            'causal_volume_direction_diffusion': float(p.volume_direction_diffusion),
            'full_label_release': str(p.release)})
    training = []
    for month in ['2026-05', '2026-07', '2026-09']:
        e = next(e for e in registry.values() if e['date'].startswith(month))
        assert pd.Timestamp(e['latest_label_release']) < pd.Timestamp(e['cutoff'])
        training.append({k: e[k] for k in ['date', 'cutoff', 'first_training_date', 'last_training_date', 'latest_label_release']}
                        | {'training_dates': e['routes']['regret_regime']['training_dates']})
    source_results = read(source / 'results.json')
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
        'post_run_attribution_only': True, 'decisions_changed': False,
        'overlapping_opportunities_are_not_capital_returns': True,
        'pair_direction_is_relative_payoff_not_TP_probability': True,
        'state_descriptions_are_not_an_acceptance_or_new_strategy': True,
        'maturity_cutoff_exclusive': str(cutoff), 'unknown_dates_retained': unknown,
        'negative_D_calibration_dates_all_cases': len(negative_slope_dates),
        'overall': summary(rows),
        'monthly': {m: summary([r for r in rows if r['month'] == m]) for m in sorted({r['month'] for r in rows})},
        'causal_state_descriptions': {s: summary([r for r in rows if r['state'] == s]) for s in sorted({r['state'] for r in rows})},
        'actual_primary_trades': actual,
        'original_registered_control_books': {n: source_results['routes'][n] for n in ['same_primary_raw', 'always_SOXL', 'trend20']},
        'daily_training_examples': training, 'input_sha256': before}
    assert before == {str(f): sha(f) for f in guarded}
    (run / 'failure_attribution.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'overall': report['overall'], 'negative_slope_dates': len(negative_slope_dates),
                      'unknown_dates': len(unknown), 'actual_trades': len(actual), 'inputs_unchanged': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    diagnose(parser.parse_args().run)
