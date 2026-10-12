"""Post-result diagnostics and prefix audit; no training or action changes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from early_training import core, ROUTES
from verify_continuation import read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_run(run):
    run = Path(run); panel = pd.read_parquet(run/'panel.parquet')
    diag = read(run/'diagnosis.json'); frozen = read(run/'decisions_frozen.json')['routes']
    cutoff = pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    locked = {n: sha(run/n) for n in ['predictions.json', 'decisions_frozen.json', 'results.json']}
    checks = 0; weight_rows = []
    for name in ROUTES:
        rows = []; unknown = []
        for e in frozen[name]:
            at = pd.Timestamp(e['at']); row = panel.loc[at]
            rec = read(run/'training'/(e['date'][:10]+'.json')); fc = rec['forecasts'][name]['event']
            if name == 'early_base_cal':
                dates = pd.DatetimeIndex(rec['fit']['dates']); day = pd.Timestamp(e['date'])
                weights = np.array(rec['fit']['parent_weights']); weights /= sum(weights)
                recent = set(panel.loc[panel.date < day, 'date'].tail(20))
                caldates = pd.DatetimeIndex(rec['forecasts'][name]['calibration']['dates'])
                weight_rows.append({'date': e['date'],
                    'base_weighted_age_calendar_days': float(weights @ ((day-dates).days.to_numpy())),
                    'base_last20_known_market_dates_weight': float(sum(w for d, w in zip(dates, weights) if d in recent)),
                    'calibration_last20_known_market_dates_fraction': float(sum(d in recent for d in caldates)/len(caldates))})
            if not fc['ready']:
                continue
            if pd.isna(row.release) or row.release >= cutoff:
                unknown.append(e['date'][:10]); continue
            sign = 1 if fc['top_symbol'] == 'SOXL' else -1
            actual = row.C+sign*row.D; state = ('support' if row.relative_5 > 0 and row.breadth_5 > 0 and row.volume_direction_diffusion > 0 else
                'weakness' if row.relative_5 < 0 and row.breadth_5 < 0 and row.volume_direction_diffusion < 0 else 'mixed')
            rows.append([fc['adjusted'][fc['top_symbol']], actual, (sign == 1) == (row.D >= 0),
                (fc['adjusted']['D']-row.D)**2, row.C+abs(row.D)-actual, e['allocation'] > 0, state])
        assert unknown == diag['routes'][name]['unknown_dates']
        frame = pd.DataFrame(rows, columns=['predicted','actual','correct','mse','regret','allowed','state'])
        groups = {'mature_opportunities': frame, 'allowed_mature_opportunities': frame[frame.allowed.eq(True)]}
        groups.update({s: frame[frame.state == s] for s in ['support', 'weakness', 'mixed']})
        for group, f in groups.items():
            saved = diag['routes'][name][group] if group.endswith('opportunities') else diag['routes'][name]['states'][group]
            assert saved['dates'] == len(f)
            if not len(f):
                continue
            expected = {'direction_correct': int(f.correct.sum()), 'predicted_mean_log': f.predicted.mean(),
                'actual_mean_log': f.actual.mean(), 'D_mean_squared_error': f.mse.mean(), 'regret_mean_log': f.regret.mean()}
            for k, v in expected.items():
                np.testing.assert_allclose(saved[k], v, atol=1e-12, rtol=1e-10); checks += 1
    f = pd.DataFrame(weight_rows); f['month'] = f.date.str[:7]
    aggregate = f.groupby('month').mean(numeric_only=True).to_dict(orient='index')
    assert all(sha(run/n) == h for n, h in locked.items())
    receipt = {'passed': True, 'post_result_read_only': True, 'diagnostic_statistics_checked': checks,
        'diagnosis_sha256': sha(run/'diagnosis.json'), 'primary_training_weight_diagnostic': aggregate,
        'weight_diagnostic_does_not_establish_failure_cause': True, 'frozen_inputs_unchanged': locked}
    core.dump(run/'diagnosis_verification.json', receipt)
    return receipt


def audit(development, cases):
    development, cases = Path(development), Path(cases)
    a, b = [read(p/'predictions.json') for p in [development, cases]]
    assert a.keys() == b.keys(); counts = {}
    for name in a:
        prior = [e for e in b[name] if e['date'][:10] < '2026-05-01']
        assert a[name] == prior; counts[name] = len(prior)
    receipt = {'passed': True, 'all_routes_exact': True, 'event_counts': counts,
        'total_unchanged_events': sum(counts.values()),
        'development_predictions_sha256': sha(development/'predictions.json'),
        'case_predictions_sha256': sha(cases/'predictions.json')}
    core.dump(cases/'historical_prefix_verification.json', receipt)
    print({'prefix': receipt, 'development': audit_run(development), 'cases': audit_run(cases)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True); parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args(); audit(args.development, args.cases)
