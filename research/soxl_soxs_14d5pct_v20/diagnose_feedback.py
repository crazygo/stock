"""Registered blend-error, abstention and state diagnostics; no new decisions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from resolved_feedback import core, ROUTES
from verify_continuation import read


def diagnose(run):
    run = Path(run); reg = read(run / 'registration.json'); p = reg['preregistration']['protocol']
    settings = p[reg['phase']]; cutoff = pd.Timestamp(settings['evaluation_end'], tz=core.ET) + pd.Timedelta(days=1)
    source = Path(read(run / 'input_manifest.json')['source_run']); panel = pd.read_parquet(source / 'panel.parquet')
    frozen = read(run / 'decisions_frozen.json')['routes']; records = {}
    for e in frozen[p['primary']]:
        records[e['date']] = read(run / 'training' / (e['date'][:10] + '.json'))
    out = {'registered_diagnostic_only': True, 'decisions_changed': False,
           'overlapping_opportunities_are_not_capital_returns': True, 'routes': {}}
    for name in ROUTES:
        rows = []; unknown = []; alphas = {'C': [], 'D': []}; reason_counts = {}; feedback = []
        for e in frozen[name]:
            b = records[e['date']]['routes'][name]['blend']
            feedback.append({'date': e['date'], **b['feedback']})
            reason_counts[e['reason']] = reason_counts.get(e['reason'], 0) + 1
            if not b['ready']:
                continue
            for k in alphas:
                alphas[k].append(b['heads'][k]['alpha_calibrated'])
            row = panel.loc[pd.Timestamp(e['at'])]
            if pd.isna(row.release) or row.release >= cutoff:
                unknown.append(e['date'][:10]); continue
            symbol = b['top_symbol']; sign = 1 if symbol == 'SOXL' else -1
            actual = float(row.C + sign * row.D); pred = b['adjusted'][symbol]
            state = ('support' if row.relative_5 > 0 and row.breadth_5 > 0 and row.volume_direction_diffusion > 0 else
                     'weakness' if row.relative_5 < 0 and row.breadth_5 < 0 and row.volume_direction_diffusion < 0 else 'mixed')
            rows.append({'date': e['date'][:10], 'state': state, 'predicted': pred, 'actual': actual,
                'error_D': (b['adjusted']['D'] - row.D) ** 2, 'regret': float(row.C + abs(row.D) - actual),
                'allowed': e['allocation'] > 0, 'direction_correct': (sign == 1) == (row.D >= 0)})
        f = pd.DataFrame(rows, columns=['date','state','predicted','actual','error_D','regret','allowed','direction_correct'])
        def stats(frame):
            return {'dates': len(frame), 'predicted_mean_log': float(frame.predicted.mean()),
                'actual_mean_log': float(frame.actual.mean()), 'mean_squared_error': float(np.mean((frame.predicted - frame.actual) ** 2)),
                'D_mean_squared_error': float(frame.error_D.mean()), 'mean_regret_log': float(frame.regret.mean()),
                'direction_correct': int(frame.direction_correct.sum())} if len(frame) else {'dates': 0}
        lags = [(pd.Timestamp(x['date'])-pd.Timestamp(x['last_selected'])).days for x in feedback if x['last_selected'] is not None]
        out['routes'][name] = {'feedback': {'median_latest_training_date_age_calendar_days': float(np.median(lags)) if lags else None, 'maximum_latest_training_date_age_calendar_days': max(lags) if lags else None, 'dates_with_known_results_waiting_behind_pending': sum(x['known_but_waiting_behind_pending']>0 for x in feedback), 'expired_unresolved_prefix_dates': sum(x['expired_unresolved_prefix'] for x in feedback)}, 'evaluation_dates': len(frozen[name]), 'forecast_ready_dates': len(alphas['C']),
            'unknown_dates': unknown, 'reasons': reason_counts, 'mature_opportunities': stats(f),
            'allowed_mature_opportunities': stats(f[f.allowed.eq(True)]),
            'state_descriptions': {s: stats(g) for s, g in f.groupby('state')},
            'alpha_calibrated': {k: {'min': min(a), 'mean': float(np.mean(a)), 'max': max(a),
                'raw_endpoint_dates': sum(x == 0 for x in a), 'calibrated_endpoint_dates': sum(x == 1 for x in a)}
                for k, a in alphas.items() if a}}
    core.dump(run / 'diagnosis.json', out)
    print({n: {'mature': v['mature_opportunities'], 'alpha': v['alpha_calibrated']} for n, v in out['routes'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    diagnose(parser.parse_args().run)
