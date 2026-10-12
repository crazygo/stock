"""Registered descriptive state-conditioned diagnostics; no new actions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from local_training import core, ROUTES
from verify_continuation import read, labels, key


def diagnose(run):
    run = Path(run); manifest = read(run/'input_manifest.json'); source = Path(manifest['source_run'])
    base = labels(source/'baseline_labels.json'); panel = pd.read_parquet(run/'panel.parquet')
    predictions = read(run/'decisions_frozen.json')['routes']; cutoff = pd.Timestamp(manifest['observation_cutoff_exclusive'])
    records = {e['at']: read(run/'training'/(e['date'][:10]+'.json')) for e in predictions['state_base_cal']}
    out = {'registered_diagnostic_only': True, 'decisions_changed': False,
           'full_window_common_diagnostic_denominator': True, 'overlapping_candidates_are_not_capital_returns': True, 'routes': {}}
    for name in ROUTES:
        rows = []; unknown = []; ready_count = 0; slopes = []; lag = []
        for capital in predictions[name]:
            rec = records[capital['at']]; saved = rec['forecasts'][name]; event = saved['event']
            last = rec['feedback']['last_early_date']
            if last is not None:
                lag.append((pd.Timestamp(capital['date'])-pd.Timestamp(last)).days)
            if not event['ready']:
                continue
            ready_count += 1; slopes.append(saved['calibration']['maps']['D']['coefficient'])
            at = pd.Timestamp(capital['at']); pair = [base[key(at, s)] for s in core.ETFS]
            if not all(l['complete_window'] and l['label_available_at'] is not None and l['label_available_at'] < cutoff for l in pair):
                unknown.append(capital['date'][:10]); continue
            actual = np.log1p([l['net_return'] for l in pair]); chosen = int(event['top_symbol'] == 'SOXS')
            row = panel.loc[at]; state = ('support' if row.relative_5 > 0 and row.breadth_5 > 0 and row.volume_direction_diffusion > 0 else
                'weakness' if row.relative_5 < 0 and row.breadth_5 < 0 and row.volume_direction_diffusion < 0 else 'mixed')
            rows.append({'state': state, 'predicted': event['adjusted'][event['top_symbol']], 'actual': actual[chosen],
                'correct': chosen == int(actual[1] > actual[0]), 'regret': max(actual)-actual[chosen],
                'error_D': (event['adjusted']['D']-(actual[0]-actual[1])/2)**2, 'allowed': capital['allocation'] > 0})
        def stats(group):
            return {'dates': len(group), 'direction_correct': int(sum(x['correct'] for x in group)),
                'predicted_mean_log': float(np.mean([x['predicted'] for x in group])),
                'actual_mean_log': float(np.mean([x['actual'] for x in group])),
                'D_mean_squared_error': float(np.mean([x['error_D'] for x in group])),
                'regret_mean_log': float(np.mean([x['regret'] for x in group]))} if group else {'dates': 0}
        out['routes'][name] = {'forecast_ready_dates': ready_count, 'unknown_dates': unknown,
            'median_latest_base_training_date_age_calendar_days': float(np.median(lag)) if lag else None,
            'D_calibration_negative_slope_dates': sum(s < 0 for s in slopes),
            'mature_opportunities': stats(rows), 'allowed_mature_opportunities': stats([x for x in rows if x['allowed']]),
            'states': {s: stats([x for x in rows if x['state'] == s]) for s in ['support', 'weakness', 'mixed']}}
    # The model uses historical states known before this query, without outcome
    # labels in distances. Concentration is not an independent sample count.
    weight_summary = {}
    for mode in ['state', 'global']:
        dates = []; concentration = []; recent_weight = []; weighted_age = []
        for rec in records.values():
            info = rec['fits'][mode]
            if not info['ready']:
                continue
            w = np.asarray(info['parent_weights']); w = w/w.sum()
            day = pd.Timestamp(rec['date']); ds = pd.DatetimeIndex(info['dates'])
            recent = set(panel.loc[panel.date < day, 'date'].tail(20))
            dates.append(rec['date']); concentration.append(float(1/(w@w)))
            recent_weight.append(float(sum(v for d, v in zip(ds, w) if d in recent)))
            weighted_age.append(float(w@((day-ds).days.to_numpy())))
        weight_summary[mode] = {'fitted_dates': len(dates),
            'effective_dates_min': min(concentration) if concentration else None,
            'effective_dates_median': float(np.median(concentration)) if concentration else None,
            'last20_market_dates_mean_weight': float(np.mean(recent_weight)) if recent_weight else None,
            'weighted_age_calendar_days_mean': float(np.mean(weighted_age)) if weighted_age else None}
    out['base_weight_summary'] = weight_summary
    core.dump(run/'diagnosis.json', out); print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    diagnose(parser.parse_args().run)
