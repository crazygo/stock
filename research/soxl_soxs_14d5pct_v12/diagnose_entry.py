"""Descriptive outcome decomposition, never used to choose thresholds or actions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from entry_value import core, LEARNERS
from verify_continuation import read, labels, key


def diagnose(run):
    run = Path(run); reg = read(run / 'registration.json'); p = reg['preregistration']['protocol']; settings = p[reg['phase']]
    base = labels(run / 'baseline_labels.json'); history = read(run / 'predictions.json')
    out = {'phase': reg['phase'], 'routes': {}, 'counterfactual_dates_are_not_independent_trades': True}
    for name in LEARNERS:
        chosen = [e for e in history[name] if settings['evaluation_start'] <= e['date'][:10] <= settings['evaluation_end']]
        rows = []; unknown = []; top = []
        for e in chosen:
            pair = [base[key(e['at'], s)] for s in core.ETFS]
            if not all(x['complete_window'] and x.get('label_available_at') is not None for x in pair):
                unknown.append(e['date']); continue
            if not e['ready']: continue
            info = read(run / 'training' / (e['date'][:10] + '.json'))['routes'][name]['fit']
            for s in core.ETFS:
                lab = base[key(e['at'], s)]; raw = e['raw'][s]; actual = np.log1p(lab['net_return'])
                rows.append({'date': e['date'], 'symbol': s, 'p': e['probabilities'][s], 'raw_p': raw['p'],
                    'hit': lab['target_hit'], 'p_baseline': info['means'][s]['p'],
                    'failure': raw['failure'], 'failure_baseline': info['means'][s]['failure'],
                    'score': e['scores'][s], 'actual_log': actual})
                if s == e['top_symbol']:
                    top.append({'date': e['date'], 'symbol': s, 'p': e['probabilities'][s], 'hit': lab['target_hit'],
                                'score': e['scores'][s], 'adjusted': e['adjusted'], 'actual': actual, 'action': e['action']})
        f = pd.DataFrame(rows); t = pd.DataFrame(top); metrics = {}
        for s in core.ETFS:
            g = f[f.symbol.eq(s)]; failure = g[g.hit.eq(0)]
            bins = []
            for a, b in zip([0., .2, .4, .6, .8], [.2, .4, .6, .8, 1.00001]):
                group = g[(g.p >= a) & (g.p < b)]
                bins.append({'lo': a, 'hi': min(b, 1.), 'n': len(group), 'predicted': float(group.p.mean()) if len(group) else None,
                             'observed': float(group.hit.mean()) if len(group) else None})
            metrics[s] = {'dates': len(g), 'hits': int(g.hit.sum()), 'mean_probability': float(g.p.mean()),
                'hit_fraction': float(g.hit.mean()), 'brier': float(np.mean((g.p - g.hit) ** 2)),
                'raw_brier': float(np.mean((g.raw_p - g.hit) ** 2)),
                'empirical_training_brier': float(np.mean((g.p_baseline - g.hit) ** 2)),
                'log_return_mse': float(np.mean((g.score - g.actual_log) ** 2)),
                'non_target_dates': len(failure), 'failure_log_mse': float(np.mean((failure.failure - failure.actual_log) ** 2)),
                'failure_training_mean_log_mse': float(np.mean((failure.failure_baseline - failure.actual_log) ** 2)),
                'reliability': bins}
        selected = t[t.action.ne('CASH')]
        out['routes'][name] = {'daily_candidates': len(chosen), 'joint_immature_dates': unknown,
            'ready_mature_dates': len(t), 'per_symbol': metrics,
            'top': {'mean_raw_value': float(t.score.mean()), 'mean_corrected_value': float(t.adjusted.mean()),
                    'mean_actual_log_return': float(t.actual.mean()), 'candidate_buy_dates': len(selected),
                    'buy_predicted_mean': float(selected.adjusted.mean()), 'buy_actual_mean': float(selected.actual.mean()),
                    'buy_hit_fraction': float(selected.hit.mean())},
            'actual_trades_are_reported_separately_in_capital_results': True}
    core.dump(run / 'diagnosis.json', out); print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    diagnose(p.parse_args().run)
