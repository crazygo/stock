"""Re-score frozen inputs with a tail check; preserve the original v1 results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from analyze import metric

ROOT = Path(__file__).resolve().parent
EPS = 1e-8


def correlation(a, b):
    if np.ptp(a) <= EPS or np.ptp(b) <= EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def direction(slope):
    return 1 if slope > EPS else -1 if slope < -EPS else 0


def diagnostics(current, history, hours):
    current, history, hours = map(np.asarray, (current, history, hours))
    # Use actual session hours, not the count of 30m/60m points. No interpolation.
    mask = hours >= hours[-1] / 2
    a, b, x = current[mask], history[mask], hours[mask]
    current_slope, history_slope = (float(np.polyfit(x, y, 1)[0]) for y in (a, b))
    da, db = np.diff(current), np.diff(history)
    signs = lambda values: np.where(values > EPS, 1, np.where(values < -EPS, -1, 0))
    full_r, tail_r = correlation(current, history), correlation(a, b)
    return {
        "full_r_recomputed": full_r,
        "tail_r": tail_r,
        "coefficient": min(full_r, tail_r) if full_r is not None and tail_r is not None else None,
        "tail_first_x": float(x[0]),
        "tail_last_x": float(x[-1]),
        "tail_points": len(x),
        "current_tail_slope": current_slope,
        "history_tail_slope": history_slope,
        "tail_same_direction": direction(current_slope) != 0 and direction(current_slope) == direction(history_slope),
        "current_tail_pct": 100 * math.expm1(float(a[-1] - a[0]) / 100),
        "history_tail_pct": 100 * math.expm1(float(b[-1] - b[0]) / 100),
        "hourly_change_r": correlation(da, db),
        "direction_agreement_count": int(np.sum(signs(da) == signs(db))),
        "direction_comparison_count": len(da),
        "curve_rmse_log_pp": float(np.sqrt(np.mean((current - history) ** 2))),
    }


def passes(sample, diag, threshold=.9):
    return (sample['r'] is not None and sample['r'] >= threshold
            and sample['amplitude_pass'] and diag['tail_r'] is not None
            and diag['tail_r'] >= threshold and diag['tail_same_direction'])


def make_clusters(samples, hours, reference_price):
    if not samples:
        return []
    mask = np.asarray(hours) >= hours[-1] / 2
    if len(samples) == 1:
        groups = [1]
    else:
        curves = np.asarray([s['curve'] for s in samples])
        full, tail = np.corrcoef(curves), np.corrcoef(curves[:, mask])
        distances = np.clip(1 - np.minimum(full, tail), 0, 2)
        np.fill_diagonal(distances, 0)
        groups = fcluster(linkage(squareform(distances, checks=False), method='complete'), .10, criterion='distance')
    clusters = []
    for group in sorted(set(groups)):
        subset = [sample for sample, index in zip(samples, groups) if index == group]
        center = np.median([s['curve'] for s in subset], axis=0)
        medoid = min(subset, key=lambda s: float(np.sum((np.asarray(s['curve']) - center) ** 2)))
        clusters.append({'id': int(group), 'n': len(subset), 'medoid_date': medoid['date'],
                         'dates': [s['date'] for s in subset],
                         'future_24h': metric(subset, 'future_24h', reference_price),
                         'future_rth': metric(subset, 'future_rth', reference_price)})
    return clusters


def main():
    path = ROOT / 'results.json'
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    hours = data['curve_extension']['prefix_x']
    protocol = {
        'version': 'whole_and_tail_shape_match_v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'exploratory_revision_after_review_of_exposed_historical_case_not_independent_validation',
        'source_results_sha256': source_hash,
        'reference_cutoff_et': '2026-10-08 09:00:00',
        'selection': 'v1 whole-curve r >= N and amplitude ratio in [0.5,2], plus tail r >= N and nonflat tail OLS slope signs agree',
        'coefficient': 'min(whole Pearson r, tail Pearson r); range [-1,1], not percent similarity or probability',
        'tail': 'price points at compressed session-hour x >= 29/2; actual observed points x=15..29, no interpolation',
        'threshold_N': .9,
        'flat_slope_epsilon_log_pp_per_session_hour': EPS,
        'direction_agreement': 'diagnostic only; up/down/flat sign agreement across all 30 consecutive price changes, epsilon 1e-8 log pp',
        'hourly_change_correlation_and_rmse': 'diagnostic only, no extra thresholds',
        'plot': 'all whole-curve candidates stay visible; v2 rejected candidates explicitly labeled and excluded from v2 forward aggregates',
        'highlight': 'latest accepted date, else latest whole-curve candidate marked rejected',
        'cluster': 'fixed N=.9 complete linkage; distance=1-min(full pairwise r, tail pairwise r), cut .10; future labels unused',
        'unknowns': 'undefined/flat tail is rejected explicitly; incomplete future remains unknown',
        'original_results_preserved': True,
        'future_labels_used_for_selection': False,
        'source_reference': 'https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html',
    }
    (ROOT / 'shape_match_v2_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    result = {'protocol': protocol, 'stocks': {}}
    for stock in data['stocks']:
        if not stock.get('current_curve'):
            continue
        records = {sample['date']: diagnostics(stock['current_curve'], sample['curve'], hours) for sample in stock['samples']}
        selected = [sample for sample in stock['samples'] if passes(sample, records[sample['date']])]
        candidates = [sample for sample in stock['samples'] if sample['r'] is not None and sample['r'] >= .9 and sample['amplitude_pass']]
        result['stocks'][stock['code']] = {
            'samples': records, 'whole_candidates_n': len(candidates), 'accepted_n': len(selected),
            'accepted_dates': sorted([s['date'] for s in selected], reverse=True),
            'clusters': make_clusters(selected, hours, stock['reference_price']),
            'stats': {h: metric(selected, h, stock['reference_price']) for h in ('future_24h', 'future_rth')},
        }
    (ROOT / 'shape_match_v2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    print(json.dumps({code: {'candidates': s['whole_candidates_n'], 'accepted': s['accepted_n']} for code, s in result['stocks'].items() if s['whole_candidates_n']}))


if __name__ == '__main__':
    main()
