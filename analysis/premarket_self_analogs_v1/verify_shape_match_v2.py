"""Independent scalar checks of frozen-input tail diagnostics and selection."""
from pathlib import Path
import hashlib
import json
import math
import statistics
from build_shape_match_v2 import passes

ROOT = Path(__file__).resolve().parent
EPS = 1e-8


def corr(a, b):
    if max(a) - min(a) <= EPS or max(b) - min(b) <= EPS:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denominator


def slope(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    return sum((t - mx) * (v - my) for t, v in zip(x, y)) / sum((t - mx) ** 2 for t in x)


def sign(value):
    return 1 if value > EPS else -1 if value < -EPS else 0


def close(actual, expected):
    assert actual is None if expected is None else actual is not None and abs(actual - expected) < 1e-7


def verify_browser(data, scored):
    observed = json.loads((ROOT / 'shape_match_v2_browser.json').read_text())
    for view in observed['views']:
        stock = next(s for s in data['stocks'] if s['code'] == view['stock'])
        diags = scored['stocks'][stock['code']]['samples']
        def admitted(sample, threshold=.9):
            whole = sample['r'] is not None and sample['r'] >= threshold and sample['amplitude_pass']
            return whole if view['mode'] == 'v1' else passes(sample, diags[sample['date']], threshold)
        candidates = [s for s in stock['samples'] if s['r'] is not None and s['r'] >= .9 and s['amplitude_pass']]
        accepted = [s for s in candidates if admitted(s)]
        canvas = view['canvas']
        assert int(canvas['historicalPathCount']) == len(candidates) == len(view['rows'])
        assert int(canvas['acceptedPathCount']) == len(accepted)
        assert int(canvas['rejectedPathCount']) == len(candidates) - len(accepted)
        assert canvas['currentEndX'] == '29' and canvas['currentPointCount'] == '31' and canvas['axisEndX'] == '88'
        assert not view['overflow']
        assert view['selected'] == max(s['date'] for s in accepted or candidates)
        assert view['metrics'][0].split('\n')[-1] == f'{len(candidates)} / {len(accepted)}'
        assert {s['date'] for s in view['rows'] if s['accepted'] == 'true'} == {s['date'] for s in accepted}
        for group in view['groups']:
            recent = group['group'] == 'recent'
            cases = [s for s in candidates if (s['date'] >= '2026-04-08') == recent]
            valid = [s for s in cases if admitted(s)]
            outcomes = [s['forward_3_sessions']['outcome'] for s in valid if s['forward_3_sessions']['outcome']['complete']]
            cells = group['cells']
            assert cells[1].split('\n')[0] == f'{len(cases)} / {len(valid)} / {len(outcomes)}'
            assert cells[2] == (f"{sum(s['close_pct'] > 0 for s in outcomes)}/{len(outcomes)}" if outcomes else '—')
            for index, field, op in [(3, 'close_pct', statistics.mean), (4, 'close_pct', statistics.median),
                                    (5, 'high_pct', statistics.median), (6, 'low_pct', statistics.median)]:
                if outcomes:
                    assert abs(float(cells[index].rstrip('%')) - op(s[field] for s in outcomes)) < .0051
                else:
                    assert cells[index] == '—'
        complete = [s['future_24h'] for s in accepted if s['future_24h']]
        for label, field, op in [('终值均值', 'close_pct', statistics.mean), ('终值中位', 'close_pct', statistics.median)]:
            cell = next(s[1] for s in view['futureStats'] if s[0] == label)
            if complete:
                assert abs(float(cell.rstrip('%')) - op(s[field] for s in complete)) < .0051
            else:
                assert cell == '—'
        for row in view['sensitivity']:
            assert int(row[1]) == sum(admitted(s, float(row[0])) for s in stock['samples'])
        for row in view['rows']:
            assert row['color'] == ('#3178b9' if row['date'] >= '2026-04-08' else '#aeb4bc')
            assert '整体r=' in row['text'] and '末半段r=' in row['text'] and '后续3交易日' in row['text']
    assert observed['narrow']['width'] == 390 and not observed['narrow']['overflow']
    assert observed['narrow']['canvas']['matchMode'] == 'v2'
    assert any(s['date'] == '2026-10-07' and '待确认' in s['text'] for s in observed['partialFutureCheck']['unknownRows'])
    assert not observed['errors']
    return {'views_verified': len(observed['views']), 'v1_and_v2_statistics_verified': True,
            'all_candidates_visible': True, 'threshold_counts_verified': True,
            'partial_future_preserved': True, '390px_overflow': False, 'errors': observed['errors']}


def main():
    original = ROOT / 'results.json'
    data, scored = (json.loads((ROOT / name).read_text()) for name in ('results.json', 'shape_match_v2.json'))
    assert hashlib.sha256(original.read_bytes()).hexdigest() == scored['protocol']['source_results_sha256']
    hours = data['curve_extension']['prefix_x']
    indexes = [i for i, x in enumerate(hours) if x >= hours[-1] / 2]
    x = [hours[i] for i in indexes]
    counts, checked, selection_independent = {}, 0, 0
    for stock in data['stocks']:
        if not stock.get('current_curve'):
            continue
        a = stock['current_curve']
        ta = [a[i] for i in indexes]
        current_slope = slope(x, ta)
        accepted = []
        result = scored['stocks'][stock['code']]
        for sample in stock['samples']:
            diag = result['samples'][sample['date']]
            b = sample['curve']
            tb = [b[i] for i in indexes]
            full_r, tail_r, history_slope = corr(a, b), corr(ta, tb), slope(x, tb)
            close(diag['full_r_recomputed'], full_r)
            close(sample['r'], full_r)
            close(diag['tail_r'], tail_r)
            close(diag['current_tail_slope'], current_slope)
            close(diag['history_tail_slope'], history_slope)
            close(diag['current_tail_pct'], 100 * math.expm1((ta[-1] - ta[0]) / 100))
            close(diag['history_tail_pct'], 100 * math.expm1((tb[-1] - tb[0]) / 100))
            close(diag['coefficient'], min(full_r, tail_r) if full_r is not None and tail_r is not None else None)
            da, db = [y - z for z, y in zip(a, a[1:])], [y - z for z, y in zip(b, b[1:])]
            close(diag['hourly_change_r'], corr(da, db))
            assert diag['direction_agreement_count'] == sum(sign(u) == sign(v) for u, v in zip(da, db))
            assert diag['direction_comparison_count'] == 30
            same_direction = sign(current_slope) != 0 and sign(current_slope) == sign(history_slope)
            assert diag['tail_same_direction'] == same_direction
            for threshold in (.80, .85, .90, .95):
                expected = (sample['r'] is not None and sample['r'] >= threshold and sample['amplitude_pass']
                            and tail_r is not None and tail_r >= threshold and same_direction)
                assert passes(sample, diag, threshold) == expected
                # Removing all future labels cannot alter eligibility.
                features_only = {'r': sample['r'], 'amplitude_pass': sample['amplitude_pass']}
                assert passes(features_only, diag, threshold) == expected
                selection_independent += 1
            if passes(sample, diag):
                accepted.append(sample)
            checked += 1
        assert sorted([s['date'] for s in accepted], reverse=True) == result['accepted_dates']
        assert len(accepted) == result['accepted_n']
        for horizon in ('future_24h', 'future_rth'):
            outcomes = [s[horizon] for s in accepted if s.get(horizon)]
            expected_stats = result['stats'][horizon]
            assert expected_stats['n'] == len(outcomes)
            assert expected_stats['unknown'] == len(accepted) - len(outcomes)
            if outcomes:
                for field, label in [('close_pct', 'close'), ('high_pct', 'high'), ('low_pct', 'low')]:
                    values = [r[field] for r in outcomes]
                    close(expected_stats[label]['mean'], statistics.mean(values))
                    close(expected_stats[label]['median'], statistics.median(values))
            else:
                assert expected_stats['mean'] is None and expected_stats['median'] is None
        assert sum(c['n'] for c in result['clusters']) == len(accepted)
        counts[stock['code']] = {'candidates': result['whole_candidates_n'], 'accepted': len(accepted), 'dates': result['accepted_dates']}
    # The reported bug: a high whole-curve r must not admit an opposing tail.
    example = scored['stocks']['US.NRG']['samples']['2026-09-08']
    assert example['full_r_recomputed'] > .9 and example['tail_r'] < 0
    assert example['current_tail_pct'] < 0 < example['history_tail_pct']
    assert scored['stocks']['US.NRG']['accepted_n'] == 0
    result = {'version': scored['protocol']['version'], 'status': 'passed', 'inputs_verified': checked,
              'four_threshold_checks_without_future_fields': selection_independent,
              'source_results_sha256_unchanged': True, 'stocks': counts,
              'reported_NRG_example': example, 'predictive_validation': 'not performed'}
    if (ROOT / 'shape_match_v2_browser.json').exists():
        result['browser'] = verify_browser(data, scored)
    (ROOT / 'shape_match_v2_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'inputs_verified': checked, 'no_future_selection_checks': selection_independent,
                      'NRG_tail_r': example['tail_r'], 'NRG_current_tail_pct': example['current_tail_pct'],
                      'NRG_history_tail_pct': example['history_tail_pct'], 'status': 'passed'}))


if __name__ == '__main__':
    main()
