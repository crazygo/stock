"""Independent NumPy recomputation against frozen raw OHLC, without renderer imports."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(view):
    run = json.loads((view / 'run.json').read_text())
    source = (view / run['source_path']).resolve()
    data = json.loads((view / 'industry_table_data.json').read_text())
    for name, expected in [('index.html', run['source_index_sha256']),
                           ('manifest.json', run['source_manifest_sha256']),
                           ('results/industry_analysis.json', run['source_industry_analysis_sha256'])]:
        assert digest(source / name) == expected, name
    for ref in data['daily_path_input_refs'].values():
        assert digest(source / ref['path']) == ref['sha256'], ref['path']
    prices = json.loads((source / data['daily_path_input_refs']['prices']['path']).read_text())
    raw = {}
    for record in prices['records']:
        values = {}
        for bar in record['bars']:
            o, h, l, c = bar[1:5]
            if np.all(np.isfinite([o, h, l, c])) and min(o, h, l, c) > 0 and l <= min(o, c) + 1e-8 and h >= max(o, c) - 1e-8:
                values[bar[0]] = c
        raw[record['code']] = values
    official = {m: [d for d in prices['calendars'][m]['days'] if d <= cutoff][-61:]
                for m, cutoff in data['market_cutoff'].items()}
    rows = list(data['security_rows'].values()) + data['groups']
    checked, gaps, unavailable = 0, 0, []
    for row in rows:
        calendar = official[row['market']]
        if row['n'] is None:
            values = raw.get(row['id'], {})
            available = [d for d in calendar if d in values]
            base = values[available[0]] if available else None
            closes = {d: values[d] / base * 100 for d in calendar if d in values} if base else {}
        else:
            members = [code for code in row['children'] if len(calendar) == 61 and all(d in raw.get(code, {}) for d in calendar)]
            assert members == row['chart']['used_ids'], row['id'] + ' basket members'
            closes = {d: float(np.mean([raw[code][d] / raw[code][calendar[0]] for code in members])) * 100
                      for d in calendar} if members else {}
        overlay = row['chart']['overlay']
        center_days = calendar[-31:]
        complete = len(center_days) == 31 and all(d in closes for d in center_days)
        assert (overlay['status'] == 'complete') == complete, row['id'] + ' availability'
        assert overlay['center_dates'] == center_days, row['id'] + ' window'
        if complete:
            center_closes = np.array([closes[d] for d in center_days])
            geometric = float(np.exp(np.mean(np.log(center_closes))))
            mean_return = float(np.mean(center_closes[1:] / center_closes[:-1] - 1))
            np.testing.assert_allclose([overlay['price_reference'], overlay['daily_return_reference']], [geometric, mean_return], rtol=1e-11, atol=1e-12)
        else:
            unavailable.append(row['id'])
            assert overlay['price_reference'] is None and overlay['daily_return_reference'] is None
        for i, point in enumerate(overlay['points'], 1):
            day, close, price_y, daily, daily_y = point
            assert day == calendar[i] and day <= data['market_cutoff'][row['market']]
            expected_close = closes.get(day)
            previous = closes.get(calendar[i-1])
            expected_return = None if expected_close is None or previous is None else expected_close / previous - 1
            expected_price = None if not complete or expected_close is None else float(100 * np.log(expected_close / geometric))
            expected_daily = None if not complete or expected_return is None else float(100 * (np.log1p(expected_return) - np.log1p(mean_return)))
            for actual, expected in zip([close, price_y, daily, daily_y], [expected_close, expected_price, expected_return, expected_daily]):
                if expected is None:
                    assert actual is None, row['id'] + ' missing filled'
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            checked += 1
            gaps += expected_close is None
    result = {'status': 'pass', 'method': 'independent NumPy recomputation from frozen daily OHLC',
              'series': len(rows), 'points_checked': checked, 'missing_price_points': gaps,
              'center_unavailable': unavailable, 'source_hashes_preserved': True,
              'price_center_closes': 31, 'daily_center_changes': 30,
              'input_cutoffs': data['market_cutoff'], 'no_future_points': True}
    destination = view / 'validation/overlay_math.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]).resolve()), ensure_ascii=False))
