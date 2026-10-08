"""Read the frozen local inputs; project daily OHLC for the price view. No acquisition."""
from pathlib import Path
from datetime import date, timedelta
import hashlib
import json
import math
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT / 'results.json').read_text())
    cutoff = date.fromisoformat(frozen['as_of'])
    mini_start = cutoff.replace(year=cutoff.year - 2).isoformat()
    us_path = ROOT / 'market_data/calendars/nasdaq_sessions_2026_v1.json'
    hk_path = OUT.parent / 'ai_trend_quadrant_v1/hk_calendar_2026.json'
    us = json.loads(us_path.read_text())
    hk = json.loads(hk_path.read_text())
    hk_days = []
    day = date.fromisoformat(hk['start'])
    while day.isoformat() <= hk['end']:
        if day.weekday() < 5 and day.isoformat() not in hk['holidays']:
            hk_days.append(day.isoformat())
        day += timedelta(days=1)
    calendars = {
        'US': {'id': us['calendar_id'], 'start': us['start_date'], 'end': us['end_date'],
               'days': [s['session_date'] for s in us['sessions']],
               'early_close': [s['session_date'] for s in us['sessions'] if s['is_early_close']],
               'source': us['source'], 'sha256': sha(us_path)},
        'HK': {'id': hk['calendar_id'], 'start': hk['start'], 'end': hk['end'],
               'days': hk_days, 'early_close': [], 'session_lengths': 'not_inferred_from_daily_bars',
               'source': hk['sources'], 'sha256': sha(hk_path)}
    }
    sources = {p['code']: p for p in frozen['provenance']}
    records = []
    for r in frozen['records']:
        market = r['code'].split('.')[0]
        projected = {'code': r['code'], 'market': market, 'currency': {'US': 'USD', 'HK': 'HKD', 'JP': 'JPY'}[market],
                     'listing_date': r.get('listing_date'), 'bars': [], 'invalid_days': [],
                     'status': 'unavailable', 'source': r['coverage'].get('source'),
                     'price_basis': r['coverage'].get('price_basis'), 'error': r['coverage'].get('fetch_error')}
        source = sources.get(r['code'])
        if source:
            path = ROOT / source['path']
            digest = sha(path)
            if digest != source['sha256']:
                raise ValueError(f"Frozen input changed: {r['code']}; use a new data revision")
            frame = pd.read_parquet(path).sort_values('day')
            if frame['day'].duplicated().any():
                raise ValueError(f"Duplicate daily dates: {r['code']}")
            for row in frame.to_dict('records'):
                day = str(row['day'])[:10]
                if day < mini_start or day > cutoff.isoformat() or (r.get('listing_date') and day < r['listing_date']):
                    continue
                o, h, l, c = [float(row[k]) for k in ['open', 'high', 'low', 'close']]
                if not all(math.isfinite(v) and v > 0 for v in [o, h, l, c]) or h < max(o, c, l) or l > min(o, c, h):
                    projected['invalid_days'].append(day)
                    continue
                # Full precision from the captured QFQ series; no independent weekly adjustment.
                projected['bars'].append([day, o, h, l, c])
            projected.update(status='available' if projected['bars'] else 'empty', path=source['path'], sha256=digest,
                             received_at=r['coverage'].get('received_at'))
        projected['first_day'] = projected['bars'][0][0] if projected['bars'] else None
        projected['last_day'] = projected['bars'][-1][0] if projected['bars'] else None
        cal = calendars.get(market)
        first = max(mini_start, r.get('listing_date') or mini_start)
        have = {b[0] for b in projected['bars']}
        projected['missing_sessions'] = [d for d in cal['days'] if first <= d <= cutoff.isoformat() and d not in have] if cal else []
        projected['calendar_unverified_bars'] = sum(1 for b in projected['bars'] if not cal or not cal['start'] <= b[0] <= cal['end'])
        records.append(projected)
    data = {'version': 'daily_weekly_price_projection_v1', 'source_run_id': frozen['run_id'],
            'source_results_sha256': sha(OUT / 'results.json'), 'as_of': cutoff.isoformat(),
            'mini_start': mini_start, 'bar_fields': ['day', 'open', 'high', 'low', 'close'],
            'calendars': calendars, 'records': records,
            'summary': {'securities': len(records), 'available': sum(bool(r['bars']) for r in records),
                        'daily_bars': sum(len(r['bars']) for r in records),
                        'invalid_bars': sum(len(r['invalid_days']) for r in records)}}
    encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (OUT / 'prices.json').write_text(encoded + '\n')
    print(json.dumps(data['summary']))


if __name__ == '__main__':
    main()
