"""Independent bookkeeping and raw-price checks of a frozen inventory."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def verify(run):
    d = json.loads((run / 'evidence.json').read_text())
    manifest = json.loads((run / 'manifest_at_launch.json').read_text())
    for path, expected in manifest['sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for name, expected in json.loads((run / 'output_hashes.json').read_text()).items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected, name

    rows = {(r['scope'], r['T'], r['P'], r['symbol']): r for r in d['counts']}
    groups = 0
    for key, stocks in d['grids'].items():
        scope, width, amplitude = key.split('|')
        for symbol, ids in stocks.items():
            assert len(ids) == len(set(ids))
            per_day = defaultdict(list)
            for eid in ids:
                e = d['events'][eid]
                assert e['symbol'] == symbol
                assert e['start'] < d['config']['scopes'][scope]
                assert e['duration'] <= int(width)
                assert e['gain'] >= float(amplitude) - 1e-12
                per_day[e['date']].append(e)
            for events in per_day.values():
                events.sort(key=lambda e: e['start'])
                assert all(a['end'] <= b['start'] for a, b in zip(events, events[1:]))
            chosen = [d['events'][eid] for eid in ids if d['events'][eid]['duration'] >= 20]
            row = rows[(scope, int(width), float(amplitude), symbol)]
            assert row['count'] == len(chosen)
            assert row['opportunity_days'] == len({e['date'] for e in chosen})
            assert row['after_two8'] == sum(e['remaining'] is not None and e['remaining'] >= .08 - 1e-12 for e in chosen)
            assert row['all_durations_count'] == len(ids)
            groups += 1

    # Re-read independently from raw bars: all 15 headline events plus 24 seeded samples.
    ids = random.Random(20260925).sample(sorted(d['events']), 24)
    ids += [eid for stock in d['grids']['first30|30|0.1'].values() for eid in stock if d['events'][eid]['duration'] >= 20]
    raw = {}
    for eid in dict.fromkeys(ids):
        e = d['events'][eid]
        if e['symbol'] not in raw:
            f = pd.read_parquet(ROOT / d['config']['source_dir'] / e['symbol'] / '2026.parquet')
            f['start_at'] = pd.to_datetime(f.start_at, utc=True)
            raw[e['symbol']] = f.set_index('start_at')
        opening = pd.Timestamp(e['date'] + ' 09:30', tz='America/New_York').tz_convert('UTC')
        times = pd.date_range(opening + pd.Timedelta(minutes=e['start'] * 5), periods=e['end'] - e['start'], freq='5min')
        part = raw[e['symbol']].loc[times]
        base = float(part.open.iloc[0])
        assert np.isclose(base, e['base'])
        assert np.isclose(part.high.max() / base - 1, e['gain'])
        second = opening + pd.Timedelta(minutes=(e['start']+1)*5)
        assert np.isclose(raw[e['symbol']].loc[second, 'close'] / base - 1, e['r10'])
        if len(part) > 3:
            assert np.isclose(part.open.iloc[3] / base - 1, e['entry_gain'])
            assert np.isclose(part.high.iloc[3:].max() / part.open.iloc[3] - 1, e['remaining'])
        else:
            assert e['remaining'] is None

    spec = importlib.util.spec_from_file_location('inventory', HERE / 'inventory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prices = np.array([[100+i, 101+i if i<5 else 110, 99+i, 100.5+i] for i in range(6)], dtype=float)
    candidates, starts = module.candidates_for_width(prices, np.ones(6, dtype=bool), 6)
    assert starts == [0] and len(candidates) == 1
    assert np.isclose(candidates[0]['gain'], .10)
    assert np.isclose(candidates[0]['remaining'], 110/103-1)
    valid = np.ones(6, dtype=bool); valid[2] = False
    assert module.candidates_for_width(prices, valid, 6) == ([], [])
    intervals = [{'start':0,'end':6,'duration':30,'gain':.1}, {'start':1,'end':5,'duration':20,'gain':.08}, {'start':6,'end':10,'duration':20,'gain':.09}]
    assert [e['start'] for e in module.representatives(intervals)] == [0,6]
    assert np.isclose(1.1 / 1.02 - 1, .0784313725490196)
    assert np.isclose(1.08 * 1.02 - 1, .1016)

    sensitivity = []
    for width in d['config']['widths_minutes']:
        events = [d['events'][eid] for stock in d['grids'][f'first30|{width}|0.1'].values() for eid in stock if d['events'][eid]['duration'] >= 20]
        row = {'T':width, 'P':.1, 'count':len(events)}
        for offset in [2,3,4]:
            n = 0
            for e in events:
                bars = d['days'][e['symbol']+'|'+e['date']]
                j = e['start'] + offset
                n += j < e['end'] and max(b[1] for b in bars[j:e['end']]) / bars[j][0] - 1 >= .08 - 1e-12
            row[f'entry_{offset*5}m_remaining8'] = n
        sensitivity.append(row)
    result = {'verified_at':datetime.now(timezone.utc).isoformat(), 'source_and_frozen_output_hashes':'pass', 'grid_stock_groups_checked':groups, 'independent_raw_events_checked':len(set(ids)), 'synthetic_checks':'pass', 'entry_sensitivity':sensitivity, 'limitations':'Descriptive only; same provider, not independent market verification or executable fills.'}
    (run / 'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    verify(p.parse_args().run.resolve())
