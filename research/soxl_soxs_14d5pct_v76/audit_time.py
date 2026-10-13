"""Determine timestamp semantics from separate regular/session and daily requests."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/soxl_soxs_14d5pct_v76'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = [ROOT/'.cache/soxl_soxs_14d5pct_v75/older_probe/SOXL_20230403_20230405_0.parquet',
               CACHE/'time_probe/regular.parquet', CACHE/'time_probe/daily.parquet', CACHE/'ibkr/SOXL.json']
    all_bars, regular, daily = [pd.read_parquet(p) for p in sources[:3]]
    ibkr = pd.DataFrame(json.loads(sources[3].read_text())['rows']).set_index('date')
    columns = ['open', 'high', 'low', 'close', 'volume']
    joined = all_bars.set_index('time_key').loc[regular.time_key]
    assert np.array_equal(joined[columns].to_numpy(), regular[columns].to_numpy())
    rows = []
    for day, bars in regular.groupby(regular.time_key.str[:10]):
        ends = pd.date_range(day+' 09:35', day+' 16:00', freq='5min').strftime('%Y-%m-%d %H:%M:%S')
        assert list(bars.time_key) == list(ends)
        aggregate = np.array([bars.iloc[0].open, bars.high.max(), bars.low.min(), bars.iloc[-1].close])
        daily_row = daily[daily.time_key.str.startswith(day)].iloc[0]
        np.testing.assert_allclose(aggregate, daily_row[columns[:4]].to_numpy(float), rtol=0, atol=1e-9)
        start_bars = all_bars[all_bars.time_key.between(day+' 09:30:00', day+' 15:55:00')]
        start_ohlc = np.array([start_bars.iloc[0].open, start_bars.high.max(), start_bars.low.min(), start_bars.iloc[-1].close])
        assert not np.allclose(start_ohlc, aggregate, rtol=0, atol=1e-9)
        external = ibkr.loc[day, columns[:4]].to_numpy(float)
        assert abs(external-aggregate).max() <= .011
        rows.append(dict(date=day, regular_bars=len(bars), end_grid_ohlc=aggregate.tolist(),
            incorrect_start_grid_ohlc=start_ohlc.tolist(), independent_ibkr_ohlc=external.tolist(),
            volume_regular=float(bars.volume.sum()), volume_daily=float(daily_row.volume),
            volume_ibkr=float(ibkr.loc[day,'volume'])))
    result = dict(passed=True, semantics='ET five-minute bar END', rows=rows,
        raw_lineage_values=len(regular)*5, independent_price_tolerance=.011,
        prior_v75_start_grid_claim='FALSIFIED: 09:30..15:55 also counts 78 but includes premarket and excludes final regular bar.',
        old_v2_end_conversion='Confirmed for sampled SOXL dates; not a universal cross-provider inference.',
        volume_status='Different coverage across endpoints/providers; no equality claim or correction.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources})
    (CACHE/'time_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__ == '__main__': main()
