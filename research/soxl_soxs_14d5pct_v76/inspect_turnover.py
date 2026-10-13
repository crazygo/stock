"""Source-field feasibility for a future activity feature, without fitting a model."""
from pathlib import Path
import datetime, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v76'


def main():
    symbols=json.loads((ROOT/'research/soxl_soxs_14d5pct_v76/DATA_PROTOCOL.json').read_text())['symbols']
    rows=[]
    for s in symbols:
        old='soxl_soxs_14d5pct_v2' if s in ['SOXL','SOXS','SOXX','QQQ'] else 'soxl_soxs_14d5pct_v70'
        for scope,p in [('earlier',CACHE/'minutes'/s/'bars'/f'{s}.parquet'),('parent',ROOT/'.cache'/old/'data/bars'/f'{s}.parquet')]:
            f=pd.read_parquet(p)
            if scope=='parent':f=f[f.session_date<='2026-04-30']
            f=f[f.volume>0];vwap=f.turnover/f.volume
            delta=((f.high+f.low+f.close)/3*f.volume/f.turnover-1)
            finite=np.isfinite(vwap)&np.isfinite(delta)&(f.turnover>0)
            outside=(vwap<f.low-.011)|(vwap>f.high+.011)
            rows.append(dict(symbol=s,scope=scope,positive_volume_bars=len(f),finite_positive_turnover=bool(finite.all()),
                vwap_outside_reported_bar_by_more_than_0011=int(outside.sum()),
                typical_price_amount_relative_error_median=float(delta.median()),
                typical_price_amount_abs_relative_error_p99=float(delta.abs().quantile(.99)),
                typical_price_amount_abs_relative_error_max=float(delta.abs().max())))
    result=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,
        purpose='Check direct turnover field as a future alternative to HLC-average times volume. No model comparison or return claim.',
        anomaly='SPY 2023-10-30 10:30 ET five-minute low 408.91 appears in 10:28 one-minute bar and its aggregation; daily Futu/IBKR low is 412.22. Minute turnover and volume aggregate exactly across resolutions. Print inclusion cause remains unknown.',
        future_requirement='Register direct-turnover versus current typical-price activity controls before changing features; preserve source anomaly and do not treat source-field availability as model effectiveness.')
    (CACHE/'turnover_feasibility.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
