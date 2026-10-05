"""T0 repeated training must match prior H3, with labels and cohorts intact."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import OUT,write,now

def main():
    checks=[];keys=['symbol','day','minute']
    for month in [f'2026-{m:02d}' for m in range(5,10)]:
        for variant in ['candidate_only','top_one']:
            columns=keys+['score','baseline','y'];frames=[]
            for rid,arm in [('R03','H3'),('R06','T0')]:frames.append(pd.read_parquet(OUT/rid/'cache/runs'/f'{arm}_{month}'/(variant+'.parquet'),columns=columns).sort_values(keys).reset_index(drop=True))
            pd.testing.assert_frame_equal(*frames,check_dtype=False,atol=1e-12,rtol=0)
            checks.append(dict(month=month,variant=variant,rows=len(frames[0]),maximum_score_error=float(np.max(abs(frames[0].score-frames[1].score)))))
    d=pd.read_parquet(OUT/'R06/cache/daily_features.parquet');bounds=[]
    for family in ['c','o','v','s']:
        for lag in [2,5]:
            prefix=f'f{lag}_{family}_';valid=d[d[prefix+'source_day'].notna()];assert (valid[prefix+'source_day']<valid.day).all();assert (valid[prefix+'available_day']<=valid.day).all()
            idx={v:i for i,v in enumerate(__import__('common').DATES)};diff=valid.day.map(idx)-valid[prefix+'source_day'].map(idx);assert (diff>=lag).all();bounds.append(dict(family=family,lag=lag,rows=len(valid),minimum_actual_lag=int(diff.min()) if len(valid) else None))
    write(OUT/'R06/control_verification.json',dict(at=now(),status='passed',checks=checks,input_lag_bounds=bounds,note='Repeated control is not independent evidence.'))
    print(json.dumps(dict(status='passed',checks=len(checks))),flush=True)

if __name__=='__main__':main()
