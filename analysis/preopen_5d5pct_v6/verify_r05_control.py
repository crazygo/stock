"""Confirm a repeated control is equivalent, not new market evidence."""
import json
import numpy as np
import pandas as pd
from common import OUT,write,now

def main():
    checks=[];keys=['symbol','day','minute']
    for month in [f'2026-{m:02d}' for m in range(5,10)]:
        for variant in ['candidate_only','top_one']:
            cols=keys+['score','baseline','y']
            old=pd.read_parquet(OUT/'R03/cache/runs'/f'H3_{month}'/(variant+'.parquet'),columns=cols).sort_values(keys).reset_index(drop=True)
            new=pd.read_parquet(OUT/'R05/cache/runs'/f'D0_{month}'/(variant+'.parquet'),columns=cols).sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(old,new,check_dtype=False,atol=1e-12,rtol=0)
            checks.append(dict(month=month,variant=variant,rows=len(old),maximum_score_error=float(np.max(abs(old.score-new.score))),passed=True))
    write(OUT/'R05/control_verification.json',dict(at=now(),status='passed',checks=checks,note='D0 repeats H3; equivalent retraining is not independent evidence.'))
    print(json.dumps(dict(status='passed',checks=len(checks))))
if __name__=='__main__':main()
