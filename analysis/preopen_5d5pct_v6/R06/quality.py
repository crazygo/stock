"""Non-price-source numeric and record-day integrity; no outcome filtering."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import OUT,DATES,write,now,sha
P=OUT/'R06'

def main():
    rows=[]
    for family in ['capital','options','volatility','short_volume']:
        for p in sorted((P/'cache/history'/family).glob('*.parquet')):
            if '.first_attempt.' in p.name:continue
            f=pd.read_parquet(p);bad={};nums=f.select_dtypes('number');bad['infinite_fields']=int(np.isinf(nums.to_numpy(dtype=float)).sum());bad['duplicate_days']=int(f.source_day.duplicated().sum());bad['outside_official_calendar']=int((~f.source_day.isin(DATES)).sum())
            def mismatch(a,b,tol):
                valid=a.notna()&b.notna();return dict(checked=int(valid.sum()),violations=int((abs(a[valid]-b[valid])>tol).sum()))
            consistency={}
            if family=='capital':
                consistency['net_flow_sum']=mismatch(f.in_flow,f[['super_in_flow','big_in_flow','mid_in_flow','sml_in_flow']].sum(axis=1,min_count=4),.02)
                consistency['main_flow_sum']=mismatch(f.main_in_flow,f.super_in_flow+f.big_in_flow,.02)
            elif family=='options':
                consistency['call_put_volume_sum']=mismatch(f.option_volume,f.call_volume+f.put_volume,1)
                consistency['call_put_OI_sum']=mismatch(f.option_open_interest,f.call_open_interest+f.put_open_interest,1)
                bad['negative_counts']=int((f[['option_volume','call_volume','put_volume','option_open_interest','call_open_interest','put_open_interest']]<0).sum().sum())
            elif family=='volatility':bad['negative_IV_HV']=int((f[['iv','hv']]<0).sum().sum())
            else:
                consistency['short_qty_sum']=mismatch(f.total_shares_short,f.nasdaq_shares_short+f.nyse_shares_short,1)
                consistency['short_percent_ratio']=mismatch(f.short_percent,f.total_shares_short/f.volume.where(f.volume>0)*100,.002)
                bad['outside_percent_range']=int(((f.short_percent<0)|(f.short_percent>100)).sum())
            rows.append(dict(symbol=p.stem,family=family,rows=len(f),missing_by_field={c:int(f[c].isna().sum()) for c in nums},invalid=bad,consistency=consistency,sha256=sha(p)))
    write(P/'source_quality.json',dict(at=now(),status='completed',files=len(rows),records=rows,note='Invalid feature values are missing in feature construction; no signals/outcomes are dropped. Numeric checks do not certify first-publication or revision vintages.'))
    print(json.dumps(dict(files=len(rows),invalid_total=sum(sum(r['invalid'].values()) for r in rows),consistency_violations=sum(x['violations'] for r in rows for x in r['consistency'].values()))),flush=True)

if __name__=='__main__':main()
