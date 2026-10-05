"""Report actual per-block usable fields, without filtering the training cohort."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import OUT,DATES,write,now
from research import MONTHS

def main():
    dest=OUT/'R06';coverage=json.loads((dest/'coverage.json').read_text());d=pd.read_parquet(dest/'cache/daily_features.parquet');labels=pd.read_parquet(dest/'cache/panel.parquet',columns=['symbol','day','y']);labels.symbol=labels.symbol.astype(str);labels.day=labels.day.astype(str);labels=labels[labels.y.notna()].drop_duplicates(['symbol','day']);rows=[]
    for month in MONTHS:
        cursor=next(i for i,s in enumerate(DATES) if s>=month+'-01');blocks=[]
        for n in [20,20,20,200]:cursor-=10;start=max(0,cursor-n);blocks.append(DATES[start:cursor]);cursor=start
        selection,top,candidate,training=blocks;training=[s for s in training if s>=coverage['actual_features'][0]]
        registered=labels[labels.day.isin(training)].groupby('symbol').day.nunique();registered=set(registered[registered>=60].index)
        for name,days in [('training',training),('candidate_calibration',candidate),('top_calibration',top),('selection',selection),('evaluation',[s for s in DATES if s.startswith(month)])]:
            f=d[d.day.isin(days)&d.symbol.isin(registered)]
            family=[]
            for k,cols in coverage['family_features'].items():family.append(dict(family=k,stock_days=len(f),any_feature_fraction=float(f[cols].notna().any(axis=1).mean()),all_features_fraction=float(f[cols].notna().all(axis=1).mean()),field_nonmissing={c:float(f[c].notna().mean()) for c in cols}))
            rows.append(dict(month=month,block=name,days=[days[0],days[-1]],registered=len(registered),coverage=family))
    write(dest/'training_coverage.json',dict(at=now(),weighting='each registered stock-day equal; unavailable values retained; no coverage filtering',blocks=rows))
    print(json.dumps(dict(status='completed',blocks=len(rows))),flush=True)

if __name__=='__main__':main()
