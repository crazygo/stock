"""Isolated v5 primitive adapter for longer, quality-filtered history.

All outputs live in v6. Provider/legacy raw prices are never changed.
"""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'preopen_ranked_policy_v5';sys.path.insert(0,str(OLD))
from data import prior_context,intraday
from horizon import label_rows,action_days,valid_bars

def make(job):
 symbol,raw_dir,output,first,last=job;dest=Path(output)/f'{symbol}.parquet'
 if dest.exists():return dict(symbol=symbol,status='existing',rows=len(pd.read_parquet(dest,columns=['day'])))
 started=time.monotonic();path=Path(raw_dir)/f'{symbol}.parquet';raw=pd.read_parquet(path).sort_values('start').drop_duplicates('start',keep='last');raw=raw[valid_bars(raw)].copy()
 context,_=prior_context(raw,symbol);features=intraday(raw,context,symbol,labels=False,first=first,last=last)
 if features.empty:return dict(symbol=symbol,status='no_causal_feature_rows',rows=0)
 actions=action_days(symbol);p=BASE/'cache/corporate_actions'/f'{symbol}.parquet'
 if p.exists():actions.update(pd.read_parquet(p).ex_div_date.astype(str).str[:10])
 labels=label_rows(raw,features,'5d5pct',actions,cutoff=last)
 for c in labels:features[c]=labels[c].to_numpy()
 # Entry and target retain float64. Numeric feature storage is compact.
 for c in features.select_dtypes('float'):
  if c not in ['entry','target']:features[c]=features[c].astype('float32')
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.writing.parquet');features.to_parquet(tmp,index=False,compression='zstd',compression_level=7);tmp.replace(dest)
 return dict(symbol=symbol,status='prepared',rows=len(features),seconds=time.monotonic()-started,first=features.day.min(),last=features.day.max(),known=int(features.y.notna().sum()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--symbols',nargs='+');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--first',default='2024-10-04');ap.add_argument('--last',default='2026-09-30');a=ap.parse_args()
 symbols=a.symbols or sorted(p.stem for p in Path(a.raw_dir).glob('*.parquet'));jobs=[(s,a.raw_dir,a.output,a.first,a.last) for s in symbols];reports=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(make,j):j[0]for j in jobs}
  for f in as_completed(futures):
   try:r=f.result()
   except Exception as e:r=dict(symbol=futures[f],status='failed',error=repr(e))
   reports.append(r);print(json.dumps(r),flush=True)
 Path(a.output).mkdir(parents=True,exist_ok=True);(Path(a.output)/'preparation.json').write_text(json.dumps(dict(records=reports,complete=all(r['status']!='failed'for r in reports)),indent=2)+'\n')
 if any(r['status']=='failed'for r in reports):raise SystemExit(1)
if __name__=='__main__':main()
