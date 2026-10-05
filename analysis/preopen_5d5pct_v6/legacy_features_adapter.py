"""Isolated subprocess imports v5 causal primitives, writes only a supplied v6 path."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
OLD=Path(__file__).resolve().parents[1]/'preopen_ranked_policy_v5'
sys.path.insert(0,str(OLD))
from data import prior_context,intraday,peers
from common import SESSIONS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();parts=[]
    cut=570+SESSIONS[a.day]['duration_minutes']-30
    for p in sorted((OLD/'raw').glob('*.parquet')):
        f=pd.read_parquet(p);f=f[(f.day<=a.day)&(f.day>=str((pd.Timestamp(a.day)-pd.Timedelta(days=130)).date()))]
        if f.empty:continue
        ctx,_=prior_context(f,p.stem);n=intraday(f,ctx,p.stem,labels=False,first=a.day,last=a.day,only_minutes=range(cut-60,cut+1,5))
        if len(n):parts.append(n)
    frame=peers(pd.concat(parts,ignore_index=True));frame.to_parquet(a.output,index=False,compression='zstd',compression_level=7)
    print(json.dumps(dict(day=a.day,minute=cut,rows=len(frame))),flush=True)
if __name__=='__main__':main()
