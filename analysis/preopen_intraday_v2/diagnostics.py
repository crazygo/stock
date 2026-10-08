"""Finite geometry diagnostics; no parameter or threshold search."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent

def metrics(g):
    return {'n':len(g),'symbols':g.symbol.nunique(),'days':g.day.nunique(),'touch3':float((g.mfe>=.03).mean()),'down3':float((g.mae<=-.03).mean()),'close_up':float((g.close_ret>0).mean()),'mfe_median':float(g.mfe.median()),'mae_median':float(g.mae.median())}

def matched_difference(g,flag):
    g=g.copy();g['flag']=flag.astype(bool);keys=['symbol','hbin','pbin'];g['cell']=g[keys].astype(str).agg('|'.join,axis=1)
    counts=g.groupby(['cell','flag']).size().unstack(fill_value=0)
    if True not in counts or False not in counts:return {'n':0,'difference':None,'ci':[None,None]}
    allowed=counts[(counts[True]>=5)&(counts[False]>=5)].index;g=g[g.cell.isin(allowed)]
    if g.empty:return {'n':0,'difference':None,'ci':[None,None]}
    g['y']=(g.mfe>=.03).astype(float)
    def diff(w):
        a=w.groupby(['cell','flag']).y.agg(['sum','count']).unstack('flag').dropna()
        if not len(a):return np.nan
        n=np.minimum(a['count'][False],a['count'][True]);d=a['sum'][True]/a['count'][True]-a['sum'][False]/a['count'][False]
        return float((d*n).sum()/n.sum())
    day=sorted(g.day.unique());blocks=[day[i:i+5] for i in range(0,len(day),5)];rng=np.random.default_rng(1003);samples=[]
    for _ in range(1000):
        weights=pd.Series([d for i in rng.integers(0,len(blocks),len(blocks)) for d in blocks[i]]).value_counts()
        w=g.loc[g.index.repeat(g.day.map(weights).fillna(0).astype(int))];value=diff(w)
        if np.isfinite(value):samples.append(value)
    return {'n':len(g),'cells':len(allowed),'difference':diff(g),'ci':np.quantile(samples,[.025,.975]).tolist(),'positive':metrics(g[g.flag]),'negative':metrics(g[~g.flag])}

def main():
    f=pd.read_parquet(OUT/'panel.parquet');f=f[f.qqq_pre_gap.notna()&f.qqq_night_ret.notna()].copy();f['ratio']=f.pre_range/f.h_range.clip(lower=.001)
    train=f[f.day<='2025-09-30'];test=f[f.day>='2026-01-01'].copy()
    thresholds=train.groupby('symbol').ratio.quantile([.25,.75]).unstack()
    test['lo']=test.symbol.map(thresholds[.25]);test['hi']=test.symbol.map(thresholds[.75]);test=test.dropna(subset=['lo','hi'])
    for c,name in [('h_range','hbin'),('pre_range','pbin')]:
        cuts=train.groupby('symbol')[c].quantile([.2,.4,.6,.8]).unstack()
        test[name]=[int((r[c]>cuts.loc[r['symbol']]).sum()) for _,r in test.iterrows()]
    high=test[test.ratio>=test.hi];low=test[test.ratio<=test.lo]
    result={'protocol_sha256':hashlib.sha256((OUT/'DIAGNOSTIC_PROTOCOL.md').read_bytes()).hexdigest(),'panel_sha256':hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(),'status':'exposed_history_descriptive','range_low':metrics(low),'range_high':metrics(high),'pre_direction_high_range':matched_difference(high,high.pre_ret>0),'late_repair':matched_difference(test[test.pre_early<0],test[test.pre_early<0].pre_late>0),'relative_gap':matched_difference(test,test.relative_pre_gap>0)}
    (OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
