"""Independently recompute selected curves, future extrema, and within-cluster correlations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
res=json.loads((OUT/'results.json').read_text());acq=json.loads((OUT/'acquisition.json').read_text());checks=[]
for s in res['stocks']:
 if s['status']!='ready':continue
 d=next(d for d in acq['datasets'] if d['code']==s['code']);df=pd.read_parquet(ROOT/d['path']);bars=df.set_index('time_key').to_dict('index')
 selected=[x for x in s['samples'] if x['r'] is not None and x['r']>=res['protocol']['threshold_N'] and x['amplitude_pass']]
 assert len(selected)==s['matched_n']
 for x in selected:
  arr=np.r_[0,100*np.log(np.array([bars[k]['close'] for k in x['feature_keys']])/bars[x['feature_keys'][0]]['open'])]
  assert np.allclose(arr,x['curve'],atol=1e-6)
  assert abs(np.corrcoef(arr,s['current_curve'])[0,1]-x['r'])<1e-6
  f=x['future_24h'];b=x['reference_price']
  if f:
   actual=df[(df.time_key>x['input_end'])&(df.time_key<=f['terminal_at_et'])]
   assert abs(100*(actual.close.iloc[-1]/b-1)-f['close_pct'])<1e-5
   assert abs(100*(actual.high.max()/b-1)-f['high_pct'])<1e-5
   assert abs(100*(actual.low.min()/b-1)-f['low_pct'])<1e-5
 for cluster in s['clusters']:
  curves=[x['curve'] for x in selected if x['date'] in cluster['dates']]
  if len(curves)>1:assert np.min(np.corrcoef(curves))>=.9-1e-6
 checks.append({'code':s['code'],'independent_curve_and_outcome_recomputed':len(selected),'cluster_pairwise_r_verified':True})
(OUT/'independent_verification.json').write_text(json.dumps({'passed':True,'cutoff':res['protocol']['cutoff_et'],'checks':checks},ensure_ascii=False,indent=2))
print('passed',len(checks),'stocks',sum(x['independent_curve_and_outcome_recomputed'] for x in checks),'matched samples')
