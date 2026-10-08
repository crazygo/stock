"""Frozen paired feature ablations on ALL pre-enrolled stock cycles."""
import json,pickle,re
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.special import logit
import evaluate as ev
OUT=ev.OUT;FRAME=None
VARIANTS=['H_only','no_volume','no_minutes','no_peers']
def init():
 global FRAME
 ev.init();FRAME=ev.FRAME

def columns(cols,variant):
 if variant=='H_only':return [k for k in cols if k=='stock_id' or k.startswith('h_') or '_h_' in k]
 if variant=='no_volume':return [k for k in cols if not (re.search(r'volume|rvol|active|vol_hhi|signed_volume|bar_vwap|vol_price|burst|h_v_|previous_volume',k))]
 if variant=='no_minutes':return [k for k in cols if not (re.search(r'(^|_)w\d+|^slide|relative_w30',k))]
 return [k for k in cols if not k.startswith('peer_')]
def job(cell):
 s=cell['symbol'];start=cell['start'];name=cell['selected'];spec=next(x for x in ev.MODELS if x[0]==name);_,scope,window,kind=spec
 prior=FRAME[FRAME.day<start];dates=sorted(prior.day.unique())[-290:];train=prior[prior.day.isin(dates[:-50][-window:])&prior.symbol.isin([s] if scope=='solo' else ev.b.GROUPS.get(ev.b.GROUP_OF.get(s,''),[s]))].copy()
 if scope=='regime':
  env=prior[prior.symbol==s].sort_values('day').iloc[-1]
  for k in ['qqq_h_return_30','qqq_h_rv_30']:
   median=float(train[k].median());mad=float((train[k]-median).abs().median())
   if np.isfinite(env[k]) and mad>0:train=train[(train[k]-env[k]).abs()<=max(mad*2,.01 if 'return' in k else .003)]
 fit=prior[prior.symbol.eq(s)&prior.day.isin(dates[-50:-25])];target=FRAME[FRAME.symbol.eq(s)&FRAME.day.between(start,cell['end'])].copy()
 with (OUT/'models'/cell['artifact']).open('rb') as h:artifact=pickle.load(h)
 records=[]
 for variant in VARIANTS:
  cols=columns(artifact['features'],variant);model=clone(artifact['model']);model.fit(train[cols],train.y)
  raw=model.predict_proba(fit[cols])[:,1];cal=LogisticRegression(C=1,random_state=1004).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),fit.y)
  score=ev.transform(model.predict_proba(target[cols])[:,1],cal)
  records.extend(dict(symbol=s,day=d,cycle_start=start,variant=variant,score=float(p),threshold=cell['threshold']) for d,p in zip(target.day,score))
 return records

def main():
 init();report=json.loads((OUT/'results.json').read_text());cells=[c for c in report['cells'] if c['selected']];records=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
  for i,f in enumerate(as_completed([pool.submit(job,c) for c in cells])):
   records.extend(f.result())
   if i%30==0:print(json.dumps({'done':i+1,'total':len(cells)}),flush=True)
 ab=pd.DataFrame(records);ab.to_parquet(OUT/'ablation_predictions.parquet',index=False,compression='zstd',compression_level=7)
 full=pd.read_parquet(OUT/'predictions.parquet');result={};rng=np.random.default_rng(1004)
 for variant,g in ab.groupby('variant'):
  merged=full.merge(g,on=['symbol','day','cycle_start'],suffixes=('_full','_ab'),validate='one_to_one');assert len(merged)==len(full)
  merged['delta']=(merged.y-merged.score_full)**2-(merged.y-merged.score_ab)**2
  blocks=merged.groupby('cycle_start').delta.agg(['sum','count']);sums=blocks['sum'].to_numpy();counts=blocks['count'].to_numpy();indices=rng.integers(0,len(blocks),size=(2000,len(blocks)));samples=sums[indices].sum(1)/counts[indices].sum(1)
  result[variant]=dict(brier=float(brier_score_loss(merged.y,merged.score_ab)),delta_full_minus_ab=float(merged.delta.mean()),delta_ci=np.quantile(samples,[.025,.975]).tolist(),signal_metrics=ev.stats(merged,merged.score_ab,merged.threshold_ab))
 payload=dict(protocol_sha256=ev.b.sha(OUT/'ABLATION_PROTOCOL.md'),cells=len(cells),rows=len(full),full_brier=float(brier_score_loss(full.y,full.score)),variants=result,interpretation='negative_delta_means_full_model_better; selected_cohort_development_audit_not_independent')
 (OUT/'ablation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(payload,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
