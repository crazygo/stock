"""Registered phase-specific calibration; base models and old results are immutable."""
from __future__ import annotations
import argparse,json,pickle
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from common import *
from train import weights,baseline
from policy import replay,gate
from store import persist
FRAME=None
def init():
 global FRAME
 FRAME=pd.read_parquet(OUT/'panel.parquet');FRAME['day']=FRAME.day.astype(str);FRAME['symbol']=FRAME.symbol.astype(str)
def phase_name(minute):return 'pre' if minute<=570 else 'regular'
def predict(a,f):
 raw=a['model'].predict_proba(f[a['features']])[:,1];p=np.full(len(f),np.nan)
 for phase,cal in a['phase_calibrators'].items():
  mask=(f.minute<=570).to_numpy() if phase=='pre' else (f.minute>570).to_numpy()
  if mask.any():p[mask]=cal.predict_proba(logit(np.clip(raw[mask],1e-5,1-1e-5)).reshape(-1,1))[:,1]
 return p
def apply_threshold(f,thresholds):
 p=f.copy();p['score_original']=p.score;p['threshold']=pd.to_numeric(p.minute.map(lambda m:thresholds[phase_name(m)]),errors='coerce');p['score']=np.where(p.threshold.notna()&(p.score>=p.threshold),p.score,-1.);return p
def solve(job):
 route,month=job;original=OUT/'models'/f'{route}_{month}.pkl';a=pickle.loads(original.read_bytes());meta=json.loads((OUT/f'{route}_{month}_meta.json').read_text());prior=FRAME[(FRAME.day<month+'-01')&FRAME.symbol.isin(a['registered'])];fit=prior[(prior.day>=meta['calibration_start'])&(prior.day<=meta['calibration_end'])&prior.y.notna()];selection=prior[(prior.day>=meta['selection_start'])&(prior.day<=meta['selection_end'])].copy();selection['baseline']=baseline(prior[prior.day<meta['selection_start']],selection)
 a['phase_calibrators']={};a['phase_thresholds']={};details={}
 for phase in ['pre','regular']:
  f=fit[fit.minute<=570] if phase=='pre' else fit[fit.minute>570];s=selection[selection.minute<=570].copy() if phase=='pre' else selection[selection.minute>570].copy();raw=a['model'].predict_proba(f[a['features']])[:,1];cal=LogisticRegression(C=1,max_iter=1000,random_state=1004).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),f.y,sample_weight=weights(f));a['phase_calibrators'][phase]=cal;s['score']=cal.predict_proba(logit(np.clip(a['model'].predict_proba(s[a['features']])[:,1],1e-5,1-1e-5)).reshape(-1,1))[:,1];choices=[]
  for t in THRESHOLDS:
   m=replay(s,t)['metrics'];choices.append(dict(threshold=t,metrics=m,passed=gate(m,n=8,days=6)))
  valid=[c for c in choices if c['passed']];best=max(valid,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'])) if valid else None;a['phase_thresholds'][phase]=best['threshold'] if best else None;details[phase]=dict(threshold=a['phase_thresholds'][phase],choices=choices,brier=float(brier_score_loss(s.y,s.score,sample_weight=weights(s))))
 path=OUT/'models'/f'phase_{route}_{month}.pkl';path.write_bytes(pickle.dumps(a));meta.update(artifact=path.name,artifact_sha256=sha(path),base_artifact_sha256=sha(original),threshold=0.,phase_thresholds=a['phase_thresholds'],phase_details=details,prior_gate_pass=any(v is not None for v in a['phase_thresholds'].values()),phase_protocol_sha256=sha(OUT/'PHASE_PROTOCOL.md'))
 if month in ['2026-08','2026-09']:
  f=FRAME[(FRAME.day.str[:7]==month)&FRAME.symbol.isin(a['registered'])].copy();f['score']=predict(a,f);f['baseline']=baseline(prior,f);f=apply_threshold(f,a['phase_thresholds']);sim=replay(f,0.);meta['outer']=sim['metrics'];meta['outer_gate_pass']=gate(meta['outer']);save(f,OUT/f'phase_{route}_{month}.parquet');write(OUT/f'phase_{route}_{month}_replay.json',sim)
 write(OUT/f'phase_{route}_{month}_meta.json',meta);print(json.dumps({'route':route,'month':month,'phase_thresholds':a['phase_thresholds'],'outer':meta.get('outer')}),flush=True);return meta
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--october',action='store_true');args=ap.parse_args();months=['2026-10'] if args.october else ['2026-08','2026-09'];metas=[]
 with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
  for future in as_completed([pool.submit(solve,(r,m)) for m in months for r in ROUTES]):metas.append(future.result())
 if not args.october:
  routes=[]
  for route in ROUTES:
   frames=[]
   for month in months:
    f=pd.read_parquet(OUT/f'phase_{route}_{month}.parquet');meta=next(m for m in metas if m['route']==route and m['month']==month);sim=json.loads((OUT/f'phase_{route}_{month}_replay.json').read_text());persist(f'v5phase:{route}:{month}',route,month,f,sim,meta);frames.append(f)
   f=pd.concat(frames,ignore_index=True);sim=replay(f,0.);meta=next(m for m in metas if m['route']==route and m['month']=='2026-09').copy();meta['month_thresholds']={m['month']:m['phase_thresholds'] for m in metas if m['route']==route};persist(f'v5phase:{route}:2026-08_09',route,'2026-08_09',f,sim,meta);write(OUT/f'phase_{route}_continuous_replay.json',sim);routes.append(dict(route=route,name=NAMES[route],metrics=sim['metrics'],passed=gate(sim['metrics']),months=[m for m in metas if m['route']==route]))
  passed=[r for r in routes if r['passed']];selected=max(passed,key=lambda r:(r['metrics']['ci'][0],r['metrics']['n']))['route'] if passed else None;write(OUT/'phase_results.json',dict(version='ranked_policy_v5_phase_v1',cycles=months,routes=routes,selected_route=selected,status='historical_forward_development',october_exposed=['2026-10-01','2026-10-02'],protocol_sha256=sha(OUT/'PROTOCOL.md'),phase_protocol_sha256=sha(OUT/'PHASE_PROTOCOL.md'),panel_sha256=sha(OUT/'panel.parquet')))
 else:
  results=json.loads((OUT/'phase_results.json').read_text());selected=results['selected_route'];meta=next((m for m in metas if m['route']==selected),None);write(OUT/'phase_deployment.json',dict(frozen_at=datetime.now(ET).isoformat(),observe_from='2026-10-05',observe_through='2026-10-30',route=selected,admitted=bool(meta and meta['prior_gate_pass']),metadata=meta,all_routes=metas,already_exposed=['2026-10-01','2026-10-02'],selection_basis='August September pre+regular top-one policy only'))
if __name__=='__main__':main()
