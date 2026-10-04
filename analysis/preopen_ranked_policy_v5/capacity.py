"""Registered smaller-size experiment; probabilities remain frozen."""
from __future__ import annotations
import json,pickle
import pandas as pd
from common import *
from phase import predict,apply_threshold
from train import baseline
from policy import replay,gate
from store import persist

def main():
 panel=pd.read_parquet(OUT/'panel.parquet');panel['day']=panel.day.astype(str);panel['symbol']=panel.symbol.astype(str);routes=[]
 for route in ROUTES:
  months=[];frames=[]
  for month in ['2026-08','2026-09']:
   base=OUT/'models'/f'phase_{route}_{month}.pkl';a=pickle.loads(base.read_bytes());meta=json.loads((OUT/f'phase_{route}_{month}_meta.json').read_text());prior=panel[(panel.day<month+'-01')&panel.symbol.isin(a['registered'])];sel=prior[(prior.day>=meta['selection_start'])&(prior.day<=meta['selection_end'])].copy();sel['score']=predict(a,sel);sel['baseline']=baseline(prior[prior.day<meta['selection_start']],sel);ths={};details={}
   for phase in ['pre','regular']:
    f=sel[sel.minute<=570] if phase=='pre' else sel[sel.minute>570];choices=[]
    for t in THRESHOLDS:
     sim=replay(f,t,capacity_capped=True);choices.append(dict(threshold=t,metrics=sim['metrics'],passed=gate(sim['metrics'],n=8,days=6)))
    passed=[c for c in choices if c['passed']];best=max(passed,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'])) if passed else None;ths[phase]=best['threshold'] if best else None;details[phase]=dict(threshold=ths[phase],brier=meta['phase_details'][phase]['brier'],choices=choices)
   a['phase_thresholds']=ths;a['capacity_capped']=True;a['min_notional']=1000.;p=OUT/'models'/f'capacity_{route}_{month}.pkl';p.write_bytes(pickle.dumps(a));meta.update(artifact=p.name,artifact_sha256=sha(p),parent_phase_artifact_sha256=sha(base),phase_thresholds=ths,phase_details=details,threshold=0.,capacity_policy_sha256=sha(OUT/'CAPACITY_PROTOCOL.md'),prior_gate_pass=any(t is not None for t in ths.values()))
   f=pd.read_parquet(OUT/f'phase_{route}_{month}.parquet');f['score']=f.score_original;f=apply_threshold(f,ths);sim=replay(f,0.,capacity_capped=True);meta.update(outer=sim['metrics'],outer_gate_pass=gate(sim['metrics']));write(OUT/f'capacity_{route}_{month}_meta.json',meta);save(f,OUT/f'capacity_{route}_{month}.parquet');persist(f'v5capacity:{route}:{month}',route,month,f,sim,meta);write(OUT/f'capacity_{route}_{month}_replay.json',sim);months.append(meta);frames.append(f);print(json.dumps(dict(route=route,month=month,thresholds=ths,n=sim['metrics']['n'],tp=sim['metrics']['tp'],precision=sim['metrics']['precision'])),flush=True)
  frame=pd.concat(frames,ignore_index=True);sim=replay(frame,0.,capacity_capped=True);meta=months[-1].copy();meta['month_thresholds']={m['month']:m['phase_thresholds'] for m in months};persist(f'v5capacity:{route}:2026-08_09',route,'2026-08_09',frame,sim,meta);write(OUT/f'capacity_{route}_continuous_replay.json',sim);routes.append(dict(route=route,name=NAMES[route],metrics=sim['metrics'],passed=gate(sim['metrics']),premarket_buys=sum(t['minute']<=570 for t in sim['trades']),months=months));print(json.dumps(dict(route=route,continuous=sim['metrics'],premarket_buys=routes[-1]['premarket_buys'])),flush=True)
  del frame,sel,prior,frames
 passed=[r for r in routes if r['passed']];selected=max(passed,key=lambda r:(r['metrics']['ci'][0],r['metrics']['n']))['route'] if passed else None
 write(OUT/'capacity_results.json',dict(version='ranked_policy_v5_capacity_v1',routes=routes,selected_route=selected,protocol_sha256=sha(OUT/'CAPACITY_PROTOCOL.md'),panel_sha256=sha(OUT/'panel.parquet'),status='historical_forward_development_extra_attempt',default_dashboard=False))

if __name__=='__main__':main()
