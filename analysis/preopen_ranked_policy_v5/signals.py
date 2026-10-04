"""Selective top-one forecasts, independent of positions, cash or fills."""
from __future__ import annotations
import argparse,json,pickle,hashlib
import numpy as np
import pandas as pd
from common import *
from data import LABELS
from phase import predict,phase_name
from train import baseline
from store import connect,dump
from maturity import outcome

def config_hash(meta):return hashlib.sha256(dump(meta).encode()).hexdigest()

def select(rows,threshold,issued=()):
 if threshold is None:return None,'confidence_threshold_unavailable',[],[]
 eligible=[];ranked=[]
 for r in sorted(rows,key=lambda r:(-float(r['score']),r['symbol'])):
  valid=bool(np.isfinite(r['score']) and r['score']>=threshold and not r.get('ex_action',0) and r['symbol'] not in issued)
  if valid:
   due=pd.Timestamp(r['day'])+pd.Timedelta(minutes=int(r['minute']),seconds=30);available=r.get('feature_available');valid=available is not None and pd.Timestamp(available)<=due
  ranked.append(dict(symbol=r['symbol'],score=r['score'],eligible=valid,threshold=threshold))
  if valid:eligible.append(r)
 if not eligible:return None,'confidence_insufficient_or_already_issued',[],ranked[:20]
 return eligible[0],'valid_signal',eligible,ranked[:20]

def replay(frame,thresholds):
 events=[];ticks=[];labels=[];issued=set();last=None
 for (day,minute),g in frame.groupby(['day','minute'],sort=True,observed=True):
  day=str(day);minute=int(minute)
  if day!=last:issued=set();last=day
  phase=phase_name(minute);threshold=thresholds[phase]
  top,reason,eligible,ranked=select(g[['day','minute','symbol','score','ex_action','feature_available']].to_dict('records'),threshold,issued)
  ticks.append(dict(day=day,minute=minute,symbol=top['symbol'] if top else None,reason=reason,eligible=len(eligible),top20=ranked,candidates=len(g)))
  if top is None:continue
  top=g[g.symbol==top['symbol']].iloc[0].to_dict()
  # Commit validity before consulting any future label or execution information.
  issued.add(top['symbol']);features={k:clean(v) for k,v in top.items() if k not in LABELS+['score','score_original','threshold','signal']}
  e=dict(day=day,symbol=top['symbol'],minute=minute,phase=phase,score=float(top['score']),threshold=threshold,baseline=float(top['baseline']),reference=float(top['reference']),feature_available=top['feature_available'],feature_row=features)
  events.append(e)
  known=pd.notna(top.get('y'))
  labels.append(dict(day=day,symbol=top['symbol'],hit=int(top['y']) if known else None,entry=clean(top.get('entry')),target=clean(top.get('entry')*1.03) if known else None,exit_minute=int(top['exit_minute']) if known else None,exit=clean(top.get('exit')),net=clean(top.get('net')),mae=clean(top.get('mae')),label_end=top.get('label_end')))
 return dict(events=events,ticks=ticks,labels=labels,metrics=metrics(events,labels,ticks))

def metrics(events,labels,ticks):
 lookup={(r['day'],r['symbol']):r for r in labels};records=[dict(e,hit=lookup.get((e['day'],e['symbol']),{}).get('hit')) for e in events]
 n=len(records);known=[r for r in records if r['hit'] is not None];tp=sum(r['hit']==1 for r in known);mature=len(known);base=float(np.mean([r['baseline'] for r in known])) if mature else None;stocks=[]
 for s in sorted({r['symbol'] for r in records}):
  rs=[r for r in records if r['symbol']==s];ks=[r for r in rs if r['hit'] is not None];k=sum(r['hit']==1 for r in ks);p=k/len(ks) if ks else None;b=float(np.mean([r['baseline'] for r in ks])) if ks else None
  stocks.append(dict(symbol=s,n=len(rs),mature=len(ks),tp=k,fp=len(ks)-k,pending=len(rs)-len(ks),precision=p,ci=wilson(k,len(ks)),baseline=b,lift=p-b if ks else None,passed=len(ks)>=12 and len(ks)==len(rs) and p>=.7 and p-b>=.03))
 return dict(n=n,mature=mature,tp=tp,fp=mature-tp,pending=n-mature,precision=tp/mature if mature else None,conservative_precision=tp/n if n else None,baseline=base,lift=tp/mature-base if mature else None,ci=wilson(tp,mature),date_count=len({r['day'] for r in records}),ticks=len(ticks),abstentions=len(ticks)-n,coverage=n/len(ticks) if ticks else None,stocks=stocks,definition='TP/(TP+FP) over issued and matured threshold-valid signals; cash/positions/fills excluded')

def gate(m,n=12,days=10):
 return bool(m['mature']>=n and m['date_count']>=days and m['pending']==0 and m['precision']>=.7 and m['lift']>=.03)

def put_run(con,run,route,mode,meta):
 digest=config_hash(meta);old=con.execute('SELECT configuration_hash FROM signal_runs WHERE id=?',(run,)).fetchone()
 if old and old['configuration_hash']!=digest:raise RuntimeError('Immutable signal configuration changed: '+run)
 con.execute('INSERT OR IGNORE INTO signal_runs VALUES(?,?,?,?,?,?)',(run,route,mode,datetime.now(ET).isoformat(),digest,dump(meta)))

def put_tick(con,run,tick):
 con.execute('INSERT OR IGNORE INTO signal_ticks VALUES(?,?,?,?,?,?,?)',(run,tick['day'],tick['minute'],tick['symbol'],tick['reason'],tick['eligible'],dump(tick)))

def put_event(con,run,event):
 e=event;con.execute('INSERT OR IGNORE INTO signal_events VALUES(?,?,?,?,?,?,?,?,?,?,?)',(run,e['day'],e['symbol'],e['minute'],e['phase'],e['score'],e['threshold'],e['baseline'],e['reference'],e['feature_available'],dump(e)))

def put_label(con,run,label,known_at):
 l=label
 if l['hit'] is None:return
 con.execute('INSERT OR IGNORE INTO signal_labels VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(run,l['day'],l['symbol'],l['hit'],l['entry'],l['target'],l['exit_minute'],l['exit'],l['net'],l['mae'],l['label_end'],known_at))

def persist(run,route,meta,sim):
 con=connect()
 with con:
  put_run(con,run,route,'historical_forward_signal',meta)
  for t in sim['ticks']:put_tick(con,run,t)
  for e in sim['events']:put_event(con,run,e)
  for l in sim['labels']:put_label(con,run,l,str(l['label_end']))
 con.close()

def build():
 panel=pd.read_parquet(OUT/'panel.parquet');panel['day']=panel.day.astype(str);panel['symbol']=panel.symbol.astype(str);routes=[];october=[]
 for route in ROUTES:
  months=[];frames=[]
  for month in ['2026-08','2026-09','2026-10']:
   meta=json.loads((OUT/f'phase_{route}_{month}_meta.json').read_text());path=OUT/'models'/meta['artifact'];assert sha(path)==meta['artifact_sha256'];a=pickle.loads(path.read_bytes())
   assert meta['selection_end']<month+'-01'
   prior=panel[(panel.day<month+'-01')&panel.symbol.isin(a['registered'])];sel=prior[(prior.day>=meta['selection_start'])&(prior.day<=meta['selection_end'])].copy();sel['score']=predict(a,sel);sel['baseline']=baseline(prior[prior.day<meta['selection_start']],sel)
   thresholds={};details={}
   for phase in ['pre','regular']:
    f=sel[sel.minute<=570] if phase=='pre' else sel[sel.minute>570];choices=[]
    for t in THRESHOLDS:
     m=replay(f,{'pre':t,'regular':t})['metrics'];choices.append(dict(threshold=t,metrics=m,passed=gate(m,n=8,days=6)))
    passed=[c for c in choices if c['passed']];best=max(passed,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'])) if passed else None;thresholds[phase]=best['threshold'] if best else None;details[phase]=dict(threshold=thresholds[phase],choices=choices)
   sm=dict(route=route,month=month,thresholds=thresholds,phase_details=details,model=meta['artifact'],model_sha256=meta['artifact_sha256'],registered=a['registered'],train_end=meta['train_end'],calibration_start=meta['calibration_start'],calibration_end=meta['calibration_end'],selection_start=meta['selection_start'],selection_end=meta['selection_end'],protocol_sha256=sha(OUT/'SIGNAL_PROTOCOL.md'),implementation_sha256=sha(OUT/'signals.py'))
   if month!='2026-10':
    f=pd.read_parquet(OUT/f'phase_{route}_{month}.parquet');f['score']=f.score_original;f['signal_threshold']=f.minute.map(lambda m:thresholds[phase_name(m)]);sim=replay(f,thresholds);sm['outer']=sim['metrics'];months.append(sm);frames.append(f);write(OUT/f'signal_{route}_{month}_replay.json',sim)
   else:october.append(sm)
   write(OUT/f'signal_{route}_{month}_meta.json',sm);print(dump(dict(route=route,month=month,thresholds=thresholds,outer=sm.get('outer'))),flush=True)
  f=pd.concat(frames,ignore_index=True);events=[];ticks=[];labels=[]
  for m in months:
   sim=replay(f[f.day.str[:7]==m['month']],m['thresholds']);events+=sim['events'];ticks+=sim['ticks'];labels+=sim['labels']
  sim=dict(events=events,ticks=ticks,labels=labels,metrics=metrics(events,labels,ticks));sm=dict(route=route,version='signal_v1',months=months,protocol_sha256=sha(OUT/'SIGNAL_PROTOCOL.md'));persist(f'signal_v1:{route}:2026-08_09',route,sm,sim);write(OUT/f'signal_{route}_continuous_replay.json',sim);routes.append(dict(route=route,name=NAMES[route],metrics=sim['metrics'],passed=gate(sim['metrics']),months=months))
  del sel,prior,frames,f
 passed=[r for r in routes if r['passed']];selected=max(passed,key=lambda r:(r['metrics']['ci'][0],r['metrics']['n']))['route'] if passed else None;oct=next((m for m in october if m['route']==selected),None)
 write(OUT/'signals_results.json',dict(version='signal_v1',routes=routes,selected_route=selected,protocol_sha256=sha(OUT/'SIGNAL_PROTOCOL.md'),status='historical_forward_development',target='delayed-entry RTH touch +3%',already_exposed=['2026-10-01','2026-10-02']))
 write(OUT/'signals_deployment.json',dict(version='signal_v1',frozen_at=datetime.now(ET).isoformat(),route=selected,admitted=bool(oct and any(v is not None for v in oct['thresholds'].values())),all_routes=october,observe_from='2026-10-05',observe_through='2026-10-30',basis='threshold-valid signals only, no cash/position/profit gate',protocol_sha256=sha(OUT/'SIGNAL_PROTOCOL.md')))

def settle(cached_raw,now=None):
 now=now or datetime.now(ET);con=connect();rows=con.execute('''SELECT e.* FROM signal_events e JOIN signal_runs r ON r.id=e.run_id LEFT JOIN signal_labels l ON l.run_id=e.run_id AND l.day=e.day AND l.symbol=e.symbol WHERE r.mode='prospective_signal' AND l.run_id IS NULL''').fetchall();added=0;cache={}
 with con:
  for row in rows:
   e=dict(row);key=(e['day'],e['symbol'])
   if key not in cache:cache[key]=cached_raw(e['symbol'])
   l=outcome(cache[key],e['day'],e['minute']+5,now)
   if l is None:continue
   l.update(day=e['day'],symbol=e['symbol']);entry_at=pd.Timestamp(l['entry_at']);close=570+SESSIONS[e['day']]['duration_minutes'];g=cache[key];g=g[(g.day==e['day'])&(g.minute>=max(570,e['minute']+5))&(g.minute<close)];hits=g[g.high>=l['target']];l['exit_minute']=int(hits.minute.iloc[0]+5) if len(hits) else close;put_label(con,e['run_id'],l,now.isoformat());added+=1
 con.close();return added

def summary(con,run):
 events=[dict(r) for r in con.execute('SELECT * FROM signal_events WHERE run_id=? ORDER BY day,minute',(run,))];labels=[dict(r) for r in con.execute('SELECT * FROM signal_labels WHERE run_id=?',(run,))];ticks=[dict(r) for r in con.execute('SELECT day,minute FROM signal_ticks WHERE run_id=?',(run,))];return metrics(events,labels,ticks)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');a=ap.parse_args()
 if a.build:build()
