"""Independent top-one, raw label and causal feature checks for weekly_signal_v1."""
import json,pickle
import numpy as np
import pandas as pd
from common import *
from data import read_raw,prior_context,intraday,peers,GROUPS,GROUP_OF
from phase import predict,phase_name
from store import connect
from portable import digest
from weekly import WK,MONTHS

def main():
 result=json.loads((WK/'results.json').read_text());con=connect();audits=[]
 for route in ROUTES:
  sim=json.loads((WK/f'{route}_replay.json').read_text());assert len({(e['day'],e['symbol']) for e in sim['events']})==len(sim['events']);rawcache={};contextcache={};artifacts={};top_checks=0;pre=0;labels=[]
  def raw(s):
   if s not in rawcache:rawcache[s]=read_raw(s)
   return rawcache[s]
  def context(s,day):
   key=(s,day)
   if key not in contextcache:
    # Context only knows previous days and this day's night, ending at 04:00.
    prefix=raw(s)[raw(s).end<=pd.Timestamp(day)+pd.Timedelta(hours=4)];contextcache[key]=prior_context(prefix,s)[0]
   return contextcache[key]
  for month in MONTHS:
   source=WK if month<'2026-08' else OUT;prefix='' if source==WK else 'signal_';meta=json.loads((source/f'{prefix}{route}_{month}_meta.json').read_text());artifact=WK/'models'/f'{route}_{month}.pkl' if source==WK else OUT/'models'/meta['model'];assert digest(artifact)==meta['model_sha256'];a=pickle.loads(artifact.read_bytes());artifacts[month]=a;assert meta['train_end']<meta['calibration_start']<=meta['calibration_end']<meta['selection_start']<=meta['selection_end']<month+'-01'
   frame=pd.read_parquet(source/f"{'' if source==WK else 'phase_'}{route}_{month}.parquet");frame['score']=frame.score if source==WK else frame.score_original;issued=set();previous=None;selected=[]
   for (day,minute),g in frame.groupby(['day','minute'],observed=True,sort=True):
    if previous!=day:issued=set();previous=day
    threshold=meta['thresholds'][phase_name(minute)]
    eligible=g.iloc[:0] if threshold is None else g[(g.score>=threshold)&(g.ex_action==0)&~g.symbol.isin(issued)]
    if len(eligible):eligible=eligible[pd.to_datetime(eligible.feature_available)<=pd.Timestamp(day)+pd.Timedelta(minutes=int(minute),seconds=30)].sort_values(['score','symbol'],ascending=[False,True])
    if len(eligible):s=str(eligible.symbol.iloc[0]);issued.add(s);selected.append((str(day),s,int(minute)))
    top_checks+=1
   actual=[(e['day'],e['symbol'],e['minute']) for e in sim['events'] if e['day'].startswith(month)];assert selected==actual
  for i,e in enumerate(sim['events']):
   s=e['symbol'];day=e['day'];t=e['minute'];a=artifacts[day[:7]];g=raw(s);bar=g[(g.day==day)&(g.minute==t+5)];close=570+SESSIONS[day]['duration_minutes'];future=g[(g.day==day)&(g.minute>=max(570,t+5))&(g.minute<close)];pre+=t<=570
   known=len(bar)==1 and np.array_equal(future.minute,np.arange(max(570,t+5),close,5));label=next(l for l in sim['labels'] if l['day']==day and l['symbol']==s)
   if known:
    entry=float(bar.open.iloc[0]*1.001);hit=int((future.high>=entry*1.03).any());assert label['hit']==hit;np.testing.assert_allclose(label['entry'],entry,rtol=6e-8,atol=1e-6);labels.append(hit)
   else:assert label['hit'] is None
   needed=sorted({s,'QQQ'}|set(GROUPS.get(GROUP_OF.get(s,''),[]))) if route=='relative_flow' else [s];rows=[]
   for other in needed:
    prefix=raw(other)[raw(other).end<=pd.Timestamp(day)+pd.Timedelta(minutes=t)];one=intraday(prefix,context(other,day),other,labels=False,first=day,last=day,only_minutes={t})
    if len(one):rows.append(one)
   f=peers(pd.concat(rows,ignore_index=True)) if route=='relative_flow' else rows[0];f=f[f.symbol==s].copy();f['stock_id']=a['stock_map'][s];cols=[c for c in a['features'] if c!='symbol']
   for c in cols:f[c]=f[c].astype('float32')
   stored=np.array([e['feature_row'].get(c,np.nan) if e['feature_row'].get(c) is not None else np.nan for c in cols],dtype='float32');computed=f[cols].to_numpy()[0];np.testing.assert_allclose(stored,computed,rtol=1e-6,atol=1e-7,equal_nan=True);np.testing.assert_allclose(predict(a,f)[0],e['score'],rtol=0,atol=1e-12)
   assert not {'y','entry','label_end','exit','net','mae','complete_future'}&e['feature_row'].keys()
   if (i+1)%20==0:print(json.dumps({'route':route,'signals_checked':i+1}),flush=True)
  m=next(r['metrics'] for r in result['routes'] if r['route']==route);assert sum(labels)==m['tp'];assert len(labels)==m['mature'];weekly=[w['routes'][route] for w in result['weeks']];assert sum(w['n'] for w in weekly)==m['n'];assert sum(w['tp'] for w in weekly)==m['tp'];assert sum(w['fp'] for w in weekly)==m['fp'];assert sum(w['ticks'] for w in weekly)==m['ticks']
  dbn=con.execute('SELECT count(*) FROM signal_events WHERE run_id=?',(f'weekly_signal_v1:{route}:2026-05_09',)).fetchone()[0];assert dbn==m['n'];audits.append(dict(route=route,signals=m['n'],raw_labels=len(labels),causal_features=m['n'],frozen_probability=m['n'],top_one_ticks=top_checks,premarket_signals=pre,weekly_counts_match=True,sqlite_counts_match=True))
 con.close();write(WK/'verification.json',dict(status='passed',checked_at=datetime.now(ET).isoformat(),audits=audits,protocol_sha256=digest(OUT/'WEEKLY_PROTOCOL.md'),results_sha256=digest(WK/'results.json'),not_independent_effectiveness=True));print(json.dumps(audits),flush=True)
if __name__=='__main__':main()
