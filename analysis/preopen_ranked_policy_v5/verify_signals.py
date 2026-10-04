"""Independently check emitted forecasts against frozen scores and raw RTH prices."""
from common import *
from data import read_raw,prior_context,intraday,GROUPS,GROUP_OF,peers
from phase import predict,phase_name
from store import connect,dump
import json,pickle
import numpy as np
import pandas as pd

def main():
 results=json.loads((OUT/'signals_results.json').read_text());con=connect();raws={};audits=[]
 def raw(s):
  if s not in raws:raws[s]=read_raw(s)
  return raws[s]
 for r in results['routes']:
  route=r['route'];run=f'signal_v1:{route}:2026-08_09';events=[dict(e) for e in con.execute('SELECT * FROM signal_events WHERE run_id=? ORDER BY day,minute',(run,))];assert len(events)==r['metrics']['n'];assert len({(e['day'],e['symbol']) for e in events})==len(events)
  for m in r['months']:
   assert m['train_end']<m['calibration_start']<=m['calibration_end']<m['selection_start']<=m['selection_end']<m['month']+'-01'
   assert sha(OUT/'models'/m['model'])==m['model_sha256'] and sha(OUT/'signals.py')==m['implementation_sha256']
   f=pd.read_parquet(OUT/f"phase_{route}_{m['month']}.parquet");f['score']=f.score_original;issued=set();days=None;selected=[]
   for (day,minute),g in f.groupby(['day','minute'],sort=True,observed=True):
    if day!=days:issued=set();days=day
    t=m['thresholds'][phase_name(minute)];eligible=g.iloc[:0] if t is None else g[(g.score>=t)&(g.ex_action==0)&~g.symbol.isin(issued)]
    if len(eligible):
     eligible=eligible[pd.to_datetime(eligible.feature_available)<=pd.Timestamp(day)+pd.Timedelta(minutes=int(minute),seconds=30)].sort_values(['score','symbol'],ascending=[False,True])
    if not len(eligible):continue
    s=str(eligible.symbol.iloc[0]);issued.add(s);selected.append((str(day),s,int(minute)))
   actual=[(e['day'],e['symbol'],e['minute']) for e in events if e['day'].startswith(m['month'])];assert selected==actual
  labels=[];pre=0
  for e in events:
   day=e['day'];s=e['symbol'];t=e['minute'];pre+=t<=570;p=json.loads(e['payload']);assert not set(['y','entry','exit','net','mae','label_end','complete_future'])&p['feature_row'].keys();meta=next(m for m in r['months'] if day.startswith(m['month']));a=pickle.loads((OUT/'models'/meta['model']).read_bytes());g=raw(s);entry_bar=g[(g.day==day)&(g.minute==t+5)];assert len(entry_bar)==1;entry=float(entry_bar.open.iloc[0]*1.001);close=570+SESSIONS[day]['duration_minutes'];remaining=g[(g.day==day)&(g.minute>=max(570,t+5))&(g.minute<close)];assert np.array_equal(remaining.minute,np.arange(max(570,t+5),close,5));hit=int((remaining.high>=entry*1.03).any());label=dict(con.execute('SELECT * FROM signal_labels WHERE run_id=? AND day=? AND symbol=?',(run,day,s)).fetchone());assert label['hit']==hit;np.testing.assert_allclose(label['entry'],entry,rtol=6e-8,atol=1e-6);np.testing.assert_allclose(label['target'],entry*1.03,rtol=6e-8,atol=1e-6);labels.append(hit)
   needed=[s]
   if route=='relative_flow':needed=sorted({s,'QQQ'}|set(GROUPS.get(GROUP_OF.get(s,''),[])))
   rows=[]
   for other in needed:
    source=raw(other);prefix=source[source.end<=pd.Timestamp(day)+pd.Timedelta(minutes=t)];ctx=prior_context(prefix,other)[0];one=intraday(prefix,ctx,other,labels=False,first=day,last=day,only_minutes={t})
    if len(one):rows.append(one)
   f=peers(pd.concat(rows,ignore_index=True)) if route=='relative_flow' else rows[0];f=f[f.symbol==s].copy();f['stock_id']=a['stock_map'][s]
   cols=[k for k in a['features'] if k!='symbol']
   for k in cols:f[k]=f[k].astype('float32')
   expected=np.array([f[k].iloc[0] for k in cols],dtype='float32');stored=np.array([p['feature_row'].get(k,np.nan) if p['feature_row'].get(k) is not None else np.nan for k in cols],dtype='float32');np.testing.assert_allclose(expected,stored,rtol=1e-6,atol=1e-7,equal_nan=True);np.testing.assert_allclose(predict(a,f)[0],e['score'],rtol=0,atol=1e-12)
  assert sum(labels)==r['metrics']['tp'];audits.append(dict(route=route,signals=len(events),raw_labels=len(labels),raw_features=len(events),frozen_probability=len(events),independent_top1=True,premarket_signals=pre))
 con.close();write(OUT/'signals_verification.json',dict(status='passed',checked_at=datetime.now(ET).isoformat(),audits=audits,execution_independent=True,model_effectiveness_not_independent=True));print(dump(audits),flush=True)

if __name__=='__main__':main()
