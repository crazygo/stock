"""Recompute each emitted multi-session label from the raw expected grid."""
import json,pickle,sqlite3
import numpy as np
import pandas as pd
from common import *
from horizon import DEST,TARGETS,MONTHS,DATES,action_days
from data import read_raw
from phase import phase_name,predict
from portable import digest
from store import dump

def independent_outcome(raw,event,target):
 day=event['day'];minute=event['minute']+5;cfg=TARGETS[target];i=DATES.index(day);ds=DATES[i:i+cfg['sessions']];last=ds[-1];close=570+SESSIONS[last]['duration_minutes'];end=str(pd.Timestamp(last)+pd.Timedelta(minutes=close));entry=raw[(raw.day==day)&(raw.minute==minute)]
 if len(entry)!=1:return None,'missing_entry',None,None,end
 price=float(entry.open.iloc[0]*1.001);target_price=price*(1+cfg['gain'])
 if last>'2026-09-30':return None,'pending_window',price,target_price,end
 if set(ds)&action_days(event['symbol']):return None,'corporate_action_window',price,target_price,end
 bars=[]
 for d in ds:
  first=max(570,minute) if d==day else 570;cl=570+SESSIONS[d]['duration_minutes'];expected=pd.date_range(pd.Timestamp(d)+pd.Timedelta(minutes=first),pd.Timestamp(d)+pd.Timedelta(minutes=cl-5),freq='5min');actual=raw[(raw.day==d)&(raw.minute>=first)&(raw.minute<cl)].sort_values('start')
  if not np.array_equal(actual.start.to_numpy(),expected.to_numpy()):return None,'missing_future_bars',price,target_price,end
  bars.append(actual)
 future=pd.concat(bars);hit=int((future.high>=target_price).any());return hit,'mature',price,target_price,end

def main():
 import pyarrow.parquet as pq
 report=json.loads((DEST/'results.json').read_text());audits=[];cache={};con=sqlite3.connect(OUT/'research.sqlite');wanted=set();parts=[];source_columns=set(['symbol','day','minute'])
 for head in report['targets']:
  for route in ROUTES:
   sim=json.loads((DEST/head['target_id']/f'{route}_replay.json').read_text());wanted.update((e['symbol'],e['day'],e['minute']) for e in sim['events'])
   for month in MONTHS:source_columns.update(json.loads((DEST/head['target_id']/f'{route}_{month}_meta.json').read_text())['features'])
 for batch in pq.ParquetFile(OUT/'weekly_v1/panel.parquet').iter_batches(batch_size=32768,columns=sorted(source_columns)):
  f=batch.to_pandas();f['symbol']=f.symbol.astype(str);f['day']=f.day.astype(str);ix=pd.MultiIndex.from_frame(f[['symbol','day','minute']]).isin(wanted)
  if ix.any():parts.append(f[ix])
 features=pd.concat(parts,ignore_index=True).set_index(['symbol','day','minute'],drop=False);assert len(features)==len(wanted)
 for head in report['targets']:
  target=head['target_id']
  for route in ROUTES:
   sim=json.loads((DEST/target/f'{route}_replay.json').read_text());lookup={(e['day'],e['symbol']):e for e in sim['labels']};assert len(lookup)==len(sim['events']);topchecks=0;probabilitychecks=0;rawchecks=0
   for month in MONTHS:
    path=DEST/target;meta=json.loads((path/f'{route}_{month}_meta.json').read_text());assert meta['train_max_label_end']<meta['calibration_start'];assert meta['calibration_max_label_end']<meta['selection_start'];assert meta['selection_max_label_end']<month+'-01';assert meta['algorithm']==ALGORITHMS[route]
    snapshot=DEST/'protocol_snapshots'/(meta['protocol_sha256']+'.md');assert digest(snapshot)==meta['protocol_sha256'];assert not set(meta['features'])&{'y','entry','target','label_end','future_status'}
    f=pd.read_parquet(path/f'{route}_{month}.parquet');selected=[];issued=set();previous=None
    for (day,minute),g in f.groupby(['day','minute'],sort=True,observed=True):
     if day!=previous:issued=set();previous=day
     threshold=meta['thresholds'][phase_name(minute)];q=g.iloc[:0] if threshold is None else g[(g.score>=threshold)&(g.ex_action==0)&~g.symbol.isin(issued)];q=q[pd.to_datetime(q.feature_available)<=pd.Timestamp(day)+pd.Timedelta(minutes=int(minute),seconds=30)].sort_values(['score','symbol'],ascending=[False,True])
     if len(q):s=str(q.symbol.iloc[0]);issued.add(s);selected.append((str(day),s,int(minute)))
     topchecks+=1
    expected=[(e['day'],e['symbol'],e['minute']) for e in sim['events'] if e['day'].startswith(month)];assert selected==expected
    model_path=path/'models'/f'{route}_{month}.pkl';assert digest(model_path)==meta['model_sha256'];artifact=pickle.loads(model_path.read_bytes());es=[e for e in sim['events'] if e['day'].startswith(month)]
    if es:
     source=features.loc[[(e['symbol'],e['day'],e['minute']) for e in es]].copy();num=source.select_dtypes('number').columns;source[num]=source[num].replace([np.inf,-np.inf],np.nan);np.testing.assert_allclose(predict(artifact,source),[e['score'] for e in es],rtol=0,atol=1e-12);probabilitychecks+=len(es)
   for e in sim['events']:
    s=e['symbol']
    if s not in cache:cache[s]=read_raw(s)
    hit,status,entry,tp,end=independent_outcome(cache[s],e,target);stored=lookup[(e['day'],s)];assert hit==stored['hit'],(target,route,e,stored,hit);assert status==stored['future_status'];assert end==stored['label_end']
    if entry is not None:np.testing.assert_allclose([entry,tp],[stored['entry'],stored['target']],rtol=0,atol=1e-10)
    rawchecks+=1
   m=next(r['metrics'] for r in head['routes'] if r['route']==route);assert sum(l['hit']==1 for l in sim['labels'])==m['tp'];assert sum(l['hit']==0 for l in sim['labels'])==m['fp'];assert sum(l['hit'] is None for l in sim['labels'])==m['pending'];assert sum(w['routes'][route]['n'] for w in head['weeks'])==m['n']
   dbn=con.execute('SELECT count(*) FROM horizon_events WHERE run_id=?',(f'horizon_v1:{target}:{route}:2026-05_09',)).fetchone()[0];assert dbn==m['n'];audits.append(dict(target=target,route=route,algorithm=ALGORITHMS[route],raw_labels=rawchecks,frozen_probability=probabilitychecks,top_one_ticks=topchecks,temporal_purge_all_folds=True,weekly_totals=True,sqlite_events=dbn));print(dump(audits[-1]),flush=True)
 con.close();write(DEST/'verification.json',dict(status='passed',checked_at=datetime.now(ET).isoformat(),audits=audits,results_sha256=digest(DEST/'results.json'),protocol_sha256=digest(OUT/'HORIZON_PROTOCOL.md'),not_independent_effectiveness=True))
if __name__=='__main__':main()
