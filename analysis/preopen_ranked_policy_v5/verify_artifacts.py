"""Audit the actual saved buys against raw prices and frozen model artifacts."""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
from data import read_raw,prior_context,intraday
from phase import predict
from store import connect

def main():
 con=connect();raws={};contexts={};audits=[];model_hashes={}
 assert pd.read_parquet(OUT/'panel.parquet',columns=['day']).day.astype(str).max()=='2026-09-30'
 for path in sorted((OUT/'models').glob('*.pkl')):model_hashes[path.name]=sha(path)
 for version in ['phase','capacity']:
  results=json.loads((OUT/f'{version}_results.json').read_text())
  for route in ROUTES:
   for month in (['2026-08','2026-09','2026-10'] if version=='phase' else ['2026-08','2026-09']):
    meta=json.loads((OUT/f'{version}_{route}_{month}_meta.json').read_text())
    assert meta['artifact_sha256']==model_hashes[meta['artifact']]
    assert meta['base_artifact_sha256']==model_hashes[f'{route}_{month}.pkl']
    assert meta['train_end']<meta['calibration_start']<=meta['calibration_end']<meta['selection_start']<=meta['selection_end']<month+'-01'
    if month!='2026-10':
     a=pickle.loads((OUT/'models'/meta['artifact']).read_bytes());f=pd.read_parquet(OUT/f'{version}_{route}_{month}.parquet');f=f.groupby('minute',observed=True).head(2)
     np.testing.assert_allclose(predict(a,f),f.score_original,rtol=0,atol=1e-12)
   run=f'v5{version}:{route}:2026-08_09'
   trades=[dict(x) for x in con.execute('SELECT t.*,o.hit,o.exit,o.exit_minute,o.net,o.mae FROM trades t JOIN outcomes o ON o.run_id=t.run_id AND o.trade_id=t.id WHERE t.run_id=? ORDER BY day,decision_minute',(run,))]
   decisions=[dict(x) for x in con.execute('SELECT * FROM decisions WHERE run_id=? ORDER BY day,minute',(run,))]
   assert len(decisions)==5796 and len({(d['day'],d['minute']) for d in decisions})==5796
   assert sum(d['minute']<=570 for d in decisions)==2772
   assert len({(t['day'],t['symbol']) for t in trades})==len(trades)
   previous=None;entry_vol_part=[];errors=[];price_error=0.;raw_features_checked=0
   for t in trades:
    day=t['day'];s=t['symbol'];minute=t['decision_minute'];p=json.loads(t['payload']);a=pickle.loads((OUT/'models'/f'{version}_{route}_{day[:7]}.pkl').read_bytes())
    eligible=con.execute('SELECT * FROM candidates WHERE run_id=? AND day=? AND minute=? AND qualified=1 ORDER BY score DESC,symbol LIMIT 1',(run,day,minute)).fetchone()
    assert eligible and eligible['symbol']==s and np.isclose(eligible['score'],t['score'],atol=1e-12)
    assert t['entry_minute']==minute+5 and pd.Timestamp(p['feature_available'])<=pd.Timestamp(day)+pd.Timedelta(minutes=minute,seconds=30)
    assert not set(['y','net','exit','exit_minute','mae','mfe','label_end','complete_future'])&p['feature_row'].keys()
    if version=='capacity':
     assert t['qty']<=np.floor(p['feature_row']['last_volume']*.01)
     decision_payload=json.loads(con.execute('SELECT payload FROM decisions WHERE run_id=? AND day=? AND minute=?',(run,day,minute)).fetchone()[0])
     estimate=p['feature_row']['reference']*1.001;budget=min(decision_payload['cash_before'],decision_payload['equity']*.2)
     planned=min(np.floor(budget/estimate),np.floor(p['feature_row']['last_volume']*.01))
     assert planned*estimate>=1000.-1e-4 and t['qty']<=planned
    if previous and previous['day']==day:assert previous['exit_minute']<=minute
    previous=t
    if s not in raws:raws[s]=read_raw(s);contexts[s]=prior_context(raws[s],s)[0]
    raw=raws[s];entry_at=pd.Timestamp(day)+pd.Timedelta(minutes=t['entry_minute']);b=raw[raw.start==entry_at];assert len(b)==1
    raw_entry=float(b.open.iloc[0]*1.001);price_error=max(price_error,abs(raw_entry-t['entry']))
    np.testing.assert_allclose(t['entry'],raw_entry,rtol=6e-8,atol=1e-6)
    end=570+SESSIONS[day]['duration_minutes'];g=raw[(raw.day==day)&(raw.minute>=max(570,t['entry_minute']))&(raw.minute<end)]
    assert np.array_equal(g.minute.to_numpy(),np.arange(max(570,t['entry_minute']),end,5))
    hit=g[g.high>=raw_entry*1.03];is_hit=int(len(hit)>0);exit_minute=int(hit.minute.iloc[0]+5) if is_hit else end;exit_price=(raw_entry*1.03 if is_hit else float(g.close.iloc[-1]))*.999
    assert t['hit']==is_hit and t['exit_minute']==exit_minute
    np.testing.assert_allclose(t['exit'],exit_price,rtol=6e-8,atol=1e-6);np.testing.assert_allclose(t['net'],exit_price/raw_entry-1,rtol=1e-6,atol=1e-7)
    path=raw[(raw.day==day)&(raw.minute>=t['entry_minute'])&(raw.minute<exit_minute)]
    np.testing.assert_allclose(t['mae'],path.low.min()/raw_entry-1,rtol=1e-6,atol=1e-7)
    assert b.volume.iloc[0]>0;entry_vol_part.append(float(t['qty']/b.volume.iloc[0]))
    # Rebuild the feature vector from a source physically truncated at cutoff.
    prefix=raw[raw.end<=pd.Timestamp(day)+pd.Timedelta(minutes=minute)]
    ctx=prior_context(prefix,s)[0];f=intraday(prefix,ctx,s,labels=False,first=day,last=day);fast=intraday(prefix,ctx,s,labels=False,first=day,last=day,only_minutes={minute});pd.testing.assert_frame_equal(f[f.minute==minute].reset_index(drop=True),fast.reset_index(drop=True));r=fast.iloc[0].to_dict();r['stock_id']=a['stock_map'][s]
    cols=[k for k in a['features'] if k not in ['symbol']]
    expected=np.array([r[k] for k in cols],dtype='float32');stored=np.array([p['feature_row'].get(k,np.nan) if p['feature_row'].get(k) is not None else np.nan for k in cols],dtype='float32')
    np.testing.assert_allclose(expected,stored,rtol=1e-6,atol=1e-7,equal_nan=True);raw_features_checked+=1
    f=pd.DataFrame([p['feature_row']])
    for k in cols:f[k]=pd.to_numeric(f[k],errors='coerce').astype('float32')
    np.testing.assert_allclose(predict(a,f)[0],t['score'],rtol=0,atol=1e-12)
   m=next(r['metrics'] for r in results['routes'] if r['route']==route)
   assert len(trades)==m['n'] and sum(t['hit']==1 for t in trades)==m['tp']
   # Cash bookkeeping is independently reconstructed from fills and T+1 dates.
   cash=100000.;unsettled=[];active=None;ti=0;days=list(SESSIONS);nextday={d:days[i+1] for i,d in enumerate(days[:-1])}
   for d in decisions:
    day=d['day'];cash+=sum(v for settle,v in unsettled if settle<=day);unsettled=[(settle,v) for settle,v in unsettled if settle>day]
    if active and (active['day']<day or active['exit_minute']<=d['minute']):
     settle=nextday[active['day']];proceeds=active['qty']*active['exit']
     if settle<=day:cash+=proceeds
     else:unsettled.append((settle,proceeds))
     active=None
    p=json.loads(d['payload']);np.testing.assert_allclose(cash,p['cash_before'],rtol=1e-12,atol=1e-6)
    if d['action']=='buy':
     assert active is None;t=trades[ti];ti+=1;assert t['day']==day and t['decision_minute']==d['minute'];assert t['qty']*t['entry']<=min(cash,p['equity']*.2)+1e-6;cash-=t['qty']*t['entry'];active=t
    assert cash>=0
   if active:unsettled.append((nextday[active['day']],active['qty']*active['exit']))
   np.testing.assert_allclose((cash+sum(v for _,v in unsettled))/100000-1,m['total_return'],rtol=1e-12,atol=1e-12)
   boot=None
   if trades:
    daily=pd.DataFrame(trades).groupby('day').hit.agg(['sum','count']).to_numpy();rng=np.random.default_rng(1004);samples=daily[rng.integers(0,len(daily),(20000,len(daily)))].sum(axis=1);boot=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist()
   audits.append(dict(version=version,route=route,trades=len(trades),raw_features_checked=raw_features_checked,single_minute_optimization_parity=raw_features_checked,raw_labels_verified=len(trades),candidate_top1_verified=len(trades),cash_T1_verified=True,premarket_buys=sum(t['decision_minute']<=570 for t in trades),entry_zero_volume=0,max_actual_entry_volume_participation=max(entry_vol_part,default=0),actual_entry_participation_above_1pct=sum(x>.01 for x in entry_vol_part),maximum_price_quantization_error_dollars=price_error,precision_date_block_bootstrap_ci=boot,decisions=len(decisions),no_actions=sum(d['action']=='no_action' for d in decisions)))
 dep=json.loads((OUT/'phase_deployment.json').read_text());assert dep['route'] is None and not dep['admitted'];assert all(all(t is None for t in a['phase_thresholds'].values()) for a in dep['all_routes'])
 counts={table:con.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['runs','candidates','decisions','trades','outcomes','equity','observations','live_fills','observation_predictions','observation_outcomes']};con.close()
 files={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.py','.html'] or p.name in ['PROTOCOL.md','PHASE_PROTOCOL.md','CAPACITY_PROTOCOL.md','phase_results.json','capacity_results.json','phase_deployment.json','coverage.json']}
 write(OUT/'artifact_verification.json',dict(status='passed',checked_at=datetime.now(ET).isoformat(),routes=audits,db_counts=counts,model_hashes=model_hashes,file_hashes=files,panel_sha256=sha(OUT/'panel.parquet'),data_cutoff='2026-09-30',model_effectiveness_passed=False,universe_coverage_complete=False,execution='OHLC price touch and next-bar Open proxies; historical bid-ask unavailable',qqq_source_sha256=json.loads((OUT/'universe.json').read_text())['funds']['QQQ'].get('raw_sha256'),price_storage='float32 panel labels; raw NONE prices float64; audited with explicit quantization tolerance'))
 print(json.dumps(clean(dict(status='passed',routes=audits,db_counts=counts)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
