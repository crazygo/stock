"""May–September extension. Never refit Aug/Sep or change October deployment."""
from __future__ import annotations
import argparse,json,pickle,csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from common import *
from data import read_raw,prior_context,intraday,peers
from train import weights,baseline,estimator
from phase import predict,phase_name
from signals import replay,metrics,gate,persist
import train

WK=OUT/'weekly_v1'
MONTHS=[f'2026-{m:02d}' for m in range(5,10)]

def warm_symbol(s):
 f=read_raw(s);ctx,_=prior_context(f,s)
 return intraday(f,ctx,s,first='2025-05-01',last='2025-05-31')

def prepare():
 WK.mkdir(exist_ok=True);frames=[]
 symbols=sorted(p.stem for p in (OUT/'raw').glob('*.parquet'))
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,future in enumerate(as_completed([pool.submit(warm_symbol,s) for s in symbols]),1):
   f=future.result()
   if not f.empty:frames.append(f)
   if i%10==0:print(json.dumps({'warmup_symbols':i,'total':len(symbols)}),flush=True)
 old=pd.read_parquet(OUT/'panel.parquet');new=peers(pd.concat(frames,ignore_index=True));mapping=old[['symbol','stock_id']].drop_duplicates().set_index('symbol').stock_id.to_dict();new['stock_id']=new.symbol.map(mapping)
 for k in new.select_dtypes('float').columns:new[k]=new[k].astype('float32')
 # Existing rows retain their exact float32 representation and frozen stock IDs.
 old['day']=old.day.astype(str);old['symbol']=old.symbol.astype(str)
 f=pd.concat([new[old.columns],old],ignore_index=True);f['day']=f.day.astype('category');f['symbol']=f.symbol.astype('category');save(f,WK/'panel.parquet')
 write(WK/'coverage.json',dict(first=f.day.astype(str).min(),last=f.day.astype(str).max(),rows=len(f),symbols=f.symbol.nunique(),added_rows=len(new),old_panel_sha256=sha(OUT/'panel.parquet'),panel_sha256=sha(WK/'panel.parquet'),membership='current_snapshot_not_PIT',protocol_sha256=sha(OUT/'WEEKLY_PROTOCOL.md')))
 print(json.dumps({'prepared_rows':len(f),'added_rows':len(new)}),flush=True)

def init():
 train.FRAME=pd.read_parquet(WK/'panel.parquet');train.FRAME['day']=train.FRAME.day.astype(str);train.FRAME['symbol']=train.FRAME.symbol.astype(str)
 cols=train.FRAME.select_dtypes('number').columns;train.FRAME[cols]=train.FRAME[cols].replace([np.inf,-np.inf],np.nan)

def fit(job):
 route,month=job;panel=train.FRAME;prior=panel[panel.day<month+'-01'];ds=sorted(prior.day.unique())[-240:]
 if len(ds)!=240:raise ValueError('Need 240 prior feature dates: '+month)
 td,cd,sd=ds[:200],ds[200:220],ds[220:]
 tr=prior[prior.day.isin(td)&(prior.minute%15==5)&prior.y.notna()].copy();counts=tr.groupby('symbol').day.nunique();registered=sorted(counts[counts>=60].index);tr=tr[tr.symbol.isin(registered)];prior=prior[prior.symbol.isin(registered)]
 ca=prior[prior.day.isin(cd)&prior.y.notna()];se=prior[prior.day.isin(sd)].copy();se['baseline']=baseline(prior[prior.day<sd[0]],se)
 assert str(tr.label_end.max())<cd[0] and cd[-1]<sd[0] and sd[-1]<month+'-01'
 cols=train.features(route);model=estimator(route,cols)
 if route=='relative_flow':model.fit(tr[cols],tr.y,sample_weight=weights(tr))
 else:model.fit(tr[cols],tr.y,learner__sample_weight=weights(tr))
 a=dict(route=route,month=month,model=model,features=cols,registered=registered,phase_calibrators={},stock_map=panel[['symbol','stock_id']].drop_duplicates().set_index('symbol').stock_id.to_dict())
 for phase in ['pre','regular']:
  f=ca[ca.minute<=570] if phase=='pre' else ca[ca.minute>570]
  raw=model.predict_proba(f[cols])[:,1];a['phase_calibrators'][phase]=LogisticRegression(C=1,max_iter=1000,random_state=1004).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),f.y,sample_weight=weights(f))
 se['score']=predict(a,se);ths={};choices={}
 for phase in ['pre','regular']:
  f=se[se.minute<=570] if phase=='pre' else se[se.minute>570];cs=[]
  for t in THRESHOLDS:
   m=replay(f,{'pre':t,'regular':t})['metrics'];cs.append(dict(threshold=t,metrics=m,passed=gate(m,n=8,days=6)))
  valid=[c for c in cs if c['passed']];best=max(valid,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'])) if valid else None;ths[phase]=best['threshold'] if best else None;choices[phase]=cs
 a['signal_thresholds']=ths;path=WK/'models'/f'{route}_{month}.pkl';path.parent.mkdir(exist_ok=True);path.write_bytes(pickle.dumps(a))
 test=panel[(panel.day.str[:7]==month)&panel.symbol.isin(registered)].copy();test['score']=predict(a,test);test['baseline']=baseline(prior,test);sim=replay(test,ths)
 meta=dict(route=route,month=month,thresholds=ths,choices=choices,registered=registered,train_first=td[0],train_end=td[-1],train_days=len(td),calibration_start=cd[0],calibration_end=cd[-1],selection_start=sd[0],selection_end=sd[-1],train_rows=len(tr),features=cols,model_sha256=sha(path),protocol_sha256=sha(OUT/'WEEKLY_PROTOCOL.md'),outer=sim['metrics'])
 save(test,WK/f'{route}_{month}.parquet');write(WK/f'{route}_{month}_meta.json',meta);write(WK/f'{route}_{month}_replay.json',sim)
 print(json.dumps({'route':route,'month':month,'thresholds':ths,'n':sim['metrics']['n'],'tp':sim['metrics']['tp']}),flush=True)
 return meta

def aggregate():
 report=dict(version='weekly_signal_v1',start='2026-05-01',end='2026-09-30',week_definition='ET Monday-Sunday, clipped to evaluation range',status='exposed_historical_forward_development',routes=[],weeks=[],protocol_sha256=sha(OUT/'WEEKLY_PROTOCOL.md'))
 sims={}
 for route in ROUTES:
  ev=[];la=[];ti=[];metas=[]
  for month in MONTHS:
   source=WK if month<'2026-08' else OUT;prefix='' if source==WK else 'signal_';sim=json.loads((source/f'{prefix}{route}_{month}_replay.json').read_text());meta=json.loads((source/f'{prefix}{route}_{month}_meta.json').read_text());ev+=sim['events'];la+=sim['labels'];ti+=sim['ticks'];metas.append(meta)
  sim=dict(events=ev,labels=la,ticks=ti,metrics=metrics(ev,la,ti));sims[route]=sim
  sim['metrics']['potential_win_stock_days']=0
  for month in MONTHS:
   source=WK if month<'2026-08' else OUT;prefix='' if source==WK else 'phase_';f=pd.read_parquet(source/f'{prefix}{route}_{month}.parquet',columns=['symbol','day','y']);sim['metrics']['potential_win_stock_days']+=len(f[f.y==1][['symbol','day']].drop_duplicates())
  possible=sim['metrics']['potential_win_stock_days'];sim['metrics']['recall_unique_stock_day']=sim['metrics']['tp']/possible if possible else None
  m=sim['metrics'];favorite_set={r['code'][3:] for r in json.loads((OUT/'universe.json').read_text())['favorites'] if r['code'].startswith('US.')};m['favorite_passed']=[s for s in m['stocks'] if s['symbol'] in favorite_set and s['passed']]
  report['routes'].append(dict(route=route,name=NAMES[route],metrics=m,months=metas,passed=gate(m)))
  write(WK/f'{route}_replay.json',sim);persist(f'weekly_signal_v1:{route}:2026-05_09',route,dict(version='weekly_signal_v1',months=[{k:x.get(k) for k in ['month','thresholds','model_sha256','train_end','selection_end']} for x in metas],protocol_sha256=report['protocol_sha256']),sim)
 days=[d for d in SESSIONS if report['start']<=d<=report['end']];weeks=sorted({str((pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in days})
 for week in weeks:
  last=str((pd.Timestamp(week)+pd.Timedelta(days=6)).date());ds=[d for d in days if week<=d<=last];row=dict(week_start=week,week_end=last,first_session=ds[0],last_session=ds[-1],sessions=len(ds),partial=week<report['start'] or last>report['end'],routes={})
  for route,sim in sims.items():
   parts={k:[e for e in sim[k] if week<=e['day']<=last] for k in ['events','labels','ticks']};row['routes'][route]=metrics(parts['events'],parts['labels'],parts['ticks'])
  report['weeks'].append(row)
 write(WK/'results.json',report)
 with (WK/'weekly.csv').open('w',newline='') as file:
  writer=csv.DictWriter(file,lineterminator='\n',fieldnames=['week_start','first_session','last_session','sessions','route','signals','tp','fp','pending','precision','ci_lower','ci_upper','abstentions','baseline','lift']);writer.writeheader()
  for w in report['weeks']:
   for route,m in w['routes'].items():writer.writerow(dict(week_start=w['week_start'],first_session=w['first_session'],last_session=w['last_session'],sessions=w['sessions'],route=route,signals=m['n'],tp=m['tp'],fp=m['fp'],pending=m['pending'],precision=m['precision'],ci_lower=m['ci'][0],ci_upper=m['ci'][1],abstentions=m['abstentions'],baseline=m['baseline'],lift=m['lift']))
 lines=['# 2026 年五月至九月 · 周粒度有效信号回测','', '表格为「命中 / 成熟有效信号数（达成率）」；— 表示未发出有效信号。每个模型分别统计，不合并跨模型重复股票日。五月首周与九月末周是截断周。','', '| ET 周（交易日） | 交易日数 | 逻辑回归 | LightGBM | ExtraTrees |','|---|---:|---:|---:|---:|']
 for w in report['weeks']:
  cells=[f"{m['tp']}/{m['mature']}（{m['precision']:.1%}）" if m['mature'] else '—' for m in w['routes'].values()];lines.append('| '+w['first_session'][5:]+'–'+w['last_session'][5:]+' | '+str(w['sessions'])+' | '+' | '.join(cells)+' |')
 lines+=['','## 合计','', '| 模型 | 有效信号 | 命中 | 失败 | 待成熟 | 达成率 | 95% Wilson | 基准 | 增量 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 def pct(x):return '—' if x is None else f'{x:.2%}'
 for r in report['routes']:
  m=r['metrics'];lines.append(f"| {r['name']} | {m['n']} | {m['tp']} | {m['fp']} | {m['pending']} | {pct(m['precision'])} | {pct(m['ci'][0])}–{pct(m['ci'][1])} | {pct(m['baseline'])} | {pct(m['lift'])} |")
 lines+=['','逐周置信区间、弃权、基准、增量与逐股结果见 results.json / weekly.csv。样本可能集中且相关，Wilson 不等于日期块独立置信界。已暴露历史不是未来独立验证；十月部署资格未改变。']
 (WK/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'summary':[{ 'route':r['route'],'n':r['metrics']['n'],'tp':r['metrics']['tp'],'precision':r['metrics']['precision']} for r in report['routes']],'weeks':len(report['weeks'])}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--fit',action='store_true');ap.add_argument('--aggregate',action='store_true');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
 if args.prepare:prepare()
 if args.fit:
  with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
   for future in as_completed([pool.submit(fit,(r,m)) for m in MONTHS[:3] for r in ROUTES]):future.result()
 if args.aggregate:aggregate()
if __name__=='__main__':main()
