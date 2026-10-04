"""Registered, purged monthly forecasts for 5 sessions/+5% and 10/+10%."""
from __future__ import annotations
import argparse,csv,json,pickle,sqlite3,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from common import *
from data import read_raw,LABELS
from train import estimator,weights,baseline
from phase import predict,phase_name
from signals import select,metrics,gate
from store import dump
import train

DEST=OUT/'horizon_v1'
TARGETS={'5d5pct':dict(sessions=5,gain=.05,name='5 个交易日触及 +5%'),
         '10d10pct':dict(sessions=10,gain=.10,name='10 个交易日触及 +10%')}
MONTHS=[f'2026-{m:02d}' for m in range(5,10)]
CUTOFF='2026-09-30'
BASE=None;PANEL=None;COLS=None;HEAD=None
DATES=sorted(SESSIONS);DATEPOS={d:i for i,d in enumerate(DATES)}

def window_end(day,target):
 i=DATEPOS[day]+TARGETS[target]['sessions']-1
 if i>=len(DATES):return None
 last=DATES[i];return str(pd.Timestamp(last)+pd.Timedelta(minutes=570+SESSIONS[last]['duration_minutes']))

def action_days(symbol):
 days=set()
 for base in [ROOT/'market_data/model_training_history_v1/corporate_actions',ROOT/'market_data/corporate_actions',ROOT/'analysis/preopen_intraday_v2/cache/corporate_actions']:
  path=base/(symbol+'.parquet')
  if path.exists():
   f=pd.read_parquet(path)
   if 'ex_div_date' in f:days.update(f.ex_div_date.astype(str).str[:10])
 return days

def valid_bars(raw):
 v=raw[['open','high','low','close','volume']].to_numpy(float)
 return np.isfinite(v).all(axis=1)&(v[:,:4]>0).all(axis=1)&(v[:,1]>=np.maximum(v[:,0],v[:,3]))&(v[:,2]<=np.minimum(v[:,0],v[:,3]))&(v[:,4]>=0)

def label_rows(raw,rows,target,actions=(),cutoff=CUTOFF):
 """Vectorized complete-window labels. All censored cohorts remain unknown."""
 n=len(rows);out=pd.DataFrame(dict(y=np.full(n,np.nan),entry=np.full(n,np.nan),target=np.full(n,np.nan),
  label_end=['']*n,future_status=['missing_entry']*n),index=rows.index)
 if raw.empty:return out
 raw=raw.sort_values('start').drop_duplicates('start',keep='last').copy();raw['_valid']=valid_bars(raw);groups={str(d):g for d,g in raw.groupby('day',sort=True)};daily={}
 for day,g in groups.items():
  if day not in SESSIONS:continue
  close=570+SESSIONS[day]['duration_minutes'];r=g[(g.minute>=570)&(g.minute<close)]
  daily[day]=dict(complete=np.array_equal(r.minute.to_numpy(),np.arange(570,close,5)) and bool(r._valid.all()),high=float(r.high.max()) if len(r) else np.nan)
 cfg=TARGETS[target]
 for day,part in rows.groupby('day',sort=True,observed=True):
  day=str(day);end=window_end(day,target);out.loc[part.index,'label_end']=end or '';g=groups.get(day)
  if g is None or day not in DATEPOS:continue
  minutes=g.minute.to_numpy();entrymins=part.minute.to_numpy(dtype=int)+5;idx=np.searchsorted(minutes,entrymins);good=idx<len(g);good[good]=(minutes[idx[good]]==entrymins[good])&g._valid.to_numpy()[idx[good]]
  entries=np.full(len(part),np.nan);entries[good]=g.open.to_numpy()[idx[good]]*1.001;out.loc[part.index,'entry']=entries;out.loc[part.index,'target']=entries*(1+cfg['gain'])
  valid=part.index[good]
  if end is None or end[:10]>cutoff:out.loc[valid,'future_status']='pending_window';continue
  days=DATES[DATEPOS[day]:DATEPOS[day]+cfg['sessions']]
  if set(days)&set(actions):out.loc[valid,'future_status']='corporate_action_window';continue
  if any(not daily.get(d,{}).get('complete',False) for d in days[1:]):out.loc[valid,'future_status']='missing_future_bars';continue
  close=570+SESSIONS[day]['duration_minutes'];r=g[(g.minute>=570)&(g.minute<close)];rm=r.minute.to_numpy();rh=r.high.to_numpy();suffix=np.maximum.accumulate(rh[::-1])[::-1] if len(r) else np.array([])
  start=np.maximum(570,entrymins);pos=np.searchsorted(rm,start);complete=np.asarray([np.array_equal(rm[p:],np.arange(m,close,5)) and bool(r._valid.iloc[p:].all()) for p,m in zip(pos,start)])
  peak=np.full(len(part),np.nan);ok=good&complete&(pos<len(r));next_high=max([daily[d]['high'] for d in days[1:]],default=-np.inf)
  peak[ok]=np.maximum(suffix[pos[ok]],next_high);out.loc[part.index[good&~complete],'future_status']='missing_future_bars';out.loc[part.index[ok],'y']=(peak[ok]>=entries[ok]*(1+cfg['gain'])).astype(float);out.loc[part.index[ok],'future_status']='mature'
 return out

def prepare():
 DEST.mkdir(exist_ok=True);keys=pd.read_parquet(OUT/'weekly_v1/panel.parquet',columns=['symbol','day','minute']);keys['day']=keys.day.astype(str);keys['symbol']=keys.symbol.astype(str)
 parts={t:[] for t in TARGETS};coverage=[]
 for i,(symbol,g) in enumerate(keys.groupby('symbol',sort=True),1):
  raw=read_raw(symbol);actions=action_days(symbol)
  for target in TARGETS:
   labels=label_rows(raw,g,target,actions);labels['row_id']=g.index;parts[target].append(labels)
  coverage.append(dict(symbol=symbol,raw_first=str(raw.start.min()),raw_last=str(raw.end.max()),rows=len(g),stored_action_dates=len(actions)))
  if i%10==0:print(dump(dict(label_symbols=i,total=keys.symbol.nunique())),flush=True)
 stats={}
 for target,ps in parts.items():
  f=pd.concat(ps).sort_values('row_id').reset_index(drop=True);assert np.array_equal(f.row_id,np.arange(len(keys)));save(f,DEST/f'labels_{target}.parquet');stats[target]=dict(rows=len(f),status=f.future_status.value_counts().to_dict(),label_sha256=sha(DEST/f'labels_{target}.parquet'))
 write(DEST/'coverage.json',dict(requested_feature_range=['2025-05-01','2026-09-30'],outcome_cutoff=CUTOFF,rows=len(keys),symbols=coverage,targets=stats,panel_sha256=sha(OUT/'weekly_v1/panel.parquet'),protocol_sha256=sha(OUT/'HORIZON_PROTOCOL.md'),membership='current_snapshot_not_PIT'))

def init(target):
 global BASE,PANEL,COLS,HEAD
 HEAD=target;BASE=pd.read_parquet(OUT/'weekly_v1/panel.parquet');BASE['day']=BASE.day.astype(str);BASE['symbol']=BASE.symbol.astype(str)
 train.FRAME=BASE;COLS={r:train.features(r) for r in ROUTES};BASE=BASE.drop(columns=LABELS,errors='ignore')
 labels=pd.read_parquet(DEST/f'labels_{target}.parquet').drop(columns='row_id');assert len(BASE)==len(labels);PANEL=pd.concat([BASE,labels],axis=1)
 cols=PANEL.select_dtypes('number').columns;PANEL[cols]=PANEL[cols].replace([np.inf,-np.inf],np.nan);BASE=None;train.FRAME=None

def split_dates(month,available):
 """Uniform ten-session purges for both heads, including selection→test."""
 end=DATEPOS[min(d for d in DATES if d>=month+'-01')]-10
 sd=DATES[max(0,end-20):end];cend=DATEPOS[sd[0]]-10;cd=DATES[max(0,cend-20):cend];tend=DATEPOS[cd[0]]-10;td=[d for d in DATES[max(0,tend-200):tend] if d in available]
 assert window_end(sd[-1],'10d10pct')[:10]<month+'-01'
 return td,cd,sd

def replay(frame,thresholds,target):
 events=[];labels=[];ticks=[];issued=set();last=None
 for (day,minute),g in frame.groupby(['day','minute'],sort=True,observed=True):
  day=str(day);minute=int(minute)
  if day!=last:issued=set();last=day
  phase=phase_name(minute);threshold=thresholds[phase];top,reason,eligible,_=select(g[['day','minute','symbol','score','ex_action','feature_available']].to_dict('records'),threshold,issued)
  ticks.append(dict(day=day,minute=minute,symbol=top['symbol'] if top else None,reason=reason,eligible=len(eligible),candidates=len(g)))
  if top is None:continue
  issued.add(top['symbol']);r=g[g.symbol==top['symbol']].iloc[0].to_dict();e={k:clean(r.get(k)) for k in ['day','symbol','minute','score','baseline','reference','feature_available']};e.update(phase=phase,threshold=threshold,target_id=target);events.append(e)
  labels.append(dict(day=day,symbol=r['symbol'],hit=int(r['y']) if pd.notna(r['y']) else None,entry=clean(r['entry']),target=clean(r['target']),label_end=r['label_end'],future_status=r['future_status']))
 return dict(events=events,labels=labels,ticks=ticks,metrics=metrics(events,labels,ticks))

def calibration_metrics(f):
 valid=f[f.y.notna()].copy()
 if valid.empty:return dict(rows=0,stock_days=0,brier=None,baseline_brier=None,bins=[])
 w=weights(valid);p=valid.score.to_numpy();y=valid.y.to_numpy();total=w.sum();bins=[]
 for lo in np.arange(0,1,.1):
  ix=(p>=lo)&(p<lo+.1+1e-12 if lo>.89 else p<lo+.1)
  if ix.any():bins.append(dict(lower=float(lo),upper=float(lo+.1),stock_day_weight=float(w[ix].sum()),predicted=float(np.average(p[ix],weights=w[ix])),observed=float(np.average(y[ix],weights=w[ix]))))
 return dict(rows=len(valid),stock_days=valid[['symbol','day']].drop_duplicates().shape[0],brier=float(np.average((p-y)**2,weights=w)),baseline_brier=float(np.average((valid.baseline.to_numpy()-y)**2,weights=w)),predicted=float(np.average(p,weights=w)),observed=float(np.average(y,weights=w)),bins=bins)

def fit(job):
 route,month=job;target=HEAD;td,cd,sd=split_dates(month,set(PANEL.day));prior=PANEL[PANEL.day<month+'-01'];tr=prior[prior.day.isin(td)&(prior.minute%15==5)&prior.y.notna()].copy();counts=tr.groupby('symbol').day.nunique();registered=sorted(counts[counts>=60].index);tr=tr[tr.symbol.isin(registered)];prior=prior[prior.symbol.isin(registered)]
 ca=prior[prior.day.isin(cd)&prior.y.notna()];se=prior[prior.day.isin(sd)].copy();assert tr.label_end.max()<cd[0] and ca.label_end.max()<sd[0] and se.label_end.max()<month+'-01'
 cols=COLS[route];assert not set(cols)&{'y','entry','target','label_end','future_status','mfe','mae'};model=estimator(route,cols)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  if route=='relative_flow':model.fit(tr[cols],tr.y,sample_weight=weights(tr))
  else:model.fit(tr[cols],tr.y,learner__sample_weight=weights(tr))
 if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('optimizer not converged: '+route)
 a=dict(route=route,month=month,target_id=target,model=model,features=cols,registered=registered,phase_calibrators={},stock_map=PANEL[['symbol','stock_id']].drop_duplicates().set_index('symbol').stock_id.to_dict())
 for phase in ['pre','regular']:
  f=ca[ca.minute<=570] if phase=='pre' else ca[ca.minute>570];raw=model.predict_proba(f[cols])[:,1];a['phase_calibrators'][phase]=LogisticRegression(C=1,max_iter=1000,random_state=1004).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),f.y,sample_weight=weights(f))
 history=prior[prior.day.isin(td+cd)&prior.y.notna()];se['score']=predict(a,se);se['baseline']=baseline(history,se);thresholds={};choices={}
 for phase in ['pre','regular']:
  f=se[se.minute<=570] if phase=='pre' else se[se.minute>570];cs=[]
  for threshold in THRESHOLDS:
   m=replay(f,{'pre':threshold,'regular':threshold},target)['metrics'];cs.append(dict(threshold=threshold,metrics=m,passed=gate(m,n=8,days=6)))
  ok=[c for c in cs if c['passed']];best=max(ok,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'])) if ok else None;thresholds[phase]=best['threshold'] if best else None;choices[phase]=cs
 a['signal_thresholds']=thresholds
 counts2=history.groupby(['symbol','minute'],observed=True).y.agg(['sum','count']);gm=history.groupby('minute',observed=True).y.mean();a['baseline_table']={f'{s}|{m}':float((r['sum']+20*gm.get(m,.1))/(r['count']+20)) for (s,m),r in counts2.iterrows()};a['baseline_minute']=gm.to_dict()
 dest=DEST/target;dest.mkdir(exist_ok=True);(dest/'models').mkdir(exist_ok=True);path=dest/'models'/f'{route}_{month}.pkl';path.write_bytes(pickle.dumps(a))
 test=PANEL[(PANEL.day.str[:7]==month)&PANEL.symbol.isin(registered)].copy();test['score']=predict(a,test) if len(test) else pd.Series(dtype=float);test['baseline']=baseline(history,test);sim=replay(test,thresholds,target)
 meta=dict(route=route,name=NAMES[route],algorithm=ALGORITHMS[route],target_id=target,month=month,thresholds=thresholds,choices=choices,registered=registered,train_first=td[0],train_end=td[-1],train_days=len(td),calibration_start=cd[0],calibration_end=cd[-1],selection_start=sd[0],selection_end=sd[-1],train_rows=len(tr),train_max_label_end=tr.label_end.max(),calibration_max_label_end=ca.label_end.max(),selection_max_label_end=se.label_end.max(),purge_sessions=10,features=cols,model_sha256=sha(path),protocol_sha256=sha(OUT/'HORIZON_PROTOCOL.md'),outer=sim['metrics'],probability_evaluation=calibration_metrics(test),selection_probability=calibration_metrics(se))
 write(dest/f'{route}_{month}_meta.json',meta);write(dest/f'{route}_{month}_replay.json',sim);save(test[['symbol','day','minute','score','baseline','reference','ex_action','feature_available','y','entry','target','label_end','future_status']],dest/f'{route}_{month}.parquet')
 print(dump(dict(target=target,route=route,month=month,thresholds=thresholds,n=sim['metrics']['n'],tp=sim['metrics']['tp'],pending=sim['metrics']['pending'])),flush=True);return meta

def persist(run,route,target,meta,sim):
 con=sqlite3.connect(OUT/'research.sqlite',timeout=60);con.execute('PRAGMA journal_mode=WAL');config=__import__('hashlib').sha256(dump(meta).encode()).hexdigest()
 con.executescript('''CREATE TABLE IF NOT EXISTS horizon_runs(id TEXT PRIMARY KEY,route TEXT,target TEXT,configuration_hash TEXT,metadata TEXT);
 CREATE TABLE IF NOT EXISTS horizon_ticks(run_id TEXT,day TEXT,minute INTEGER,payload TEXT,PRIMARY KEY(run_id,day,minute));
 CREATE TABLE IF NOT EXISTS horizon_events(run_id TEXT,day TEXT,symbol TEXT,minute INTEGER,score REAL,threshold REAL,payload TEXT,PRIMARY KEY(run_id,day,symbol));
 CREATE TABLE IF NOT EXISTS horizon_labels(run_id TEXT,day TEXT,symbol TEXT,hit INTEGER,status TEXT,payload TEXT,PRIMARY KEY(run_id,day,symbol));''')
 old=con.execute('SELECT configuration_hash FROM horizon_runs WHERE id=?',(run,)).fetchone()
 if old and old[0]!=config:raise RuntimeError('Immutable horizon configuration changed: '+run)
 with con:
  con.execute('INSERT OR IGNORE INTO horizon_runs VALUES(?,?,?,?,?)',(run,route,target,config,dump(meta)))
  con.executemany('INSERT OR IGNORE INTO horizon_ticks VALUES(?,?,?,?)',[(run,t['day'],t['minute'],dump(t)) for t in sim['ticks']])
  con.executemany('INSERT OR IGNORE INTO horizon_events VALUES(?,?,?,?,?,?,?)',[(run,e['day'],e['symbol'],e['minute'],e['score'],e['threshold'],dump(e)) for e in sim['events']])
  con.executemany('INSERT OR IGNORE INTO horizon_labels VALUES(?,?,?,?,?,?)',[(run,l['day'],l['symbol'],l['hit'],l['future_status'],dump(l)) for l in sim['labels']])
 con.close()

def block_interval(sim):
 lookup={(l['day'],l['symbol']):l for l in sim['labels']};ds=sorted({t['day'] for t in sim['ticks']});weeks=sorted({str((pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in ds});counts=np.zeros((len(weeks),2))
 for e in sim['events']:
  hit=lookup[(e['day'],e['symbol'])]['hit']
  if hit is None:continue
  week=str((pd.Timestamp(e['day'])-pd.Timedelta(days=pd.Timestamp(e['day']).weekday())).date());i=weeks.index(week);counts[i]+=[hit,1]
 if not counts[:,1].sum():return [None,None]
 # Two adjacent week blocks accommodate overlapping ten-session targets.
 rng=np.random.default_rng(1004);values=[]
 for _ in range(2000):
  starts=rng.integers(0,len(weeks),size=(len(weeks)+1)//2);indices=np.asarray([[i,(i+1)%len(weeks)] for i in starts]).ravel()[:len(weeks)];s=counts[indices].sum(axis=0)
  if s[1]:values.append(s[0]/s[1])
 return np.quantile(values,[.025,.975]).tolist()

def aggregate():
 favorites={r['code'][3:] for r in json.loads((OUT/'universe.json').read_text())['favorites'] if r['code'].startswith('US.')};report=dict(version='horizon_v1',start='2026-05-01',end=CUTOFF,outcome_cutoff=CUTOFF,status='exposed_historical_forward_development',protocol_sha256=sha(OUT/'HORIZON_PROTOCOL.md'),targets=[])
 for target,cfg in TARGETS.items():
  head=dict(target_id=target,**cfg,routes=[],weeks=[]);sims={}
  for route in ROUTES:
   dest=DEST/target;metas=[];sim=dict(events=[],labels=[],ticks=[])
   for month in MONTHS:
    part=json.loads((dest/f'{route}_{month}_replay.json').read_text());meta=json.loads((dest/f'{route}_{month}_meta.json').read_text());metas.append(meta)
    for k in sim:sim[k]+=part[k]
   sim['metrics']=metrics(sim['events'],sim['labels'],sim['ticks']);m=sim['metrics'];m['block_ci']=block_interval(sim);m['mean_signal_probability']=float(np.mean([e['score'] for e in sim['events']])) if sim['events'] else None;m['favorite_passed']=[s for s in m['stocks'] if s['symbol'] in favorites and s['passed']]
   # Exact stock-day weighted scores: aggregate month means by their weights.
   ps=[x['probability_evaluation'] for x in metas];total=sum(p['stock_days'] for p in ps);prob={k:sum(p[k]*p['stock_days'] for p in ps if p[k] is not None)/total if total else None for k in ['brier','baseline_brier','predicted','observed']};prob['stock_days']=total
   head['routes'].append(dict(route=route,name=NAMES[route],algorithm=ALGORITHMS[route],metrics=m,months=metas,passed=gate(m),probability_evaluation=prob));sims[route]=sim;write(dest/f'{route}_replay.json',sim);persist(f'horizon_v1:{target}:{route}:2026-05_09',route,target,dict(protocol_sha256=report['protocol_sha256'],months=[{k:x[k] for k in ['month','thresholds','model_sha256','train_end','selection_end']} for x in metas]),sim)
  days=[d for d in DATES if report['start']<=d<=CUTOFF];weeks=sorted({str((pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in days})
  for week in weeks:
   last=str((pd.Timestamp(week)+pd.Timedelta(days=6)).date());ds=[d for d in days if week<=d<=last];row=dict(week_start=week,first_session=ds[0],last_session=ds[-1],sessions=len(ds),routes={})
   for route,sim in sims.items():
    parts={k:[r for r in sim[k] if week<=r['day']<=last] for k in ['events','labels','ticks']};row['routes'][route]=metrics(**parts)
   head['weeks'].append(row)
  report['targets'].append(head)
 write(DEST/'results.json',report)
 with (DEST/'weekly.csv').open('w',newline='') as file:
  writer=csv.DictWriter(file,lineterminator='\n',fieldnames=['target','week_start','first_session','last_session','route','algorithm','signals','tp','fp','pending','precision','baseline','lift']);writer.writeheader()
  for h in report['targets']:
   for w in h['weeks']:
    for r,m in w['routes'].items():writer.writerow(dict(target=h['target_id'],week_start=w['week_start'],first_session=w['first_session'],last_session=w['last_session'],route=r,algorithm=ALGORITHMS[r],signals=m['n'],tp=m['tp'],fp=m['fp'],pending=m['pending'],precision=m['precision'],baseline=m['baseline'],lift=m['lift']))
 report_markdown(report)
 print(dump(dict(summary=[dict(target=h['target_id'],routes=[dict(route=r['route'],algorithm=r['algorithm'],**{k:r['metrics'][k] for k in ['n','tp','fp','pending','precision','baseline','lift']}) for r in h['routes']]) for h in report['targets']])),flush=True)

def report_markdown(report):
 def pct(x):return '—' if x is None else f'{x:.2%}'
 lines=['# 跨日目标 · 三条路线与算法','',f'2026 年 5–9 月历史前向开发回测；结果行情截止 {CUTOFF}。买入当日为第 1 个交易日，延迟后的 5m Open×1.001 为评价价，只统计买入后的常规盘 High。有效性在看到标签前确定，未知保留。不同模型、目标与重叠日期不能合并当独立证据。','']
 for head in report['targets']:
  lines += ['## '+head['name'],'','| 路线 / 算法 | 有效 | 达成 | 失败 | 待成熟/缺失 | 达成率 | 平均预测概率 | 同股基准 | 增量 | 两周块 95% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
  for r in head['routes']:
   m=r['metrics'];lines.append(f"| {r['name']} / {r['algorithm']} | {m['n']} | {m['tp']} | {m['fp']} | {m['pending']} | {pct(m['precision'])} | {pct(m['mean_signal_probability'])} | {pct(m['baseline'])} | {pct(m['lift'])} | {pct(m['block_ci'][0])}–{pct(m['block_ci'][1])} |")
  lines+=['','### 周粒度（命中 / 成熟信号；待成熟另列）','','| ET 交易日 | 基础风险 / LogisticRegression | 相关股 / LightGBM | 恢复 / ExtraTrees |','|---|---|---|---|']
  for w in head['weeks']:
   cells=[]
   for route in ROUTES:
    m=w['routes'][route];cell=f"{m['tp']}/{m['mature']}（{pct(m['precision'])}）" if m['mature'] else '—'
    if m['pending']:cell+=f"；待 {m['pending']}"
    cells.append(cell)
   lines.append('| '+w['first_session'][5:]+'–'+w['last_session'][5:]+' | '+' | '.join(cells)+' |')
  lines+=['','### 全部预测的概率检查（股票日等权）','','| 路线 / 算法 | 股票日 | Brier | 基准 Brier | 预测均值 | 实际触及率 |','|---|---:|---:|---:|---:|---:|']
  for r in head['routes']:
   p=r['probability_evaluation'];lines.append(f"| {r['name']} / {r['algorithm']} | {p['stock_days']} | {p['brier']:.5f} | {p['baseline_brier']:.5f} | {pct(p['predicted'])} | {pct(p['observed'])} |")
  lines+=['','### 特别关注的历史逐股子集（评价后发现，不赋予未来资格）','','| 路线 / 算法 | 股票 | 达成 / 有效 | 达成率 | 同股基准 | 增量 |','|---|---|---:|---:|---:|---:|']
  for r in head['routes']:
   for s in r['metrics']['favorite_passed']:lines.append(f"| {r['name']} / {r['algorithm']} | {s['symbol']} | {s['tp']}/{s['n']} | {pct(s['precision'])} | {pct(s['baseline'])} | {pct(s['lift'])} |")
 old_path=DEST/'old_signals_followup.json'
 if old_path.exists():
  old=json.loads(old_path.read_text());lines+=['','## 原当天 +3% 有效信号的长窗口后续结果','','这些是固定旧信号的事后后续结果；旧概率/门槛只针对当天+3%，没有针对新目标校准。不能作为新目标当时预测概率；不沿用旧+3%基准做跨日增量。','','| 原路线 / 算法 | 5日+5% 达成 / 成熟 | 达成率 | 待成熟/未知 | 10日+10% 达成 / 成熟 | 达成率 | 待成熟/未知 |','|---|---:|---:|---:|---:|---:|---:|']
  for r in old['routes']:
   a=r['targets']['5d5pct']['metrics'];b=r['targets']['10d10pct']['metrics'];lines.append(f"| {r['name']} / {r['algorithm']} | {a['tp']}/{a['mature']} | {pct(a['precision'])} | {a['pending']} | {b['tp']}/{b['mature']} | {pct(b['precision'])} | {b['pending']} |")
 lines+=['','## 解释与限制','','门槛取自月前选择块，没有合格门槛则不发信号。完整窗口才计分，待成熟/缺失仍在全部有效信号分母中，不能只保留提前触及的成功。基准是同股同分钟的月前成熟历史，长窗口达成率高不自动意味着模型有增量。','', '本表评价信号后的触及概率，未模拟跨日持仓资金占用和限价成交；旧十月自动运行与 +3% 模型保持原冻结配置。当前成员回溯不是历史 PIT；历史暴露、重叠窗口及可能的标的集中限制未来外推。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')

def old_signals():
 result=dict(version='old_3pct_signals_followup_v1',outcome_cutoff=CUTOFF,routes=[]);cache={}
 for route in ROUTES:
  sim=json.loads((OUT/'weekly_v1'/f'{route}_replay.json').read_text());row=dict(route=route,name=NAMES[route],algorithm=ALGORITHMS[route],signals=len(sim['events']),targets={})
  for target in TARGETS:
   ls=[]
   for e in sim['events']:
    s=e['symbol']
    if s not in cache:cache[s]=read_raw(s)
    f=pd.DataFrame([dict(day=e['day'],minute=e['minute'])]);last=window_end(e['day'],target)[:10];subset=cache[s][(cache[s].day>=e['day'])&(cache[s].day<=last)];l=label_rows(subset,f,target,action_days(s)).iloc[0];ls.append(dict(day=e['day'],symbol=s,hit=int(l.y) if pd.notna(l.y) else None,**{k:clean(l[k]) for k in ['entry','target','label_end','future_status']}))
   m=metrics(sim['events'],ls,sim['ticks']);m['baseline']=None;m['lift']=None;m['definition']='Follow-up outcomes of old +3% signals; old probabilities/thresholds are not calibrated for this target'
   for stock in m['stocks']:stock.update(baseline=None,lift=None,passed=False)
   row['targets'][target]=dict(metrics=m,labels=ls)
  result['routes'].append(row)
 write(DEST/'old_signals_followup.json',result)
 if (DEST/'results.json').exists():report_markdown(json.loads((DEST/'results.json').read_text()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--fit',action='store_true');ap.add_argument('--october',action='store_true');ap.add_argument('--aggregate',action='store_true');ap.add_argument('--old-signals',action='store_true');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--target',choices=list(TARGETS));a=ap.parse_args()
 if a.prepare:prepare()
 if a.fit:
  for target in ([a.target] if a.target else TARGETS):
   with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(target,)) as pool:
    for f in as_completed([pool.submit(fit,(r,m)) for m in (['2026-10'] if a.october else MONTHS) for r in ROUTES]):f.result()
 if a.aggregate:aggregate()
 if a.old_signals:old_signals()
if __name__=='__main__':main()
