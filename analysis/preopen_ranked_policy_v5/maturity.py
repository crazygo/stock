"""Score prospective forecasts after their remaining RTH window is complete.

These are forecast labels, never purchases or fills. Missing windows stay pending.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from common import *
from store import connect

def outcome(raw,day,entry_minute,now):
 close=570+SESSIONS[day]['duration_minutes'];end=pd.Timestamp(day)+pd.Timedelta(minutes=close)
 if pd.Timestamp(now).tz_localize(None)<end+pd.Timedelta(seconds=1):return None
 g=raw[raw.day==day].sort_values('start');bar=g[g.minute==entry_minute]
 if len(bar)!=1:return None
 remaining=g[(g.minute>=max(570,entry_minute))&(g.minute<close)]
 if not np.array_equal(remaining.minute.to_numpy(),np.arange(max(570,entry_minute),close,5)):return None
 entry=float(bar.open.iloc[0]*1.001);target=entry*1.03;hits=remaining[remaining.high>=target];hit=int(len(hits)>0)
 exit=(target if hit else float(remaining.close.iloc[-1]))*.999;exit_minute=int(hits.minute.iloc[0]+5) if hit else close
 path=g[(g.minute>=entry_minute)&(g.minute<exit_minute)]
 complete_path=np.array_equal(path.minute.to_numpy(),np.arange(entry_minute,exit_minute,5))
 return dict(entry=entry,target=target,hit=hit,exit=exit,net=exit/entry-1,mae=float(path.low.min()/entry-1) if complete_path else None,label_end=str(end),entry_at=str(pd.Timestamp(day)+pd.Timedelta(minutes=entry_minute)))

def settle(cached_raw,now=None):
 now=now or datetime.now(ET);con=connect();dep=json.loads((OUT/'phase_deployment.json').read_text());added=0
 closed_days=[d for d in SESSIONS if pd.Timestamp(d)+pd.Timedelta(minutes=570+SESSIONS[d]['duration_minutes'],seconds=1)<=pd.Timestamp(now).tz_localize(None)]
 last_closed=max(closed_days)
 pending=con.execute('''SELECT p.* FROM observation_predictions p
 LEFT JOIN observation_outcomes x ON x.observation_id=p.observation_id AND x.symbol=p.symbol
 WHERE x.observation_id IS NULL AND p.day>=? AND p.day<=? ORDER BY p.day,p.symbol,p.minute''',(dep['observe_from'],last_closed)).fetchall()
 records=[];day_cache={}
 for row in pending:
  day=row['day'];key=(day,row['symbol'])
  if key not in day_cache:
   f=cached_raw(row['symbol']);day_cache[key]=f[f.day==day]
  result=outcome(day_cache[key],day,row['minute']+5,now)
  if result is None:continue
  records.append((row['observation_id'],row['symbol'],row['phase'],result['entry_at'],result['entry'],result['target'],result['label_end'],result['hit'],result['exit'],result['net'],result['mae'],now.isoformat()))
 with con:con.executemany('INSERT OR IGNORE INTO observation_outcomes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',records)
 added+=len(records)
 con.close();return added

def summary(con):
 total=con.execute('SELECT count(*) FROM observation_predictions').fetchone()[0]
 rows=con.execute('''SELECT phase,sum(n) n,count(*) stock_days,avg(brier) brier FROM (
 SELECT p.phase,p.symbol,p.day,count(*) n,avg((p.score-x.hit)*(p.score-x.hit)) brier
 FROM observation_predictions p JOIN observation_outcomes x ON x.observation_id=p.observation_id AND x.symbol=p.symbol
 WHERE x.hit IS NOT NULL GROUP BY p.phase,p.symbol,p.day) GROUP BY phase''').fetchall()
 phases={p:dict(n=0,stock_days=0,brier=None) for p in ['pre','regular']}
 for r in rows:phases[r['phase']]={k:r[k] for k in ['n','stock_days','brier']}
 mature=sum(r['n'] for r in phases.values())
 return dict(prediction_rows=total,mature=mature,pending=total-mature,phases=phases,definition='All stored prospective estimates, stock-day equal-weight Brier; rows are not independent trades')

def stats(rows):
 records=[dict(r) for r in rows];total=len(records);mature=[r for r in records if r['hit'] is not None];out={}
 for phase in ['pre','regular']:
  rs=[r for r in mature if r['phase']==phase]
  if not rs:out[phase]=dict(n=0,stock_days=0,brier=None);continue
  f=pd.DataFrame(rs);f['error']=(f.score-f.hit)**2;out[phase]=dict(n=len(f),stock_days=len(f.groupby(['symbol','day'])),brier=float(f.groupby(['symbol','day']).error.mean().mean()))
 return dict(prediction_rows=total,mature=len(mature),pending=total-len(mature),phases=out,definition='All stored prospective estimates, stock-day equal-weight Brier; rows are not independent trades')
