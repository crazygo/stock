"""Loopback research dashboard, live acquisition and idempotent paper ledger."""
from __future__ import annotations
import argparse,csv,io,json,pickle,threading,time,traceback
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
import pandas as pd
from common import *
from store import connect,dump
from data import read_raw,prior_context,intraday,peers
from train import score

JOB={'running':False,'phase':'未刷新'};LOCK=threading.Lock();LAST=None;RAW_CACHE={};MODEL_CACHE={}
def cached_raw(s):
 p=OUT/'raw'/f'{s}.parquet';stamp=p.stat().st_mtime if p.exists() else 0
 if s not in RAW_CACHE or RAW_CACHE[s][0]!=stamp:RAW_CACHE[s]=(stamp,read_raw(s))
 return RAW_CACHE[s][1]
def deployment():return json.loads((OUT/'deployment.json').read_text())
def model(route):
 p=OUT/'models'/f'{route}_2026-10.pkl'
 if route not in MODEL_CACHE:MODEL_CACHE[route]=pickle.loads(p.read_bytes())
 return MODEL_CACHE[route]
def baseline_for(a,f):return [a['baseline_table'].get(f'{s}|{int(m)}',a['baseline_minute'].get(int(m),.1)) for s,m in zip(f.symbol,f.minute)]
def market_state(now):
 day=now.date().isoformat();minute=now.hour*60+now.minute
 if day not in SESSIONS:return 'closed'
 close=570+SESSIONS[day]['duration_minutes']
 if minute<240 or minute>=1200:return 'night'
 if minute<570:return 'pre'
 if minute<close:return 'regular'
 return 'post'
def snapshot(refetch=False):
 global LAST
 from acquire import refresh
 if refetch:
  JOB.update(phase='检查并补齐最新数据');refresh(live=True,status=JOB);RAW_CACHE.clear();settle_live()
 now=datetime.now(ET);state=market_state(now);dep=deployment();results=json.loads((OUT/'results.json').read_text());route=dep['route'] or max(results['routes'],key=lambda r:(r['metrics']['precision'] or 0,r['metrics']['n']))['route'];a=model(route)
 day=now.date().isoformat() if state=='regular' else max(d for d in SESSIONS if d<now.date().isoformat() or (d==now.date().isoformat() and state=='post'))
 cut=min((now.hour*60+now.minute)//5*5,570+SESSIONS[day]['duration_minutes']-30) if state=='regular' else 570+SESSIONS[day]['duration_minutes']-30
 frames=[];missing=[];u=json.loads((OUT/'universe.json').read_text());allowed=set(u['members'])
 for s in a['registered']:
  if s not in allowed and s not in ['QQQ','SOXX','IGV']:continue
  f=cached_raw(s);tail=f[f.start>=pd.Timestamp(day)-pd.Timedelta(days=100)]
  ctx,_=prior_context(tail,s);p=intraday(tail[tail.end<=pd.Timestamp(day)+pd.Timedelta(minutes=cut)],ctx,s,labels=False,first=day,last=day)
  if len(p) and cut in p.minute.to_numpy():frames.append(p[p.minute==cut])
  else:missing.append(s)
 if frames:
  f=peers(pd.concat(frames,ignore_index=True));f['stock_id']=f.symbol.map(a['stock_map']);f['score']=score(a,f);f['baseline']=baseline_for(a,f);f=f.sort_values(['score','symbol'],ascending=[False,True]);rows=[]
  threshold=a['threshold']
  for rank,r in enumerate(f.to_dict('records'),1):
   s=r['symbol'];m=u['members'].get(s,{});rows.append(dict(rank=rank,symbol=s,name=m.get('name',s),groups=m.get('groups',[]),favorite='特别关注' in m.get('groups',[]),score=r['score'],baseline=r['baseline'],lift=r['score']-r['baseline'],reference=r['reference'],target=r['reference']*1.001*1.03,threshold=threshold,qualified=bool(dep['admitted'] and threshold is not None and r['score']>=threshold and not r['ex_action']),feature_available=r['feature_available'],last_volume=r['last_volume'],feature_row=r))
 else:rows=[]
 active=state=='regular' and 575<=now.hour*60+now.minute<=570+SESSIONS[day]['duration_minutes']-30 and day>=dep['observe_from'] and day<=dep['observe_through']
 reason='market_closed' if not active else 'model_not_admitted' if not dep['admitted'] else 'no_qualified_candidate';action='no_action'
 qualified=[r for r in rows if r['qualified']]
 # Model probability refers to a fixed next-bar entry, not a stale day's old target.
 if state!='regular':
  for r in rows:r['qualified']=False
 if active and dep['admitted'] and qualified:
  top=qualified[0];reason,action=paper_decide(top,now,day,cut,a,dep)
 payload=dict(observed_at=datetime.now(ET).isoformat(),market_state=state,reference_day=day,feature_cutoff=f'{day}T{cut//60:02d}:{cut%60:02d}:00-04:00',entry_at=f'{day}T{(cut+5)//60:02d}:{(cut+5)%60:02d}:00-04:00',action=action,reason=reason,route=route,admitted=dep['admitted'],frozen_at=dep['frozen_at'],selected_symbol=qualified[0]['symbol'] if action=='buy_pending' else None,rows=rows,missing=missing,current_probability=state=='regular',probability_definition='next_5m_open_plus_10bp_then_same_day_touch_3pct',winning_condition='实际模拟买入价起，当天剩余常规盘触及 +3%；未触及日终退出',membership_at=u['watchlist_observed_at'],acquisition=json.loads((OUT/'acquisition.json').read_text()) if (OUT/'acquisition.json').exists() else None)
 key=f"{sha(OUT/'deployment.json')}:{day}:{cut}:{state}";con=connect()
 with con:con.execute('INSERT OR IGNORE INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',(now.isoformat(),key,state,action,reason,dump(payload)))
 con.close();LAST=clean(payload);write(OUT/'latest.json',LAST);JOB.update(phase='刷新完成',running=False);return LAST

def paper_decide(top,now,day,cut,a,dep):
 # A new decision may reserve one entry. Quotes validate availability, not outcome.
 con=connect();pending=con.execute('SELECT payload FROM observations WHERE action="buy_pending"').fetchall();fills=con.execute('SELECT * FROM live_fills').fetchall();filledids={f['observation_id'] for f in fills}
 for r in pending:
  p=json.loads(r['payload'])
  # A previous pending entry is resolved by settle_live; do not duplicate it.
  if p['reference_day']==day and p.get('selected_symbol')==top['symbol']:con.close();return 'already_bought_today','no_action'
 openfills=[f for f in fills if f['exit'] is None]
 unresolved=con.execute('SELECT id FROM observations WHERE action="buy_pending"').fetchall()
 if openfills or any(r['id'] not in filledids for r in unresolved):con.close();return 'position_or_entry_pending','no_action'
 equity=100000.+sum((f['exit']-f['entry'])*f['qty'] for f in fills if f['exit'] is not None);cash=100000.
 for f in fills:
  cash-=f['entry']*f['qty']
  if f['exit'] is not None and f['exit_at'][:10]<day:cash+=f['exit']*f['qty']
 con.close();qty=int(min(cash,equity*.2)/(top['reference']*1.001))
 if qty<1:return 'settled_cash_insufficient','no_action'
 if qty>top['last_volume']*.01:return 'top1_liquidity_rejected','no_action'
 now=datetime.now(ET)
 if now.replace(tzinfo=None)>=pd.Timestamp(day)+pd.Timedelta(minutes=cut+5):return 'entry_deadline_missed','no_action'
 import futu as ft
 ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
 try:
  ret,_=q.subscribe(['US.'+top['symbol']],[ft.SubType.QUOTE,ft.SubType.ORDER_BOOK],subscribe_push=False)
  if ret!=ft.RET_OK:return 'quote_subscription_missing','no_action'
  ret,quote=q.get_stock_quote(['US.'+top['symbol']]);rb,book=q.get_order_book('US.'+top['symbol'],num=1)
  if ret!=ft.RET_OK or rb!=ft.RET_OK or not book.get('Ask') or not book.get('Bid'):return 'bid_ask_missing','no_action'
  stamp=pd.Timestamp(str(quote.data_date.iloc[0])+' '+str(quote.data_time.iloc[0])).tz_localize(ET);age=(datetime.now(ET)-stamp.to_pydatetime()).total_seconds();ask=float(book['Ask'][0][0]);bid=float(book['Bid'][0][0])
  if age<0 or age>15 or bid<=0 or ask<bid:return 'quote_stale','no_action'
  if ask/bid-1>.003 or abs(ask/(top['reference']*1.001)-1)>.005:return 'spread_or_price_deviation','no_action'
  top.update(ask=ask,bid=bid,quote_at=stamp.isoformat(),qty=qty,budget=min(cash,equity*.2))
  return 'rank1_qualified','buy_pending'
 finally:q.close()

def settle_live():
 con=connect();now=datetime.now(ET)
 with con:
  pending=con.execute('SELECT id,payload FROM observations WHERE action="buy_pending"').fetchall()
  for r in pending:
   p=json.loads(r['payload']);s=p.get('selected_symbol');top=next((x for x in p['rows'] if x['symbol']==s),None)
   if top is None:continue
   f=cached_raw(s);entry_at=pd.Timestamp(p['entry_at']).tz_localize(None);bar=f[f.start==entry_at]
   if not len(bar) or bar.end.iloc[0]>pd.Timestamp(now.replace(tzinfo=None)):continue
   existing=con.execute('SELECT 1 FROM live_fills WHERE observation_id=?',(r['id'],)).fetchone()
   if not existing:
    entry=float(bar.open.iloc[0]*1.001);qty=min(top['qty'],int(top['budget']/entry))
    if qty<1:continue
    con.execute('INSERT INTO live_fills(observation_id,symbol,entry_at,entry,target,qty,payload) VALUES(?,?,?,?,?,?,?)',(r['id'],s,p['entry_at'],entry,entry*1.03,qty,dump(dict(source='next_5m_open_proxy',reference_probability=top['score']))))
  for pos in con.execute('SELECT * FROM live_fills WHERE exit IS NULL').fetchall():
   day=pos['entry_at'][:10];close=pd.Timestamp(day)+pd.Timedelta(minutes=570+SESSIONS[day]['duration_minutes']);f=cached_raw(pos['symbol']);g=f[(f.start>=pd.Timestamp(pos['entry_at']).tz_localize(None))&(f.end<=close)&(f.end<=pd.Timestamp(now.replace(tzinfo=None)))];hits=g[g.high>=pos['target']]
   if len(hits):exit=float(pos['target']*.999);stamp=str(hits.end.iloc[0]);hit=1
   elif len(g) and g.end.max()==close:exit=float(g.close.iloc[-1]*.999);stamp=str(close);hit=0
   else:continue
   con.execute('UPDATE live_fills SET exit_at=?,exit=?,hit=?,net=? WHERE observation_id=?',(stamp,exit,hit,exit/pos['entry']-1,pos['observation_id']))
 con.close()

def refresh_job():
 if not LOCK.acquire(False):return
 JOB.update(running=True,phase='检查最新数据',error=None)
 try:snapshot(True);settle_live()
 except Exception as e:JOB.update(running=False,phase='刷新失败',error=type(e).__name__+': '+str(e)[:240]);traceback.print_exc()
 finally:LOCK.release()

def chart(s,start,end):
 if s not in json.loads((OUT/'coverage.json').read_text())['coverage_symbols']:raise ValueError('symbol not in registered chart coverage')
 f=cached_raw(s);associated=(f.start+pd.to_timedelta((f.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d');g=f[(associated>=start)&(associated<=end)&associated.isin(SESSIONS)];bars=[[r.start.strftime('%Y-%m-%dT%H:%M'),r.end.strftime('%Y-%m-%dT%H:%M'),float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume)] for r in g.itertuples()]
 daily=json.loads((OUT/'daily.json').read_text()).get(s,[])
 for day,r in f[(f.day>'2026-10-02')&(f.minute>=570)&(f.minute<960)].groupby('day'):
  if day in SESSIONS:
   close=570+SESSIONS[day]['duration_minutes'];r=r[r.minute<close]
   if len(r):daily.append(dict(day=day,o=float(r.open.iloc[0]),h=float(r.high.max()),l=float(r.low.min()),c=float(r.close.iloc[-1]),v=float(r.volume.sum()),partial=len(r)<SESSIONS[day]['duration_minutes']//5))
 latest=max(['2026-10-02']+[d['day'] for d in daily]);first=str((pd.Timestamp(latest)-pd.DateOffset(years=2)+pd.Timedelta(days=1)).date());allcal={d:x['duration_minutes'] for d,x in SESSIONS.items()};return dict(symbol=s,bars=bars,daily=daily,calendar=allcal,mini_calendar={d:n for d,n in allcal.items() if first<=d<=latest},grain_minutes=5,full_day=True)

class Handler(BaseHTTPRequestHandler):
 def send(self,x,status=200,mime='application/json'):
  raw=dump(x).encode() if mime=='application/json' else x.encode();self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  p=urlparse(self.path);q={k:v[0] for k,v in parse_qs(p.query).items()}
  try:
   if p.path in ['/','/index.html','/hazard_linear.html','/relative_flow.html','/recovery_forest.html']:return self.send((OUT/'index.html').read_text(),mime='text/html')
   if p.path=='/api/status':return self.send(dict(job=JOB,latest=LAST or (json.loads((OUT/'latest.json').read_text()) if (OUT/'latest.json').exists() else None),deployment=deployment(),results=json.loads((OUT/'results.json').read_text()),coverage=json.loads((OUT/'coverage.json').read_text())))
   if p.path=='/api/chart':return self.send(chart(q['symbol'],q.get('start','2026-08-01'),q.get('end','2026-09-30')))
   con=connect()
   try:
    if p.path=='/api/runs':return self.send([dict(r) for r in con.execute('SELECT * FROM runs ORDER BY id')])
    if p.path=='/api/ledger':
     run=q.get('run','v4:relative_flow:2026-08_09');rows=[dict(r) for r in con.execute('SELECT t.*,o.hit,o.net,o.mae,o.exit,o.exit_minute FROM trades t LEFT JOIN outcomes o ON o.run_id=t.run_id AND o.trade_id=t.id WHERE t.run_id=? AND t.day>=? AND t.day<=? ORDER BY t.day,t.decision_minute',(run,q.get('start','2026-08-01'),q.get('end','2026-09-30')))];return self.send(rows)
    if p.path=='/api/decision':
     run=q['run'];day=q['day'];minute=int(q['minute']);d=con.execute('SELECT * FROM decisions WHERE run_id=? AND day=? AND minute=?',(run,day,minute)).fetchone();c=[dict(x) for x in con.execute('SELECT * FROM candidates WHERE run_id=? AND day=? AND minute=? ORDER BY rank LIMIT 20',(run,day,minute))];return self.send(dict(decision=dict(d) if d else None,candidates=c))
    if p.path=='/api/decisions':return self.send([dict(r) for r in con.execute('SELECT day,minute,action,reason,top_symbol,top_score,qualified FROM decisions WHERE run_id=? AND day>=? AND day<=? ORDER BY day,minute LIMIT 5000',(q['run'],q.get('start','2026-08-01'),q.get('end','2026-09-30')))])
    if p.path=='/api/live':return self.send(dict(observations=[dict(r) for r in con.execute('SELECT * FROM observations ORDER BY id DESC LIMIT 100')],fills=[dict(r) for r in con.execute('SELECT * FROM live_fills ORDER BY observation_id DESC')]))
   finally:con.close()
   return self.send({'error':'not_found'},404)
  except Exception as e:traceback.print_exc();self.send({'error':type(e).__name__+': '+str(e)[:240]},400)
 def do_POST(self):
  if self.path!='/api/refresh':return self.send({'error':'not_found'},404)
  threading.Thread(target=refresh_job,daemon=True).start();self.send(dict(accepted=True,job=JOB),202)
 def log_message(self,fmt,*args):pass

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8770);ap.add_argument('--once',action='store_true');a=ap.parse_args()
 if a.once:snapshot(True);settle_live();return
 print('http://127.0.0.1:'+str(a.port),flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
