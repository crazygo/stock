"""Loopback research dashboard, live acquisition and idempotent paper ledger."""
from __future__ import annotations
import argparse,csv,io,json,pickle,threading,time,traceback
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
import pandas as pd
from common import *
from store import connect,dump,record_predictions
from data import read_raw,prior_context,intraday,peers
from train import score
from phase import predict as phase_predict
from maturity import settle as settle_forecasts,summary as forecast_summary
from signals import settle as settle_signals
from signal_runtime import deployment as signal_deployment,evaluate as evaluate_signal,observation_summary as signal_summary

SCHEDULE={'interval_minutes':5,'decision_offset_seconds':30,'enabled':True};JOB={'running':False,'phase':'未刷新'};LOCK=threading.Lock();LAST=None;RAW_CACHE={};MODEL_CACHE={}
def cached_raw(s,full=False):
 p=OUT/'raw'/f'{s}.parquet';stamp=p.stat().st_mtime if p.exists() else 0
 key=(s,full)
 if key not in RAW_CACHE or RAW_CACHE[key][0]!=stamp:
  f=read_raw(s) if full or not p.exists() else pd.read_parquet(p,filters=[('end','>=',pd.Timestamp(datetime.now(ET).date())-pd.Timedelta(days=110))])
  RAW_CACHE[key]=(stamp,f)
  full_keys=[k for k in RAW_CACHE if k[1]]
  if len(full_keys)>3:RAW_CACHE.pop(full_keys[0])
 return RAW_CACHE[key][1]
def deployment():return json.loads((OUT/'phase_deployment.json').read_text())
def model(route):
 p=OUT/'models'/f'phase_{route}_2026-10.pkl'
 expected=next(m['artifact_sha256'] for m in deployment()['all_routes'] if m['route']==route)
 if sha(p)!=expected:raise RuntimeError('Frozen model artifact changed: '+route)
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
  JOB.update(phase='检查并补齐最新数据');refresh(live=True,status=JOB);RAW_CACHE.clear();settle_live();settle_forecasts(cached_raw);settle_signals(cached_raw)
 now=datetime.now(ET);state=market_state(now);dep=deployment();sdep=signal_deployment();results=json.loads((OUT/'phase_results.json').read_text());route=sdep['route'] or dep['route'] or max(results['routes'],key=lambda r:(r['metrics']['precision'] or 0,r['metrics']['n']))['route'];a=model(route)
 day=now.date().isoformat() if state in ['pre','regular'] else max(d for d in SESSIONS if d<now.date().isoformat() or (d==now.date().isoformat() and state=='post'))
 cut=min((now.hour*60+now.minute)//5*5,570+SESSIONS[day]['duration_minutes']-30) if state in ['pre','regular'] else 570+SESSIONS[day]['duration_minutes']-30
 decision_due=now.replace(tzinfo=None)>=pd.Timestamp(day)+pd.Timedelta(minutes=cut,seconds=30)
 evaluated_minute=(now.hour*60+now.minute)//5*5 if state in ['pre','regular'] else cut
 key=f"{sha(OUT/'phase_deployment.json')}:{sha(OUT/'signals_deployment.json')}:{day}:{evaluated_minute}:{state}:{int(decision_due)}"
 con=connect();existing=con.execute('SELECT payload FROM observations WHERE decision_key=?',(key,)).fetchone();con.close()
 in_window=state in ['pre','regular'] and 245<=now.hour*60+now.minute<=570+SESSIONS[day]['duration_minutes']-30 and dep['observe_from']<=day<=dep['observe_through']
 if existing and in_window:
  LAST=json.loads(existing['payload']);LAST['refresh_checked_at']=datetime.now(ET).isoformat();LAST['acquisition']=json.loads((OUT/'acquisition.json').read_text()) if (OUT/'acquisition.json').exists() else None;write(OUT/'latest.json',LAST);JOB.update(phase='刷新完成',running=False);return LAST
 frames=[];missing=[];u=json.loads((OUT/'universe.json').read_text());allowed=set(u['members'])
 for s in a['registered']:
  f=cached_raw(s);tail=f[f.start>=pd.Timestamp(day)-pd.Timedelta(days=100)]
  ctx,_=prior_context(tail,s);p=intraday(tail[tail.end<=pd.Timestamp(day)+pd.Timedelta(minutes=cut)],ctx,s,labels=False,first=day,last=day,only_minutes={cut})
  if len(p) and cut in p.minute.to_numpy():frames.append(p[p.minute==cut])
  else:missing.append(s)
 if frames:
  f=peers(pd.concat(frames,ignore_index=True));f['stock_id']=f.symbol.map(a['stock_map'])
  for k in a['features']:
   if k!='symbol':f[k]=f[k].astype('float32')
  f['score']=phase_predict(a,f);f['baseline']=baseline_for(a,f);f=f.sort_values(['score','symbol'],ascending=[False,True]);rows=[]
  threshold=a['phase_thresholds']['pre' if cut<=570 else 'regular'];account=paper_account(now);budget=min(account['cash'],account['equity']*.2)
  for rank,r in enumerate(f.to_dict('records'),1):
   s=r['symbol'];m=u['members'].get(s,{});qty=int(budget/(r['reference']*1.001));liquid=qty>=1 and qty<=r['last_volume']*.01;rows.append(dict(rank=rank,symbol=s,name=m.get('name',s),groups=m.get('groups',[]),favorite='特别关注' in m.get('groups',[]),score=r['score'],baseline=r['baseline'],lift=r['score']-r['baseline'],reference=r['reference'],target=r['reference']*1.001*1.03,threshold=threshold,qualified=bool(dep['admitted'] and threshold is not None and r['score']>=threshold and not r['ex_action'] and liquid and s not in account['bought_today']),liquid=liquid,feature_available=r['feature_available'],last_volume=r['last_volume'],feature_row=r))
 else:rows=[]
 active=decision_due and state in ['pre','regular'] and 245<=now.hour*60+now.minute<=570+SESSIONS[day]['duration_minutes']-30 and day>=dep['observe_from'] and day<=dep['observe_through']
 reason='decision_time_not_reached' if state in ['pre','regular'] and not decision_due else 'remaining_window_too_short' if state=='regular' and evaluated_minute>570+SESSIONS[day]['duration_minutes']-30 else 'market_closed' if not active else 'model_not_admitted' if not dep['admitted'] else 'no_qualified_candidate';action='no_action'
 qualified=[r for r in rows if r['qualified']]
 # Model probability refers to a fixed next-bar entry, not a stale day's old target.
 if not active:
  for r in rows:r['qualified']=False
 if active and dep['admitted'] and qualified:
  top=qualified[0];reason,action=paper_decide(top,now,day,cut,a,dep)
 signal_now=datetime.now(ET);entry_deadline=pd.Timestamp(day)+pd.Timedelta(minutes=cut+5)
 before_entry=signal_now.replace(tzinfo=None)<entry_deadline
 signal=evaluate_signal(rows,day,cut,route,active and before_entry,sdep)
 for r in rows:
  r['signal_threshold']=signal['threshold'];r['valid_signal']=bool(signal['valid'] and signal['symbol']==r['symbol'])
 payload=dict(observed_at=datetime.now(ET).isoformat(),market_state=state,reference_day=day,feature_cutoff=f'{day}T{cut//60:02d}:{cut%60:02d}:00-04:00',entry_at=f'{day}T{(cut+5)//60:02d}:{(cut+5)%60:02d}:00-04:00',action=action,reason=reason,route=route,admitted=dep['admitted'],frozen_at=dep['frozen_at'],selected_symbol=qualified[0]['symbol'] if action=='buy_pending' else None,rows=rows,missing=missing,current_probability=in_window,probability_definition='pre_or_regular_next_5m_open_plus_10bp_then_RTH_touch_3pct',winning_condition='信号后下一根5m Open ×1.001为评价价，当天剩余常规盘触及该价+3%；无需发生买入',membership_at=u['watchlist_observed_at'],acquisition=json.loads((OUT/'acquisition.json').read_text()) if (OUT/'acquisition.json').exists() else None)
 payload['signal']=signal
 payload['current_probability']=bool(in_window and before_entry)
 con=connect()
 with con:
  con.execute('INSERT OR IGNORE INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',(now.isoformat(),key,state,action,reason,dump(payload)))
  if in_window and decision_due and before_entry:
   oid=con.execute('SELECT id FROM observations WHERE decision_key=?',(key,)).fetchone()[0];record_predictions(con,oid,rows,day,cut)
 con.close();LAST=clean(payload);write(OUT/'latest.json',LAST);JOB.update(phase='刷新完成',running=False);return LAST

def paper_account(now):
 day=now.date().isoformat();con=connect();fills=con.execute('SELECT * FROM live_fills').fetchall();pending=con.execute('SELECT id,payload FROM observations WHERE action="buy_pending"').fetchall();con.close();cash=100000.;equity=100000.;bought=set();filledids={f['observation_id'] for f in fills}
 for f in fills:
  cash-=f['entry']*f['qty']
  if f['entry_at'][:10]==day and f['qty']>0:bought.add(f['symbol'])
  if f['exit'] is not None:
   equity+=(f['exit']-f['entry'])*f['qty']
   if f['exit_at'][:10]<day:cash+=f['exit']*f['qty']
  else:
   raw=cached_raw(f['symbol']);past=raw[raw.end<=pd.Timestamp(now.replace(tzinfo=None))]
   if len(past):equity+=(float(past.close.iloc[-1])-f['entry'])*f['qty']
 for r in pending:
  p=json.loads(r['payload'])
  if r['id'] not in filledids and p['reference_day']==day:bought.add(p['selected_symbol'])
 return dict(cash=cash,equity=equity,bought_today=bought,pending=any(r['id'] not in filledids for r in pending),position=any(f['exit'] is None for f in fills))

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
  ret,_=q.subscribe(['US.'+top['symbol']],[ft.SubType.QUOTE,ft.SubType.ORDER_BOOK],subscribe_push=False,extended_time=True,session=ft.Session.ALL)
  if ret!=ft.RET_OK:return 'quote_subscription_missing','no_action'
  ret,quote=q.get_stock_quote(['US.'+top['symbol']]);rb,book=q.get_order_book('US.'+top['symbol'],num=1)
  if ret!=ft.RET_OK or rb!=ft.RET_OK or not book.get('Ask') or not book.get('Bid'):return 'bid_ask_missing','no_action'
  stamps=[pd.Timestamp(book[k]).tz_localize(ET) for k in ['svr_recv_time_bid','svr_recv_time_ask'] if book.get(k)]
  if len(stamps)!=2:return 'quote_stale','no_action'
  stamp=min(stamps);age=(datetime.now(ET)-stamp.to_pydatetime()).total_seconds();ask=float(book['Ask'][0][0]);bid=float(book['Bid'][0][0])
  if age<0 or age>15 or bid<=0 or ask<bid:return 'quote_stale','no_action'
  if ask/bid-1>.003 or abs(ask/(top['reference']*1.001)-1)>.005:return 'spread_or_price_deviation','no_action'
  top.update(ask=ask,bid=bid,quote_at=stamp.isoformat(),qty=qty,budget=min(cash,equity*.2))
  return 'rank1_qualified','buy_pending'
 finally:q.close()

def live_exit_event(raw,entry_at,target,now):
 day=entry_at[:10];begin=max(570,pd.Timestamp(entry_at).hour*60+pd.Timestamp(entry_at).minute);end=570+SESSIONS[day]['duration_minutes'];close=pd.Timestamp(day)+pd.Timedelta(minutes=end)
 g=raw[(raw.day==day)&(raw.minute>=begin)&(raw.minute<end)&(raw.end<=pd.Timestamp(now.replace(tzinfo=None)))].sort_values('minute')
 hits=g[g.high>=target]
 if len(hits):
  first=hits.iloc[0];prefix=g[g.minute<=first.minute]
  if not np.array_equal(prefix.minute.to_numpy(),np.arange(begin,int(first.minute)+5,5)):return None
  return dict(exit=float(target*.999),exit_at=str(first.end),hit=1)
 if not len(g) or g.end.max()!=close or not np.array_equal(g.minute.to_numpy(),np.arange(begin,end,5)):return None
 return dict(exit=float(g.close.iloc[-1]*.999),exit_at=str(close),hit=0)

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
   event=live_exit_event(cached_raw(pos['symbol']),pos['entry_at'],pos['target'],now)
   if event is None:continue
   con.execute('UPDATE live_fills SET exit_at=?,exit=?,hit=?,net=? WHERE observation_id=?',(event['exit_at'],event['exit'],event['hit'],event['exit']/pos['entry']-1,pos['observation_id']))
 con.close()

def refresh_job():
 if not LOCK.acquire(False):return
 JOB.update(running=True,phase='检查最新数据',error=None)
 try:snapshot(True);settle_live()
 except Exception as e:JOB.update(running=False,phase='刷新失败',error=type(e).__name__+': '+str(e)[:240]);traceback.print_exc()
 finally:LOCK.release()

def chart(s,start,end):
 if s not in json.loads((OUT/'coverage.json').read_text())['coverage_symbols']:raise ValueError('symbol not in registered chart coverage')
 f=cached_raw(s,full=True);associated=(f.start+pd.to_timedelta((f.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d');g=f[(associated>=start)&(associated<=end)&associated.isin(SESSIONS)];bars=[[r.start.strftime('%Y-%m-%dT%H:%M'),r.end.strftime('%Y-%m-%dT%H:%M'),float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume)] for r in g.itertuples()]
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
   if p.path in ['/horizon_v1/REPORT.md','/horizon_v1/weekly.csv']:return self.send((OUT/p.path[1:]).read_text(),mime='text/plain')
   if p.path=='/api/options':
    from recommend import run
    target=q.get('target','intraday3pct')
    if target not in ['intraday3pct','5d5pct','10d10pct']:raise ValueError('Unknown target')
    options=run(argparse.Namespace(target=target,data_dir=None,offline=True,futu_host=None,futu_port=11111,as_of=None,top=3,save_features=None,output=str(OUT/'recommendation_latest.json') if target=='intraday3pct' else None,record=None));return self.send(options)
   if p.path=='/api/horizons':return self.send(json.loads((OUT/'horizon_v1/results.json').read_text()) if (OUT/'horizon_v1/results.json').exists() else None)
   if p.path=='/api/status':return self.send(dict(job=JOB,schedule=SCHEDULE,latest=LAST or (json.loads((OUT/'latest.json').read_text()) if (OUT/'latest.json').exists() else None),weekly_results=json.loads((OUT/'weekly_v1/results.json').read_text()) if (OUT/'weekly_v1/results.json').exists() else None,recommendation=json.loads((OUT/'recommendation_latest.json' if (OUT/'recommendation_latest.json').exists() else OUT/'reference_recommendation.json').read_text()) if (OUT/'reference_recommendation.json').exists() else None,deployment=deployment(),signal_deployment=signal_deployment(),signal_results=json.loads((OUT/'signals_results.json').read_text()),results=json.loads((OUT/'phase_results.json').read_text()),capacity_results=json.loads((OUT/'capacity_results.json').read_text()),shared_phase_results=json.loads((OUT/'results.json').read_text()),coverage=json.loads((OUT/'coverage.json').read_text()),raw_coverage=json.loads((OUT/'raw_coverage_audit.json').read_text()),universe_coverage=json.loads((OUT/'universe_coverage.json').read_text())))
   if p.path=='/api/chart':return self.send(chart(q['symbol'],q.get('start','2026-08-01'),q.get('end','2026-09-30')))
   con=connect()
   try:
    if p.path in ['/api/horizon-events','/api/horizon-inspect']:
     target=q.get('target');route=q.get('route')
     if target not in ['5d5pct','10d10pct'] or route not in ROUTES:raise ValueError('Unknown horizon target/route')
     run=f'horizon_v1:{target}:{route}:2026-05_09';rows=[dict(r) for r in con.execute('SELECT e.*,l.hit,l.status,l.payload AS label_payload FROM horizon_events e LEFT JOIN horizon_labels l ON l.run_id=e.run_id AND l.day=e.day AND l.symbol=e.symbol WHERE e.run_id=? AND e.day>=? AND e.day<=? ORDER BY e.day,e.minute',(run,q.get('start','2026-05-01'),q.get('end','2026-09-30')))]
     for row in rows:row.update(json.loads(row.pop('label_payload')))
     if p.path=='/api/horizon-events':return self.send(rows)
     event=next((r for r in rows if r['day']==q.get('day') and r['symbol']==q.get('symbol')),None)
     if event is None:raise ValueError('Unknown horizon event')
     dest=OUT/'horizon_v1'/target;meta=json.loads((dest/f"{route}_{event['day'][:7]}_meta.json").read_text());cols=list(dict.fromkeys(['symbol','day','minute']+meta['features']));features=pd.read_parquet(OUT/'weekly_v1/panel.parquet',columns=cols,filters=[('symbol','==',event['symbol']),('day','==',event['day']),('minute','==',event['minute'])]);frame=pd.read_parquet(dest/f"{route}_{event['day'][:7]}.parquet",filters=[('day','==',event['day']),('minute','==',event['minute'])]);frame['symbol']=frame.symbol.astype(str)
     issued={r[0] for r in con.execute('SELECT symbol FROM horizon_events WHERE run_id=? AND day=? AND minute<?',(run,event['day'],event['minute']))};frame=frame.sort_values(['score','symbol'],ascending=[False,True]);frame['eligible']=(frame.score>=event['threshold'])&(frame.ex_action==0)&~frame.symbol.isin(issued);candidates=frame[['symbol','score','eligible']].head(20).to_dict('records');first_touch=None
     if event['hit']==1:
      raw=cached_raw(event['symbol'],full=True);start=pd.Timestamp(event['day'])+pd.Timedelta(minutes=event['minute']+5);finish=pd.Timestamp(event['label_end']);mask=(raw.start>=start)&(raw.end<=finish)&(raw.day<= '2026-09-30')&(raw.minute>=570)&(raw.minute<raw.day.map(lambda d:570+SESSIONS[d]['duration_minutes'] if d in SESSIONS else 0));hits=raw[mask&(raw.high>=event['target'])]
      if len(hits):first_touch=str(hits.end.iloc[0])
     return self.send(dict(event=event,algorithm=ALGORITHMS[route],features=features.iloc[0].to_dict(),candidates=candidates,first_touch=first_touch))
    if p.path=='/api/runs':return self.send([dict(r) for r in con.execute('SELECT * FROM runs ORDER BY id')])
    if p.path=='/api/signals':
     if q.get('route') not in ROUTES:raise ValueError('Unknown signal route')
     if q.get('period','2026-08_09') not in ['2026-08_09','2026-05_09']:raise ValueError('Unknown signal period')
     run=f"weekly_signal_v1:{q['route']}:2026-05_09" if q.get('period')=='2026-05_09' else f"signal_v1:{q['route']}:2026-08_09";return self.send([dict(r) for r in con.execute('SELECT e.*,l.hit,l.entry,l.target,l.exit_minute,l.exit,l.net,l.mae,l.label_end FROM signal_events e LEFT JOIN signal_labels l ON l.run_id=e.run_id AND l.day=e.day AND l.symbol=e.symbol WHERE e.run_id=? AND e.day>=? AND e.day<=? ORDER BY e.day,e.minute',(run,q.get('start','2026-05-01'),q.get('end','2026-09-30')))])
    if p.path=='/api/signal-tick':return self.send(dict(con.execute('SELECT * FROM signal_ticks WHERE run_id=? AND day=? AND minute=?',(q['run'],q['day'],int(q['minute']))).fetchone()))
    if p.path=='/api/ledger':
     run=q.get('run','v5phase:relative_flow:2026-08_09');rows=[dict(r) for r in con.execute('SELECT t.*,o.hit,o.net,o.mae,o.exit,o.exit_minute FROM trades t LEFT JOIN outcomes o ON o.run_id=t.run_id AND o.trade_id=t.id WHERE t.run_id=? AND t.day>=? AND t.day<=? ORDER BY t.day,t.decision_minute',(run,q.get('start','2026-08-01'),q.get('end','2026-09-30')))];return self.send(rows)
    if p.path=='/api/decision':
     run=q['run'];day=q['day'];minute=int(q['minute']);d=con.execute('SELECT * FROM decisions WHERE run_id=? AND day=? AND minute=?',(run,day,minute)).fetchone();c=[dict(x) for x in con.execute('SELECT * FROM candidates WHERE run_id=? AND day=? AND minute=? ORDER BY rank LIMIT 20',(run,day,minute))];return self.send(dict(decision=dict(d) if d else None,candidates=c))
    if p.path=='/api/decisions':return self.send([dict(r) for r in con.execute('SELECT day,minute,action,reason,top_symbol,top_score,qualified FROM decisions WHERE run_id=? AND day>=? AND day<=? ORDER BY day,minute',(q['run'],q.get('start','2026-08-01'),q.get('end','2026-09-30')))])
    if p.path=='/api/live':return self.send(dict(observations=[dict(r) for r in con.execute('SELECT id,observed_at,market_state,action,reason FROM observations ORDER BY id DESC LIMIT 100')],fills=[dict(r) for r in con.execute('SELECT * FROM live_fills ORDER BY observation_id DESC')],forecasts=forecast_summary(con),signals=signal_summary(con)))
   finally:con.close()
   return self.send({'error':'not_found'},404)
  except Exception as e:traceback.print_exc();self.send({'error':type(e).__name__+': '+str(e)[:240]},400)
 def do_POST(self):
  if self.path!='/api/refresh':return self.send({'error':'not_found'},404)
  threading.Thread(target=refresh_job,daemon=True).start();self.send(dict(accepted=True,job=JOB),202)
 def log_message(self,fmt,*args):pass

def monitor():
 # Application scheduler: aligned ET bar-end +30s, not a brokerage order loop.
 last=None
 while True:
  now=datetime.now(ET);m=now.hour*60+now.minute;dep=deployment();key=(now.date().isoformat(),m//5)
  day=now.date().isoformat();close=570+SESSIONS[day]['duration_minutes'] if day in SESSIONS else None
  regular_tick=close is not None and market_state(now) in ['pre','regular'] and 245<=m<close
  closing_labels=close is not None and m==close+5
  if now.minute%5==0 and now.second>=30 and key!=last and (regular_tick or closing_labels) and dep['observe_from']<=day<=dep['observe_through']:
   last=key;refresh_job()
  time.sleep(1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8770);ap.add_argument('--once',action='store_true');a=ap.parse_args()
 if a.once:snapshot(True);settle_live();return
 threading.Thread(target=monitor,daemon=True).start();print('http://127.0.0.1:'+str(a.port),flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
