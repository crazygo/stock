"""One-shot report: three ranked options for EACH frozen model, no orders."""
from __future__ import annotations
import argparse,json,sys,sqlite3,time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from common import OUT,ET,ROUTES,NAMES,ALGORITHMS,SESSIONS,clean,write
from data import prior_context,intraday,peers,GROUPS
from cloud_data import MarketData
from portable import NativeModel,digest

def clock(minute):return f'{minute//60:02d}:{minute%60:02d}'
def at(day,minute,seconds=0):return (pd.Timestamp(day,tz=ET)+pd.Timedelta(minutes=minute,seconds=seconds)).isoformat()

def schedule(now):
 day=now.date().isoformat();m=now.hour*60+now.minute;session=SESSIONS.get(day);close=570+session['duration_minutes'] if session else None
 state='closed' if session is None else 'pre' if 240<=m<570 else 'regular' if 570<=m<close else 'post' if close<=m<1200 else 'night'
 if state in ['pre','regular'] and 245<=m<=close-30:
  # Wait for the +30s decision point; before it, use the previous eligible cut.
  cut=m//5*5
  if now.replace(tzinfo=None)<pd.Timestamp(day)+pd.Timedelta(minutes=cut,seconds=30):cut-=5
  if cut>=245:return dict(day=day,cut=cut,state=state,in_window=True)
 eligible=[d for d in SESSIONS if d<day or (d==day and state=='post')]
 reference=max(eligible);return dict(day=reference,cut=570+SESSIONS[reference]['duration_minutes']-30,state=state,in_window=False)

def build_features(market,symbols,day,cut):
 rows=[];missing=[];until=pd.Timestamp(day)+pd.Timedelta(minutes=cut)
 for s in symbols:
  raw=market.raw(s)
  if raw.empty:missing.append(s);continue
  f=raw[(raw.end<=until)&(raw.end>=pd.Timestamp(day)-pd.Timedelta(days=110))]
  ctx,_=prior_context(f,s);one=intraday(f,ctx,s,labels=False,first=day,last=day,only_minutes={cut})
  if len(one):rows.append(one)
  else:missing.append(s)
 if not rows:return pd.DataFrame(),missing
 result=peers(pd.concat(rows,ignore_index=True))
 # Future labels must NEVER appear in a report feature snapshot.
 assert not {'y','entry','target','exit','label_end','complete_future'}&set(result)
 return result,missing

def persist_report(report,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);con=sqlite3.connect(path)
 with con:
  con.execute('CREATE TABLE IF NOT EXISTS recommendation_reports(id TEXT PRIMARY KEY,observed_at TEXT,feature_cutoff TEXT,current_probability INTEGER,payload TEXT)')
  key=report['model_manifest_sha256']+':'+report['observed_at'];con.execute('INSERT OR IGNORE INTO recommendation_reports VALUES(?,?,?,?,?)',(key,report['observed_at'],report['feature_cutoff'],int(report['current_probability']),json.dumps(clean(report),ensure_ascii=False,allow_nan=False)))
 con.close()

def run(args,now=None):
 now=now or datetime.now(ET);started=time.monotonic();target_id=getattr(args,'target','intraday3pct');specs={'intraday3pct':(1,.03),'5d5pct':(5,.05),'10d10pct':(10,.10)}
 if target_id not in specs:raise ValueError('Unknown target')
 sessions,gain=specs[target_id]
 model_dir=OUT/'portable_models' if target_id=='intraday3pct' else OUT/'horizon_v1'/target_id/'portable_models';manifest=json.loads((model_dir/'manifest.json').read_text());models={m['route']:NativeModel(model_dir/m['file'],m['sha256']) for m in manifest['all_routes']};plan=schedule(now)
 pool=sorted(set(sum([m['registered'] for m in manifest['all_routes']],[]))|set(sum(GROUPS.values(),[]))|set(getattr(args,'extra_symbols',[])));market=MarketData(args.data_dir)
 progress=lambda msg:print(msg,file=sys.stderr,flush=True)
 if not args.offline:market.refresh(pool,plan['day'],plan['cut'],args.futu_host,args.futu_port,progress)
 frame,missing=build_features(market,pool,plan['day'],plan['cut']);missing_requested=list(missing);requested_scored=len(frame);source='market_data';day=plan['day'];cut=plan['cut'];current=bool(plan['in_window']);reference_path=model_dir/'reference_snapshot.json'
 if not current:
  # Outside the forecast window, prefer the newest usable reference, not stale R2.
  available_days=[str(market.raw(s).day.max()) for s in pool if not market.raw(s).empty]
  possible=[d for d in available_days if d in SESSIONS and d<=day]
  actual=max(possible) if possible else None
  if actual and actual!=day:
   day=actual;cut=570+SESSIONS[day]['duration_minutes']-30;frame,missing=build_features(market,pool,day,cut)
 if reference_path.exists() and (frame.empty or not current):
  reference=json.loads(reference_path.read_text())
  if reference['model_manifest_sha256']!=digest(model_dir/'manifest.json'):raise ValueError('Reference snapshot model mismatch')
  reference_available=pd.Timestamp(at(reference['day'],reference['minute'],1)).to_pydatetime()
  if reference_available<=now and (frame.empty or reference['day']>day):
   frame=pd.DataFrame(reference['features']);day=reference['day'];cut=reference['minute'];source='bundled_reference_snapshot';current=False;missing=reference['missing'];progress('使用随代码保存的参考快照；这不是当前可操作概率')
 # The real clock governs expiry for live invocations; --as-of is replay only.
 finish=datetime.now(ET) if args.as_of is None else now
 entry_at=pd.Timestamp(at(day,cut+5));expired=finish>=entry_at.to_pydatetime();current=bool(current and len(frame) and not args.as_of and day==now.date().isoformat() and not expired and day[:7]==manifest['model_month'])
 dates=sorted(SESSIONS);last_index=dates.index(day)+sessions-1
 if last_index>=len(dates):raise ValueError('Target exceeds registered official calendar')
 last_day=dates[last_index];win_end=at(last_day,570+SESSIONS[last_day]['duration_minutes']);condition=f'下一根 5m Open×1.001 为评价价；买入当日计第 1 个交易日，买入后至第 {sessions} 个交易日常规盘收盘的 High≥评价价×{1+gain:.2f}。只统计常规盘；指示目标按参考 Close 推算，入场后重算。'
 report=dict(version='recommendation_v1',target_id=target_id,target_sessions=sessions,target_gain=gain,model_manifest_sha256=digest(model_dir/'manifest.json'),observed_at=finish.isoformat(),market_state=plan['state'],data_source=source,reference_day=day,feature_cutoff=at(day,cut),requested_feature_cutoff=at(plan['day'],plan['cut']),requested_feature_symbols=requested_scored,missing_requested_features=missing_requested,decision_at=at(day,cut,30),evaluation_entry_at=entry_at.isoformat(),window_end=win_end,current_probability=current,replay=bool(args.as_of),model_month=manifest['model_month'],model_frozen_at=manifest['frozen_at'],signal_admitted=manifest['admitted'],winning_condition=condition,options_are_issued_signals=False,missing=missing,acquisition=dict(attempts=market.audit,errors=market.errors),routes=[],elapsed_seconds=round(time.monotonic()-started,2))
 for route in ROUTES:
  model=models[route];a=model.a;meta=next(m for m in manifest['all_routes'] if m['route']==route);f=frame[frame.symbol.isin(a['registered'])].copy() if len(frame) else frame.copy();phase='pre' if cut<=570 else 'regular';threshold=a['thresholds'][phase];options=[]
  if len(f):
   f['stock_id']=f.symbol.map(a['stock_map'])
   for c in a['features']:
    if c!='symbol':f[c]=f[c].astype('float32')
   f['probability']=model.predict(f);f=f[np.isfinite(f.probability)].sort_values(['probability','symbol'],ascending=[False,True])
   for rank,r in enumerate(f.head(args.top).to_dict('records'),1):
    probability=float(r['probability']);available=pd.Timestamp(r['feature_available']);due=pd.Timestamp(day)+pd.Timedelta(minutes=cut,seconds=30);fresh=bool(available<=due and available>=pd.Timestamp(day)+pd.Timedelta(minutes=cut));confidence=bool(threshold is not None and probability>=threshold);valid=bool(current and manifest['admitted'] and not args.as_of and confidence and fresh and not r['ex_action'])
    reason='replay_only' if args.as_of else 'reference_only' if not current else 'research_target_not_admitted' if not manifest['admitted'] else 'confidence_threshold_unavailable' if threshold is None else 'below_confidence_threshold' if not confidence else 'corporate_action' if r['ex_action'] else 'features_not_timely' if not fresh else 'confidence_qualified_reference'
    baseline=a['baseline_table'].get(f"{r['symbol']}|{cut}",a['baseline_minute'].get(str(cut),.1));reference=float(r['reference'])
    options.append(dict(rank=rank,symbol=r['symbol'],probability=probability,baseline=baseline,threshold=threshold,confidence_qualified=confidence,current_qualified_candidate=valid,issued_signal=False,status=reason,reference_price=reference,indicative_evaluation_price=reference*1.001,indicative_target_price=reference*1.001*(1+gain),target_formula=f'actual_next_5m_open * 1.001 * {1+gain:.2f}',feature_available=pd.Timestamp(r['feature_available']).tz_localize(ET).isoformat(),evaluation_entry_at=entry_at.isoformat(),window_end=report['window_end'],winning_condition=report['winning_condition']))
  report['routes'].append(dict(route=route,name=NAMES[route],algorithm=ALGORITHMS[route],threshold=threshold,registered=len(a['registered']),scored=len(f),missing=sorted(set(a['registered'])-set(f.symbol if len(f) else [])),model_sha256=meta['sha256'],options=options))
 if args.save_features:
  if frame.empty:raise ValueError('Cannot save an empty reference snapshot')
  write(Path(args.save_features),dict(day=day,minute=cut,model_manifest_sha256=digest(model_dir/'manifest.json'),missing=missing,features=frame.to_dict('records'),not_current=True,source='causal available prefix; no future labels'))
 if args.output:write(Path(args.output),report)
 if args.record:persist_report(report,args.record)
 return clean(report)

def markdown(report):
 lines=[f"目标：{report['target_sessions']} 个交易日触及 +{report['target_gain']:.0%}（买入当日计第 1 日）。",f"行情截止：{report['feature_cutoff']}；计算：{report['observed_at']}。",'当前概率。' if report['current_probability'] else '参考概率：闭市、数据不足或入场时限已过；不属于当前有效信号。',f"模型冻结：{report['model_frozen_at']}；十月有效信号资格：{'通过' if report['signal_admitted'] else '未通过'}。",'']
 for route in report['routes']:
  threshold='未通过，弃权' if route['threshold'] is None else f"{route['threshold']:.0%}"
  lines += [f"### {route['name']}（{route['algorithm']}）",f"已计算 {route['scored']}/{route['registered']} 只；本时段置信度门槛：{threshold}。",'', '| 排名 | 股票 | 预测达成概率 | 参考价 | 指示目标价 | 状态 |','|---:|---|---:|---:|---:|---|']
  for o in route['options']:lines.append(f"| {o['rank']} | {o['symbol']} | {o['probability']:.2%} | ${o['reference_price']:.4f} | ${o['indicative_target_price']:.4f} | {'合格候选（未发信号）' if o['current_qualified_candidate'] else '参考 / 无有效信号'} |")
  if not route['options']:lines.append('| — | 行情不足，未制造概率 | — | — | — | 待补齐 |')
  lines+=['',f"每个选项达标条件：{report['evaluation_entry_at']} 的 Open×1.001 为评价价；买入后的常规盘至 {report['window_end']} 的 High 触及评价价×{1+report['target_gain']:.2f}。盘前触及不算。表中目标是参考 Close 推算值，实际入场后重算。",'']
 lines+=['预测概率不等于已验证的达成率；这次参考报告不会写成已发有效信号，也不下单。']
 if report['acquisition']['errors']:lines+=['','行情补齐情况：'+json.dumps(report['acquisition']['errors'],ensure_ascii=False)]
 return '\n'.join(lines)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--target',choices=['intraday3pct','5d5pct','10d10pct','all'],default='intraday3pct');ap.add_argument('--top',type=int,default=3);ap.add_argument('--format',choices=['markdown','json'],default='markdown');ap.add_argument('--data-dir');ap.add_argument('--futu-host');ap.add_argument('--futu-port',type=int,default=11111);ap.add_argument('--offline',action='store_true');ap.add_argument('--as-of',help='Timezone-aware historical replay time; never current/issued');ap.add_argument('--output');ap.add_argument('--record',help='Optional local SQLite reference-report log');ap.add_argument('--save-features',help=argparse.SUPPRESS);args=ap.parse_args()
 if not 1<=args.top<=20:ap.error('--top must be between 1 and 20')
 now=None
 if args.as_of:
  stamp=pd.Timestamp(args.as_of)
  if stamp.tzinfo is None:ap.error('--as-of must contain a timezone offset')
  now=stamp.tz_convert(ET).to_pydatetime()
 try:
  if args.target=='all':
   reports=[];extra=set()
   for target in ['intraday3pct','5d5pct','10d10pct']:
    path=OUT/'portable_models/manifest.json' if target=='intraday3pct' else OUT/'horizon_v1'/target/'portable_models/manifest.json';manifest=json.loads(path.read_text());extra.update(sum([m['registered'] for m in manifest['all_routes']],[]))
   for i,target in enumerate(['intraday3pct','5d5pct','10d10pct']):
    options=argparse.Namespace(**vars(args));options.target=target;options.extra_symbols=sorted(extra);options.offline=args.offline or i>0;options.output=None;options.save_features=None;reports.append(run(options,now))
   report=dict(version='all_targets_recommendation_v1',orders_sent=False,targets=reports)
   if args.output:write(Path(args.output),report)
   print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) if args.format=='json' else '\n\n'.join(markdown(r) for r in reports))
  else:
   report=run(args,now);print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) if args.format=='json' else markdown(report))
 except Exception as e:
  print(json.dumps({'error':type(e).__name__,'detail':str(e)[:300],'orders_sent':False},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=='__main__':main()
