"""One ranked purchase, settled cash and conservative five-minute exits."""
from __future__ import annotations
import math
import pandas as pd
from common import *

def replay(frame,threshold,initial=100000.):
 f=frame.sort_values(['day','minute','score','symbol'],ascending=[True,True,False,True]).copy()
 cash=initial;unsettled=[];position=None;bought=set();trades=[];decisions=[];equity=[];last_day=None
 days=list(SESSIONS);nextday={d:days[i+1] for i,d in enumerate(days[:-1])}
 for (day,minute),g in f.groupby(['day','minute'],sort=True,observed=True):
  day=str(day);minute=int(minute)
  if day!=last_day:
   paid=sum(v for d,v in unsettled if d<=day);cash+=paid;unsettled=[(d,v) for d,v in unsettled if d>day];last_day=day
  if position and (position['day']<day or position['exit_minute']<=minute):
   if position['exit'] is not None:
    proceeds=position['qty']*position['exit']
    if nextday[position['day']]<=day:cash+=proceeds
    else:unsettled.append((nextday[position['day']],proceeds))
    position=None
   # Missing future data remains an unresolved holding, never frees capital.
  mark=position['entry'] if position else 0.
  if position:
   same=g[g.symbol==position['symbol']]
   if len(same):mark=float(same.reference.iloc[0])
  value=cash+sum(v for _,v in unsettled)+(position['qty']*mark if position else 0.)
  available=g[(g.score>=threshold)&(g.ex_action==0)].sort_values(['score','symbol'],ascending=[False,True]) if threshold is not None else g.iloc[:0]
  available=available[~available.symbol.map(lambda s:(day,str(s)) in bought)]
  best=available.iloc[0] if len(available) else None
  d=dict(day=day,minute=minute,candidate_count=len(g),qualified=len(available),top_symbol=str(best.symbol) if best is not None else None,top_score=float(best.score) if best is not None else None,action='no_action',reason='no_qualified_candidate',cash_before=cash,equity=value)
  if position:d['reason']='position_open'
  elif best is not None:
   budget=min(cash,value*.2);estimate=float(best.reference)*1.001;qty=int(budget/estimate)
   if qty<1:d['reason']='settled_cash_insufficient'
   elif qty>float(best.last_volume)*.01:d['reason']='top1_liquidity_rejected'
   else:
    # Next-bar fill is a future execution event, never a ranking input.
    entry=float(best.entry) if pd.notna(best.get('entry')) else estimate;qty=min(qty,int(budget/entry))
    if qty<1:d['reason']='entry_gap_cash_rejected'
    else:
     known=pd.notna(best.get('y'));trade=dict(id=len(trades)+1,day=day,minute=minute,entry_minute=int(best.entry_minute),symbol=str(best.symbol),score=float(best.score),baseline=float(best.baseline),reference=float(best.reference),entry=entry,target=entry*1.03,exit=float(best.exit) if known else None,exit_minute=int(best.exit_minute) if known else 570+SESSIONS[day]['duration_minutes'],y=int(best.y) if known else None,net=float(best.net) if known else None,mae=float(best.mae) if known else None,qty=qty,budget=entry*qty,feature_available=best.feature_available,feature_row={k:clean(best[k]) for k in g.columns if k not in ['entry','exit','exit_minute','y','net','mfe','mae','label_end','complete_future','score','signal']})
     cash-=qty*entry;position=trade;bought.add((day,str(best.symbol)));trades.append(trade);d.update(action='buy',reason='rank1_qualified',trade_id=trade['id'])
  decisions.append(d);equity.append(dict(day=day,minute=minute,value=value,cash=cash,unsettled=sum(v for _,v in unsettled),position=position['symbol'] if position else None))
 # Last session close is an event even though predictions stop 30m earlier.
 if position and position['exit'] is not None:
  unsettled.append((nextday[position['day']],position['qty']*position['exit']));position=None
 final=cash+sum(v for _,v in unsettled)+(position['qty']*position['entry'] if position else 0.)
 if last_day:equity.append(dict(day=last_day,minute=570+SESSIONS[last_day]['duration_minutes'],value=final,cash=cash,unsettled=sum(v for _,v in unsettled),position=position['symbol'] if position else None))
 m=metrics(trades,decisions,equity,initial)
 if 'y' in frame:
  potential=int((frame.groupby(['symbol','day'],observed=True).y.max()==1).sum());m.update(potential_win_stock_days=potential,recall_unique_stock_day=sum(t['y']==1 for t in trades)/potential if potential else None,recall_definition='distinct winning bought stock-days / distinct stock-days with any delayed-entry +3% opportunity')
 return dict(trades=trades,decisions=decisions,equity=equity,metrics=m)

def metrics(trades,decisions,equity,initial=100000.):
 n=len(trades);tp=sum(t['y']==1 for t in trades);known=[t for t in trades if t['net'] is not None];baseline=float(np.mean([t['baseline'] for t in trades])) if n else None;values=np.asarray([x['value'] for x in equity])
 bystock=[]
 for s in sorted(set(t['symbol'] for t in trades)):
  ts=[t for t in trades if t['symbol']==s];k=sum(t['y']==1 for t in ts);p=k/len(ts);base=float(np.mean([t['baseline'] for t in ts]));bystock.append(dict(symbol=s,n=len(ts),tp=k,precision=p,ci=wilson(k,len(ts)),baseline=base,lift=p-base,passed=len(ts)>=12 and p>=.7 and p-base>=.03))
 return dict(n=n,tp=tp,fp=n-tp,unknown=n-len(known),precision=tp/n if n else None,ci=wilson(tp,n),date_count=len(set(t['day'] for t in trades)),baseline=baseline,lift=tp/n-baseline if n else None,net_mean=float(np.mean([t['net'] for t in known])) if known else None,total_return=float(values[-1]/initial-1) if len(values) else 0.,max_drawdown=float(np.min(values/np.maximum.accumulate(values)-1)) if len(values) else 0.,worst_trade=float(min(t['net'] for t in known)) if known else None,worst_mae=float(min(t['mae'] for t in known)) if known else None,buy_decisions=n,total_decisions=len(decisions),no_action=len(decisions)-n,holding_minutes_mean=float(np.mean([t['exit_minute']-t['entry_minute'] for t in trades])) if n else None,stocks=bystock)

def gate(m,n=12,days=10):
 return bool(m['n']>=n and m['date_count']>=days and m['precision']>=.7 and m['lift']>=.03 and m['net_mean'] is not None and m['net_mean']>0 and m['unknown']==0)
