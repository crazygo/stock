"""Causal minute windows, one row per stock/session; label-independent peers."""
from __future__ import annotations
import hashlib,json,sys,importlib.util
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preopen_v2_source',ROOT/'analysis/preopen_intraday_v2/build.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
START,END='2024-10-03','2026-10-02'
GROUPS={
 'optics':['AAOI','COHR','LITE','CRDO','CIEN','MRVL','VRT','ALAB'],
 'chips':['AMD','AVGO','ARM','AMAT','QCOM','SNPS','TER','RMBS','SMTC','NVDA','INTC'],
 'memory':['MU','WDC','STX'],
 'cloud':['NBIS','CRWV','ORCL','PLTR','GOOG'],
 'quantum':['IONQ','RGTI','QBTS'],
 'biology':['TXG','TWST','SDGR','LIFE'],
 'other':['AXTI','NOK','CBRS']}
GROUP_OF={s:k for k,v in GROUPS.items() for s in v}
WINDOWS=[10,15,30,45,60,90,120,180]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(g,prefix,anchor):
 if g is None or not len(g):return {prefix+'_missing':1.,prefix+'_count':0.,prefix+'_volume':np.nan}
 o,h,l,c,v=[g[k].to_numpy(float) for k in ['open','high','low','close','volume']]
 step=np.diff(np.log(np.r_[o[0],c]));travel=np.abs(step).sum();total=v.sum();spread=max(h.max()-l.min(),1e-9)
 weights=v/total if total else np.zeros(len(v));vw=np.sum((h+l+c)/3*v)/total if total else np.nan
 signed=np.sign(c-o);last=max(1,len(v)//3)
 corr=float(np.corrcoef(np.abs(step),np.log1p(v))[0,1]) if np.std(np.abs(step))>1e-9 and np.std(np.log1p(v))>1e-9 and len(v)>2 else 0.
 return {prefix+'_missing':0.,prefix+'_count':float(len(g)),prefix+'_ret':float(c[-1]/o[0]-1),prefix+'_gap':float(c[-1]/anchor-1),
  prefix+'_range':float((h.max()-l.min())/anchor),prefix+'_eff':float(abs(np.log(c[-1]/o[0]))/travel) if travel else 0.,
  prefix+'_position':float((c[-1]-l.min())/spread),prefix+'_recovery':float(c[-1]/l.min()-1),prefix+'_fade':float(c[-1]/h.max()-1),
  prefix+'_trough':float(np.argmin(l)/max(1,len(l)-1)),prefix+'_volume':float(total),prefix+'_active':float((v>0).mean()),
  prefix+'_vol_hhi':float(np.sum(weights**2)),prefix+'_signed_volume':float(np.sum(v*signed)/total) if total else 0.,
  prefix+'_late_volume':float(v[-last:].sum()/max(total,1)),prefix+'_burst':float(v.max()/max(v.mean(),1)),
  prefix+'_bar_vwap_gap':float(c[-1]/vw-1) if np.isfinite(vw) else np.nan,prefix+'_vol_price_corr':corr,
  prefix+'_rv':float(np.sqrt(np.sum(step**2))),prefix+'_positive':float((step>0).mean())}
def make_symbol(s,calendar,write_raw=True):
 f,sources,audit=legacy.load(s)
 if f.empty:return [],[],[],dict(symbol=s,status='missing',**audit),sources
 if write_raw:
  (OUT/'raw').mkdir(exist_ok=True);f.to_parquet(OUT/'raw'/f'{s}.parquet',index=False,compression='zstd',compression_level=7)
 close={d:570+t['duration_minutes'] for d,t in calendar.items()};days=list(calendar);previous={d:days[i-1] for i,d in enumerate(days) if i}
 regular={};daily=[]
 for d,g in f[(f.minute>=570)&(f.minute<f.day.map(close))].groupby('day'):
  if d in close and np.array_equal(g.minute.to_numpy(),np.arange(570,close[d],5)):
   regular[d]=g;o=float(g.open.iloc[0]);daily.append(dict(day=d,o=o,h=float(g.high.max()),l=float(g.low.min()),c=float(g.close.iloc[-1]),v=float(g.volume.sum()),mfe=float(g.high.max()/o-1),ret=float(g.close.iloc[-1]/o-1),span=float((g.high.max()-g.low.min())/o)))
 if not daily:return [],[],[],dict(symbol=s,status='no_rth',**audit),sources
 # Reindex BEFORE shifting: today's future RTH completeness must not decide
 # whether today's already-known historical features exist.
 dframe=pd.DataFrame(daily).set_index('day').reindex([d for d in days if '2024-08-01'<=d<=END]);hist=pd.DataFrame(index=dframe.index)
 for n in [3,5,10,20,30,60]:
  hist[f'h_return_{n}']=dframe.c.pct_change(n).shift(1)
  hist[f'h_rv_{n}']=np.log(dframe.c).diff().rolling(n,min_periods=min(n,10)).std().shift(1)
  for col in ['span','ret','v']:
   hist[f'h_{col}_{n}']=dframe[col].rolling(n,min_periods=min(n,10)).mean().shift(1)
  hist[f'h_base_{n}']=(dframe.mfe>=.03).astype(float).where(dframe.mfe.notna()).rolling(n,min_periods=min(n,10)).mean().shift(1)
 hist['h_previous_ret']=dframe.ret.shift(1);hist['h_previous_mfe']=dframe.mfe.shift(1);hist['h_previous_volume']=dframe.v.shift(1)
 actions=set()
 for base in [legacy.HISTORY/'corporate_actions',ROOT/'market_data/corporate_actions',legacy.OUT/'cache/corporate_actions']:
  path=base/(s+'.parquet')
  if path.exists():
   a=pd.read_parquet(path);sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),kind='actions'))
   if 'ex_div_date' in a:actions.update(a.ex_div_date.astype(str).str[:10])
 pre={d:g for d,g in f[(f.minute>=240)&(f.end.dt.hour*60+f.end.dt.minute<=565)].groupby('day')}
 f['night_day']=(f.start+pd.to_timedelta((f.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d')
 night={d:g for d,g in f[(f.minute>=1200)|(f.minute<240)].groupby('night_day')}
 rows=[];context=[];reasons=Counter();volhistory={k:[] for k in ['pre','night']+[f'w{n}' for n in WINDOWS]}
 for d in days:
  if not '2024-08-01'<=d<=END:continue
  p=pre.get(d);prev=previous.get(d);cutoff=pd.Timestamp(d+' 09:25');decision=cutoff+pd.Timedelta(seconds=30)
  if p is not None:p=p[(p.end<=cutoff)&(p.available<=decision)]
  if prev not in regular or p is None or len(p)<50 or p.end.max()!=cutoff or (p.volume>0).sum()<1:
   reasons['missing_pre_or_prior_session']+=1;continue
  anchor=float(regular[prev].close.iloc[-1]);n=night.get(d)
  if n is not None:n=n[(n.end<=pd.Timestamp(d+' 04:00'))&(n.available<=decision)]
  row=dict(symbol=s,day=d,group=GROUP_OF.get(s,'benchmark'),decision_at=decision.isoformat(),feature_end=cutoff.isoformat(),
   max_source_available=max(p.available.max(),n.available.max() if n is not None and len(n) else p.available.max()).isoformat(),pre_reference=float(p.close.iloc[-1]),**stat(p,'pre',anchor),**stat(n,'night',anchor))
  row['night_quiet']=float(n is not None and len(n)>0 and n.volume.sum()==0)
  row['pre_trade_age']=float((cutoff-p.loc[p.volume>0,'end'].max()).total_seconds()/60)
  parts={'pre':p,'night':n}
  for minutes in WINDOWS:
   k=f'w{minutes}';g=p[p.start>=cutoff-pd.Timedelta(minutes=minutes)];parts[k]=g;row.update(stat(g,k,anchor))
  for k,g in parts.items():
   value=float(g.volume.sum()) if g is not None and len(g) else np.nan
   priorvol=volhistory[k];med=np.median(priorvol[-20:]) if len(priorvol)>=10 else np.nan
   row[k+'_rvol']=value/max(med,1) if np.isfinite(med) and np.isfinite(value) else np.nan
   if np.isfinite(value):priorvol.append(value)
  # Minute windows slide in 5-minute steps; these are features, not duplicated labels.
  for minutes in [15,30,60]:
   k=minutes//5;returns=p.close.to_numpy()[k-1:]/p.open.to_numpy()[:len(p)-k+1]-1
   row[f'slide{minutes}_max']=float(returns.max());row[f'slide{minutes}_min']=float(returns.min());row[f'slide{minutes}_dispersion']=float(returns.std());row[f'slide{minutes}_latest_rank']=float((returns<=returns[-1]).mean())
  if d in hist.index:row.update({k:float(v) for k,v in hist.loc[d].items()})
  else:row.update({k:np.nan for k in hist.columns})
  scale=max(row.get('h_span_20',.01),.001)
  for key in ['pre_gap','pre_range','pre_recovery','pre_fade','w30_ret','w60_ret','w120_ret','night_ret']:
   row[key+'_scaled']=row.get(key,np.nan)/scale
  context.append(row.copy())
  if d<START:continue
  reg=regular.get(d)
  if reg is None:reasons['incomplete_future_rth']+=1;continue
  if d in actions:reasons['corporate_action']+=1;continue
  if not np.isfinite(row.get('h_span_20',np.nan)):reasons['history_warmup']+=1;continue
  entry=float(reg.open.iloc[0]);mfe=float(reg.high.max()/entry-1);mae=float(reg.low.min()/entry-1);close_ret=float(reg.close.iloc[-1]/entry-1);hit=reg[reg.high>=entry*1.03]
  delayed=float(reg.open.iloc[1]);dr=reg.iloc[1:];dmfe=float(dr.high.max()/delayed-1)
  row.update(reference=entry,mfe=mfe,mae=mae,close_ret=close_ret,y=int(mfe>=.03),label_end=reg.end.iloc[-1].isoformat(),
   touch_at=hit.start.iloc[0].isoformat() if len(hit) else '',cost_proxy=.028 if mfe>=.03 else close_ret-.002,
   delayed_mfe=dmfe,delayed_y=int(dmfe>=.03),delayed_cost_proxy=.028 if dmfe>=.03 else float(dr.close.iloc[-1]/delayed-1)-.002)
  rows.append(row)
 return rows,context,daily,dict(symbol=s,status='available',rows=len(rows),complete_rth=len(daily),first_raw=str(f.start.min()),last_raw=str(f.end.max()),reasons=dict(reasons),**audit),sources

def enrich_peer(frame,context):
 """Use all prefix-qualified contexts even when that peer's future label is absent."""
 keys=['pre_gap','w30_ret','w60_ret','pre_rvol','night_ret','h_return_30','h_rv_30','h_base_30']
 lookup=context.set_index(['symbol','day']);parts=[]
 for s,g in frame.groupby('symbol'):
  p=g.copy();peers=[x for x in GROUPS.get(GROUP_OF.get(s,''),[]) if x!=s]
  for bench in ['QQQ','SOXX','IGV']:
   b=context[context.symbol==bench].set_index('day')
   for k in keys:p[bench.lower()+'_'+k]=p.day.map(b[k]) if k in b else np.nan
  for k in keys:
   available=context[context.symbol.isin(peers)].groupby('day')[k].mean() if peers else pd.Series(dtype=float)
   p['peer_'+k]=p.day.map(available)
  breadth=context[context.symbol.isin(peers)].groupby('day').pre_gap.agg(lambda x:float((x>0).mean())) if peers else pd.Series(dtype=float)
  p['peer_breadth']=p.day.map(breadth);p['peer_count']=p.day.map(context[context.symbol.isin(peers)].groupby('day').size())
  p['relative_gap']=p.pre_gap-p.qqq_pre_gap;p['relative_w30']=p.w30_ret-p.qqq_w30_ret;p['peer_relative_gap']=p.pre_gap-p.peer_pre_gap
  parts.append(p)
 return pd.concat(parts,ignore_index=True).sort_values(['day','symbol']).reset_index(drop=True)

def main():
 universe=json.loads((legacy.OUT/'universe.json').read_text());favorites=sorted(s for s,m in universe['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK')
 calendar={s['session_date']:s for s in json.loads((legacy.HISTORY/'calendar.json').read_text())['sessions']}
 symbols=sorted(set(favorites+['QQQ','SOXX','IGV']+sum(GROUPS.values(),[])));rows=[];contexts=[];daily={};coverage=[];sources=[]
 for s in symbols:
  r,c,d,a,src=make_symbol(s,calendar);rows.extend(r);contexts.extend(c);daily[s]=d;coverage.append(a);sources.extend(src);print(json.dumps({'symbol':s,'rows':len(r),'contexts':len(c)}),flush=True)
 context=pd.DataFrame(contexts);panel=enrich_peer(pd.DataFrame(rows),context)
 panel.to_parquet(OUT/'panel.parquet',index=False,compression='zstd',compression_level=7);context.to_parquet(OUT/'contexts.parquet',index=False,compression='zstd',compression_level=7)
 (OUT/'daily.json').write_text(json.dumps(daily,separators=(',',':')))
 audit=dict(start=START,end=END,favorites=favorites,groups=GROUPS,coverage=coverage,sources=sources,panel_sha256=sha(OUT/'panel.parquet'),calendar_sha256=sha(legacy.HISTORY/'calendar.json'),universe_sha256=sha(legacy.OUT/'universe.json'),protocol_sha256=sha(OUT/'PROTOCOL.md'),availability='assumed_end_plus_1s',universe='current_snapshot_not_PIT',total_rows=len(panel))
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps({'rows':len(panel),'features':len(panel.columns),'favorites':len(favorites)}),flush=True)
if __name__=='__main__':main()
