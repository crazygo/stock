"""Shared causal feature code for replay and live. Never condition features on future RTH."""
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from common import *
from feature_primitives import GROUPS,GROUP_OF,stat
STATIC=['h_span_3','h_span_20','h_base_20','h_return_30','h_rv_30','h_previous_ret','h_previous_mfe','pre_gap','pre_range','pre_rvol','night_ret','night_missing','pre_trade_age','w30_ret','w30_eff','w30_signed_volume','w60_ret','slide30_min','slide30_max']
META=['symbol','day','minute','reference','feature_available','entry_minute']
LABELS=['entry','exit','exit_minute','y','net','mfe','mae','label_end','complete_future']

def read_raw(s):
 p=OUT/'raw'/f'{s}.parquet'
 if p.exists():return pd.read_parquet(p)
 p=ROOT/'analysis/preopen_ranked_policy_v4/raw'/f'{s}.parquet'
 if p.exists():return pd.read_parquet(p)
 p=ROOT/'analysis/preopen_stock_cycle_v3/raw'/f'{s}.parquet'
 if p.exists():return pd.read_parquet(p)
 paths=sorted((ROOT/'market_data/model_training_history_v1/parts'/s).glob('*/bars.parquet'))
 frames=[]
 for p in paths:
  r=pd.read_parquet(p)
  if not {'start_at_et','end_at_et','price_basis'}.issubset(r) or set(r.price_basis.dropna())!={'NONE'}:continue
  n=r[['open','high','low','close','volume']].copy()
  for dest,src in [('start','start_at_et'),('end','end_at_et')]:n[dest]=pd.to_datetime(r[src],utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
  n['available']=n.end+pd.Timedelta(seconds=1);n['day']=n.start.dt.strftime('%Y-%m-%d');n['minute']=n.start.dt.hour*60+n.start.dt.minute;frames.append(n)
 return pd.concat(frames,ignore_index=True).drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True) if frames else pd.DataFrame(columns=['start','end','available','day','minute','open','high','low','close','volume'])

def prior_context(f,s):
 days=list(SESSIONS);endminute={d:570+x['duration_minutes'] for d,x in SESSIONS.items()}
 daily=[]
 for d,g in f[(f.minute>=570)&(f.minute<f.day.map(endminute))].groupby('day',sort=True):
  if d not in endminute or not np.array_equal(g.minute.to_numpy(),np.arange(570,endminute[d],5)):continue
  o=float(g.open.iloc[0]);daily.append(dict(day=d,o=o,h=float(g.high.max()),l=float(g.low.min()),c=float(g.close.iloc[-1]),v=float(g.volume.sum()),span=float((g.high.max()-g.low.min())/o),ret=float(g.close.iloc[-1]/o-1),mfe=float(g.high.max()/o-1)))
 df=pd.DataFrame(daily).set_index('day').reindex(days) if daily else pd.DataFrame(index=days,columns=['o','h','l','c','v','span','ret','mfe'])
 hist=pd.DataFrame(index=days)
 for n in [3,20]:hist['h_span_'+str(n)]=df.span.rolling(n,min_periods=min(n,10)).mean().shift(1)
 hist['h_base_20']=(df.mfe>=.03).astype(float).where(df.mfe.notna()).rolling(20,min_periods=10).mean().shift(1)
 hist['h_return_30']=df.c.pct_change(30,fill_method=None).shift(1);hist['h_rv_30']=np.log(df.c).diff().rolling(30,min_periods=10).std().shift(1)
 hist['h_previous_ret']=df.ret.shift(1);hist['h_previous_mfe']=df.mfe.shift(1);hist['prior_close']=df.c.shift(1);hist['prior_volume']=df.v.rolling(20,min_periods=10).median().shift(1)
 pre={d:g for d,g in f[(f.minute>=240)&(f.minute<570)].groupby('day')}
 night=f[(f.minute>=1200)|(f.minute<240)].copy();night['nday']=(night.start+pd.to_timedelta((night.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d');night={d:g for d,g in night.groupby('nday')}
 contexts=[];vols=[]
 for d in days:
  p=pre.get(d);cut=pd.Timestamp(d+' 09:30');decision=cut+pd.Timedelta(seconds=30);anchor=hist.loc[d,'prior_close']
  if not np.isfinite(anchor):continue
  if p is not None:p=p[(p.end<=cut)&(p.available<=decision)]
  n=night.get(d)
  if n is not None:n=n[(n.end<=pd.Timestamp(d+' 04:00'))&(n.available<=decision)]
  r=dict(symbol=s,day=d,**hist.loc[d].to_dict(),**stat(p,'pre',anchor),**stat(n,'night',anchor))
  r['night_available']=str(n.available.max()) if n is not None and len(n) else ''
  r['pre_volume_median']=float(np.median(vols[-20:])) if len(vols)>=10 else np.nan
  r['pre_rvol']=float(p.volume.sum()/max(r['pre_volume_median'],1)) if p is not None and len(p) and len(vols)>=10 else np.nan
  if p is not None and len(p):vols.append(float(p.volume.sum()))
  r['pre_trade_age']=float((cut-p.loc[p.volume>0,'end'].max()).total_seconds()/60) if p is not None and (p.volume>0).any() else np.nan
  for m in [30,60]:r.update(stat(p[p.start>=cut-pd.Timedelta(minutes=m)] if p is not None else None,'w'+str(m),anchor))
  if p is not None and len(p)>=6:
   a=p.close.to_numpy()[5:]/p.open.to_numpy()[:-5]-1;r['slide30_min']=float(a.min());r['slide30_max']=float(a.max())
  else:r['slide30_min']=r['slide30_max']=np.nan
  contexts.append(r)
 return pd.DataFrame(contexts),daily

def intraday(f,contexts,s,labels=True,first='2025-06-01',last='2026-09-30',only_minutes=None):
 if contexts.empty:return pd.DataFrame()
 context=contexts.set_index('day');rows=[];actions=set()
 for base in [ROOT/'market_data/model_training_history_v1/corporate_actions',ROOT/'market_data/corporate_actions',ROOT/'analysis/preopen_intraday_v2/cache/corporate_actions']:
  p=base/(s+'.parquet')
  if p.exists():
   a=pd.read_parquet(p)
   if 'ex_div_date' in a:actions.update(a.ex_div_date.astype(str).str[:10])
 for d,g in f[(f.day>=first)&(f.day<=last)&(f.minute>=240)&(f.minute<960)].groupby('day',sort=True):
  if d not in SESSIONS or d not in context.index:continue
  close=570+SESSIONS[d]['duration_minutes'];g=g[g.minute<close].sort_values('start');mins=g.minute.to_numpy();o,h,l,c,v=[g[k].to_numpy(float) for k in ['open','high','low','close','volume']]
  if not len(g):continue
  ri=np.flatnonzero(mins>=570);complete=np.array_equal(mins[ri],np.arange(570,close,5));rowcontext=context.loc[d];anchor=rowcontext.prior_close
  for i in range(len(g)):
   t=int(mins[i]+5)
   if t<245 or t>close-30 or (only_minutes is not None and t not in only_minutes):continue
   ispre=t<=570;known=np.arange(i+1);reg=known[mins[:i+1]>=570];pre=known[mins[:i+1]<570]
   prefix=pre if ispre else reg;expected=np.arange(240 if ispre else 570,t,5)
   if not np.array_equal(mins[prefix],expected):continue
   decision=pd.Timestamp(d)+pd.Timedelta(minutes=t,seconds=30)
   if g.available.iloc[:i+1].max()>decision:continue
   r={k:float(rowcontext.get(k,np.nan)) for k in STATIC}
   if rowcontext.get('night_available') and pd.Timestamp(rowcontext.night_available)>decision:r['night_ret']=np.nan;r['night_missing']=1.
   # Recompute every same-day pre feature from the available prefix. Full-day
   # pre context is used only for prior-day denominators and never leaks here.
   for k in ['pre_gap','pre_range','pre_rvol','pre_trade_age','w30_ret','w30_eff','w30_signed_volume','w60_ret','slide30_min','slide30_max']:r[k]=np.nan
   if len(pre):
    pe=pre[-1];pf=pre[0];r['pre_gap']=float(c[pe]/anchor-1);r['pre_range']=float((h[pre].max()-l[pre].min())/anchor);den=rowcontext.get('pre_volume_median',np.nan)
    r['pre_rvol']=float(v[pre].sum()/max(den*min(t-240,330)/330,1)) if np.isfinite(den) else np.nan
    active=pre[v[pre]>0];r['pre_trade_age']=float(t-(mins[active[-1]]+5)) if len(active) else np.nan
    for m in [30,60]:
     idx=pre[mins[pre]>=min(t,570)-m];travel=np.abs(np.diff(np.log(np.r_[o[idx[0]],c[idx]]))).sum() if len(idx) else 0.
     if len(idx):
      r['w'+str(m)+'_ret']=float(c[idx[-1]]/o[idx[0]]-1)
      if m==30:r['w30_eff']=float(abs(np.log(c[idx[-1]]/o[idx[0]]))/max(travel,1e-9));r['w30_signed_volume']=float(np.sum(v[idx]*np.sign(c[idx]-o[idx]))/max(v[idx].sum(),1))
    if len(pre)>=6:
     returns=c[pre][5:]/o[pre][:-5]-1;r['slide30_min']=float(returns.min());r['slide30_max']=float(returns.max())
   r.update(symbol=s,day=d,minute=t,entry_minute=t+5,reference=float(c[i]),feature_available=str(g.available.iloc[:i+1].max()),ex_action=float(d in actions),prior_volume=float(rowcontext.prior_volume),is_pre=float(ispre),rth_missing=float(not len(reg)),remaining=float((close-max(t+5,570))/390),elapsed=float(max(0,t-570)/390),pre_elapsed=float(min(t-240,330)/330),rth_ret=np.nan,rth_gap=np.nan,rth_range=np.nan,rth_fade=np.nan,rth_recovery=np.nan,rth_position=np.nan,rth_volume=np.nan,rth_rvol=np.nan,last_volume=float(v[i]),liquidity_dollars=float(v[i]*c[i]))
   if len(reg):
    r.update(rth_ret=float(c[i]/o[reg[0]]-1),rth_gap=float(c[i]/anchor-1),rth_range=float((h[reg].max()-l[reg].min())/anchor),rth_fade=float(c[i]/h[reg].max()-1),rth_recovery=float(c[i]/l[reg].min()-1),rth_position=float((c[i]-l[reg].min())/max(h[reg].max()-l[reg].min(),1e-9)),rth_volume=float(v[reg].sum()),rth_rvol=float(v[reg].sum()/max(rowcontext.prior_volume*(t-570)/SESSIONS[d]['duration_minutes'],1)))
   for m in [5,15,30,60]:
    idx=known[mins[:i+1]>=t-m];j=idx[0];vv=v[idx];steps=np.diff(np.log(np.r_[o[j],c[idx]]));travel=np.abs(steps).sum();r['i'+str(m)+'_ret']=float(c[i]/o[j]-1);r['i'+str(m)+'_rv']=float(np.sqrt(np.sum(steps**2)));r['i'+str(m)+'_eff']=float(abs(np.log(c[i]/o[j]))/max(travel,1e-9));r['i'+str(m)+'_signedvol']=float(np.sum(vv*np.sign(c[idx]-o[idx]))/max(vv.sum(),1))
   segment=pre if ispre else reg;total=v[segment].sum();vw=float(np.sum((h[segment]+l[segment]+c[segment])/3*v[segment])/total) if total else np.nan;r['bar_vwap_gap']=float(c[i]/vw-1) if np.isfinite(vw) else np.nan
   if labels:
    entry_index=np.flatnonzero(mins==t+5);r.update(complete_future=bool(complete),label_end=d+f' {close//60:02d}:{close%60:02d}')
    if complete and len(entry_index):
     ei=int(entry_index[0]);entry=float(o[ei]*1.001);target=entry*1.03;future=known[:0];future=np.flatnonzero(mins>=max(570,t+5));hits=future[h[future]>=target];xi=int(hits[0]) if len(hits) else len(g)-1;exitprice=target if len(hits) else float(c[-1]);r.update(entry=entry,exit=exitprice*.999,exit_minute=int(mins[xi]+5),y=int(bool(len(hits))),net=float(exitprice*.999/entry-1),mfe=float(h[future].max()/entry-1),mae=float(l[ei:xi+1].min()/entry-1))
    else:r.update(entry=np.nan,exit=np.nan,exit_minute=close,y=np.nan,net=np.nan,mfe=np.nan,mae=np.nan)
   rows.append(r)
 return pd.DataFrame(rows)

def peers(f):
 parts=[];keys=['rth_ret','i15_ret','i30_ret','rth_rvol','rth_recovery','pre_gap','w30_ret','pre_rvol']
 for s,g in f.groupby('symbol',observed=True):
  p=g.copy();related=[x for x in GROUPS.get(GROUP_OF.get(s,''),[]) if x!=s]
  for key in keys:
   b=f[f.symbol=='QQQ'].set_index(['day','minute'])[key];p['qqq_'+key]=pd.MultiIndex.from_frame(p[['day','minute']]).map(b)
   env=f[f.symbol.isin(related)].groupby(['day','minute'],observed=True)[key].mean();p['peer_'+key]=pd.MultiIndex.from_frame(p[['day','minute']]).map(env)
  p['relative_ret']=p.rth_ret-p.qqq_rth_ret;p['relative_i15']=p.i15_ret-p.qqq_i15_ret;parts.append(p)
 return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def make(s):
 f=read_raw(s)
 if f.empty:return s,None,None,None
 save(f,OUT/'raw'/f'{s}.parquet');ctx,daily=prior_context(f,s);a=intraday(f,ctx,s)
 return s,a,ctx,daily

def main():
 u=json.loads((OUT/'universe.json').read_text());available=set(p.name for p in (ROOT/'market_data/model_training_history_v1/parts').iterdir())|set(p.stem for p in (ROOT/'analysis/preopen_stock_cycle_v3/raw').glob('*.parquet'))
 symbols=sorted((set(u['members'])|{'QQQ','SOXX','IGV'}|set(sum(GROUPS.values(),[])))&available)
 write(OUT/'calendar.json',CALENDAR);frames=[];contexts=[];daily={};coverage=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(make,s):s for s in symbols}
  for j in as_completed(jobs):
   s,f,c,d=j.result()
   if f is not None and not f.empty:frames.append(f);contexts.append(c);daily[s]=d;coverage.append(dict(symbol=s,rows=len(f),days=f.day.nunique(),labeled_days=f[f.y.notna()].day.nunique(),first=f.day.min(),last=f.day.max()))
   print(json.dumps({'symbol':s,'rows':len(f) if f is not None else 0}),flush=True)
 panel=peers(pd.concat(frames,ignore_index=True));panel['stock_id']=panel.symbol.map({s:i for i,s in enumerate(sorted(panel.symbol.unique()))}).astype(float)
 for k in panel.select_dtypes('float').columns:panel[k]=panel[k].astype('float32')
 panel['day']=panel.day.astype('category');panel['symbol']=panel.symbol.astype('category');save(panel,OUT/'panel.parquet');save(pd.concat(contexts,ignore_index=True),OUT/'contexts.parquet');write(OUT/'daily.json',daily)
 write(OUT/'coverage.json',dict(requested_pool=len(u['members']),feature_symbols=len(coverage),coverage_symbols=sorted(daily),coverage=coverage,unavailable=sorted(set(u['members'])-set(daily)),etf_membership_at=u['observed_at'],membership_mode='current_snapshot_not_PIT',raw_sources=[dict(symbol=p.stem,sha256=sha(p)) for p in (OUT/'raw').glob('*.parquet')],panel_sha256=sha(OUT/'panel.parquet')))
 print(json.dumps({'rows':len(panel),'features':len(panel.columns),'symbols':len(coverage)}),flush=True)
if __name__=='__main__':main()
