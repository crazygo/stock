"""Local-only, causal 09:25 features and same-day 5m labels, with scope audit."""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HISTORY=ROOT/'market_data/model_training_history_v1'
START,END='2024-10-03','2026-10-02'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load(symbol):
    paths=[]
    folder=HISTORY/'parts'/symbol
    paths.extend(p for p in sorted(folder.glob('*/bars.parquet')) if '2024-08'<=p.parent.name<='2026-10')
    for year in (2024,2025,2026):
        for base in (HISTORY/'us_5m',ROOT/'market_data/us_5m'):
            p=base/symbol/f'{year}.parquet'
            if p.exists(): paths.append(p)
    paths.extend(sorted((ROOT/'research/after_open_3d5pct/runs/premarket_tail_v1/overnight').glob(f'{symbol}*.parquet')))
    paths.extend(p for p in sorted((OUT/'cache'/symbol).glob('*.parquet')) if '.partial.' not in p.name)
    frames=[]; sources=[]; rejected=[]
    for p in paths:
        f=pd.read_parquet(p)
        if f.empty: continue
        if not {'start_at_et','end_at_et'}.issubset(f.columns):
            rejected.append(str(p.relative_to(ROOT))); continue
        if 'price_basis' in f and set(f.price_basis.dropna())!={'NONE'}: raise ValueError(str(p)+' price basis')
        f['start']=pd.to_datetime(f.start_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        f['end']=pd.to_datetime(f.end_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        f['available']=pd.to_datetime(f.available_at,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None) if 'available_at' in f else f.end+pd.Timedelta(seconds=1)
        f=f[(f.start>=pd.Timestamp('2024-08-01'))&(f.start<pd.Timestamp(END)+pd.Timedelta(days=1))]
        sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'rows_in_window':len(f)})
        frames.append(f[['start','end','available','open','high','low','close','volume']])
    if not frames:return pd.DataFrame(),sources,{'rejected_unknown_schema':rejected}
    f=pd.concat(frames,ignore_index=True)
    duplicates=int(f.start.duplicated().sum())
    # Later cache entries have explicit precedence; archive provenance is retained.
    f=f.drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True)
    prices=f[['open','high','low','close']].to_numpy(float)
    good=np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)&np.isfinite(f.volume)&(f.volume>=0)
    good &= (f.high>=np.max(prices,axis=1))&(f.low<=np.min(prices,axis=1))&(f.end-f.start==pd.Timedelta(minutes=5))
    bad=int((~good).sum());f=f.loc[good].reset_index(drop=True)
    f['day']=f.start.dt.strftime('%Y-%m-%d'); f['minute']=f.start.dt.hour*60+f.start.dt.minute
    return f,sources,{'deduplicated_rows':duplicates,'invalid_ohlcv':bad,'rejected_unknown_schema':rejected}

def segment(g, prefix, anchor):
    o=g.open.to_numpy(float);c=g.close.to_numpy(float);h=g.high.to_numpy(float);l=g.low.to_numpy(float);v=g.volume.to_numpy(float)
    path=np.r_[o[0],c];steps=np.diff(np.log(path));travel=np.abs(steps).sum();spread=max(h.max()-l.min(),1e-9)
    mid=max(1,len(c)//2);late=max(1,len(c)//6)
    return {prefix+'_ret':c[-1]/o[0]-1,prefix+'_gap':c[-1]/anchor-1,
            prefix+'_range':(h.max()-l.min())/anchor,prefix+'_eff':abs(np.log(c[-1]/o[0]))/travel if travel>0 else 0.,
            prefix+'_fade':c[-1]/h.max()-1,prefix+'_rebound':c[-1]/l.min()-1,
            prefix+'_position':(c[-1]-l.min())/spread,prefix+'_trough':int(np.argmin(l))/max(len(l)-1,1),
            prefix+'_early':c[mid-1]/o[0]-1,prefix+'_late':c[-1]/o[-late]-1,
            prefix+'_volume':float(v.sum()),prefix+'_active':float((v>0).mean()),prefix+'_positive':float((steps>0).mean())}

def first_barrier(g, entry, target=.03, stop=-.01):
    for r in g.itertuples():
        ht=r.high>=entry*(1+target);hs=r.low<=entry*(1+stop)
        if hs: return min(r.open/entry-1,stop)-.002, 'stop_both' if ht else 'stop'
        if ht:return target-.002,'target'
    return g.close.iloc[-1]/entry-1-.002,'close'

def build_symbol(symbol,calendar):
    f,sources,audit=load(symbol)
    if f.empty:return [],{},dict(symbol=symbol,status='no_local_5m',eligible=0,**audit),sources
    close_minutes={d:570+s['duration_minutes'] for d,s in calendar.items()}
    days=list(calendar);previous={days[i]:days[i-1] for i in range(1,len(days))}
    f['close_minute']=f.day.map(close_minutes)
    f['kind']=np.select([(f.minute>=570)&(f.minute<f.close_minute), (f.minute>=240)&(f.minute<570),
                         (f.minute>=f.close_minute)&(f.minute<1200)],['regular','pre','post'],default='night')
    # Sunday 20:00 belongs to Monday overnight input, not Friday's next date.
    f['night_day']=(f.start+pd.to_timedelta((f.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d')
    pre={d:g for d,g in f[(f.kind=='pre')&(f.end.dt.hour*60+f.end.dt.minute<=565)].groupby('day')}
    night={d:g for d,g in f[(f.minute>=1200)|(f.minute<240)].groupby('night_day')}
    regular={};daily=[];reasons=Counter()
    for d,g in f[f.kind=='regular'].groupby('day'):
        if d not in calendar:continue
        expected=np.arange(570,close_minutes[d],5)
        if np.array_equal(g.minute.to_numpy(),expected):
            regular[d]=g
            entry=float(g.open.iloc[0]); daily.append({'day':d,'open':entry,'high':float(g.high.max()),'low':float(g.low.min()),
                       'close':float(g.close.iloc[-1]),'volume':float(g.volume.sum()),'range':float((g.high.max()-g.low.min())/entry),
                       'mfe':float(g.high.max()/entry-1),'rth_ret':float(g.close.iloc[-1]/entry-1)})
        else:reasons['incomplete_rth']+=1
    df=pd.DataFrame(daily)
    if df.empty:return [],{},dict(symbol=symbol,status='no_complete_rth',eligible=0,**audit),sources
    df=df.set_index('day'); past=df[['range','mfe','rth_ret']].rolling(20,min_periods=20).mean().shift(1)
    history_count=Counter()
    for d in df.index: history_count[d[:4]]+=1
    actions=set()
    action_sources=[]
    for base in (HISTORY/'corporate_actions',ROOT/'market_data/corporate_actions',OUT/'cache/corporate_actions'):
        p=base/(symbol+'.parquet')
        if p.exists():
            a=pd.read_parquet(p);sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'rows_in_window':len(a),'kind':'corporate_actions'});action_sources.append(str(p.relative_to(ROOT)))
            if 'ex_div_date' in a: actions.update(a.ex_div_date.astype(str).str[:10])
    contexts=[]
    # Cross-sectional state is built before looking at today's RTH label coverage.
    # A missing future bar may exclude a label, but must not change 09:25 breadth.
    for d in days:
        if d<START or d>END:continue
        p=pre.get(d);n=night.get(d);prev=previous.get(d)
        if prev not in regular or p is None:continue
        cutoff=pd.Timestamp(d+' 09:25');p=p[(p.end<=cutoff)&(p.available<=cutoff+pd.Timedelta(seconds=30))]
        if len(p)<50 or p.end.iloc[-1]!=cutoff or (p.volume>0).sum()<6:continue
        nr=np.nan
        if n is not None:
            n=n[(n.end<=pd.Timestamp(d+' 04:00'))&(n.available<=cutoff+pd.Timedelta(seconds=30))]
            if len(n)>=6 and (n.volume>0).sum()>=3:nr=float(n.close.iloc[-1]/n.open.iloc[0]-1)
        contexts.append({'symbol':symbol,'day':d,'pre_ret':float(p.close.iloc[-1]/p.open.iloc[0]-1),
                         'pre_gap':float(p.close.iloc[-1]/regular[prev].close.iloc[-1]-1),
                         'pre_late':float(p.close.iloc[-1]/p.open.iloc[-max(1,len(p)//6)]-1),'night_ret':nr})
    rows=[]; prehistory=[]; nighthistory=[]
    for d in days:
        if d<'2024-08-01' or d>END:continue
        p=pre.get(d);n=night.get(d);reg=regular.get(d)
        prevol=np.nan if p is None else float(p.volume.sum())
        nightvol=np.nan if n is None else float(n.volume.sum())
        pv=np.median(prehistory[-20:]) if len(prehistory)>=10 else np.nan
        nv=np.median(nighthistory[-20:]) if len(nighthistory)>=10 else np.nan
        if p is not None and len(p)>=50:prehistory.append(prevol)
        if n is not None and len(n)>=6:nighthistory.append(nightvol)
        if d<START:continue
        if reg is None:reasons['no_complete_rth_label']+=1;continue
        prev=previous.get(d)
        if prev not in regular:reasons['missing_previous_official_session']+=1;continue
        if d in actions:reasons['corporate_action_day']+=1;continue
        if d not in past.index or past.loc[d].isna().any():reasons['insufficient_20_prior_complete_days']+=1;continue
        cutoff=pd.Timestamp(d+' 09:25:00');decision=cutoff+pd.Timedelta(seconds=30)
        if p is None or len(p)<50 or p.end.iloc[-1]!=cutoff or (p.volume>0).sum()<6:
            reasons['incomplete_or_inactive_pre']+=1;continue
        if n is None or len(n)<6 or (n.volume>0).sum()<3:
            reasons['missing_or_inactive_night']+=1;continue
        p=p[(p.end<=cutoff)&(p.available<=decision)];n=n[(n.end<=pd.Timestamp(d+' 04:00'))&(n.available<=decision)]
        if len(p)<50 or p.end.iloc[-1]!=cutoff or len(n)<6 or (p.volume>0).sum()<6 or (n.volume>0).sum()<3:reasons['late_or_outside_cutoff']+=1;continue
        anchor=float(regular[prev].close.iloc[-1]);entry=float(reg.open.iloc[0]);rth_close=float(reg.close.iloc[-1])
        if len(reg)<2:reasons['missing_delayed_entry']+=1;continue
        delayed=float(reg.open.iloc[1]);delayed_reg=reg.iloc[1:]
        r=dict(symbol=symbol,day=d,decision_at=decision.isoformat(),feature_end=p.end.iloc[-1].isoformat(),
               max_source_available=max(p.available.max(),n.available.max()).isoformat(),
               reference=entry,pre_reference=float(p.close.iloc[-1]),delayed_reference=delayed,
               h_range=float(past.loc[d,'range']),h_mfe=float(past.loc[d,'mfe']),h_ret=float(past.loc[d,'rth_ret']),
               h_previous_ret=float(df.loc[prev,'rth_ret']),h_previous_range=float(df.loc[prev,'range']),
               **segment(p,'pre',anchor),**segment(n,'night',anchor))
        r['pre_rvol']=float(p.volume.sum())/pv if np.isfinite(pv) and pv>0 else np.nan
        r['night_rvol']=float(n.volume.sum())/nv if np.isfinite(nv) and nv>0 else np.nan
        r['gap_scaled']=r['pre_gap']/max(r['h_range'],1e-6)
        r['mfe']=float(reg.high.max()/entry-1);r['mae']=float(reg.low.min()/entry-1)
        r['close_ret']=rth_close/entry-1;r['label_end']=reg.end.iloc[-1].isoformat()
        r['delayed_mfe']=float(delayed_reg.high.max()/delayed-1)
        r['delayed_mae']=float(delayed_reg.low.min()/delayed-1)
        r['delayed_close_ret']=rth_close/delayed-1
        r['cost_proxy'],r['cost_exit']=first_barrier(reg,entry,target=.03)
        r['n_pre']=len(p);r['n_night']=len(n)
        r['pre_trade_age_minutes']=float((cutoff-p.loc[p.volume>0,'end'].max()).total_seconds()/60)
        r['night_trade_age_minutes']=float((pd.Timestamp(d+' 04:00')-n.loc[n.volume>0,'end'].max()).total_seconds()/60)
        r['night_span_minutes']=float((n.end.max()-n.start.min()).total_seconds()/60)
        rows.append(r)
    # Hour candles are aggregates of actual 5m bars, never native/executable proxies.
    a=f.copy();a['slot']=np.where(a.kind=='regular',(a.minute-570)//60,np.where(a.kind=='pre',(a.minute-240)//60,np.where(a.kind=='post',(a.minute-a.close_minute)//60,a.minute//60)))
    ag=a.groupby(['day','kind','slot'],sort=False).agg(start=('start','min'),end=('end','max'),o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum'),n=('open','size')).sort_values('start')
    candles=[[r.start.strftime('%Y-%m-%dT%H:%M'),r.end.strftime('%Y-%m-%dT%H:%M'),*[round(float(x),4) for x in (r.o,r.h,r.l,r.c,r.v)],r.Index[1],r.n] for r in ag.itertuples() if r.start.strftime('%Y-%m-%d')>=START]
    chart={'daily':[[d,*[round(float(df.loc[d,k]),4) for k in ('open','high','low','close','volume')]] for d in df.index if d>=START], 'hourly':candles,'contexts':contexts}
    coverage=dict(symbol=symbol,status='scored' if rows else 'no_joint_night_pre_samples',eligible=len(rows),
                  complete_rth=len(df.loc[df.index>=START]),rth_years=dict(history_count),reasons=dict(reasons),
                  first_raw=f.start.min().isoformat(),last_raw=f.end.max().isoformat(),
                  corporate_action_sources=action_sources,first_eligible=rows[0]['day'] if rows else None,last_eligible=rows[-1]['day'] if rows else None,**audit)
    return rows,chart,coverage,sources

def main():
    universe_bytes=(OUT/'universe.json').read_bytes();universe=json.loads(universe_bytes)
    calendar_payload=json.loads((HISTORY/'calendar.json').read_text()); calendar={s['session_date']:s for s in calendar_payload['sessions']}
    all_rows=[];charts={};coverage=[];sources=[];contexts=[]
    for symbol in sorted(universe['members']):
        rows,chart,audit,source=build_symbol(symbol,calendar)
        all_rows.extend(rows)
        if chart:
            contexts.extend(chart.pop('contexts',[]));charts[symbol]=chart
        coverage.append(audit);sources.extend(source)
        print(json.dumps({'symbol':symbol,'eligible':len(rows),'complete_rth':audit.get('complete_rth',0)}),flush=True)
    f=pd.DataFrame(all_rows).sort_values(['day','symbol']).reset_index(drop=True)
    context=pd.DataFrame(contexts);qqq=context[context.symbol=='QQQ'].set_index('day')
    for c in ('pre_ret','pre_gap','pre_late','night_ret'):
        f['qqq_'+c]=f.day.map(qqq[c]);f['relative_'+c]=f[c]-f['qqq_'+c]
    broad=context.groupby('day').agg(breadth=('pre_ret',lambda s:float((s>0).mean())),dispersion=('pre_gap','std'),available_pool=('symbol','size'))
    for c in broad.columns:f[c]=f.day.map(broad[c])
    f.to_parquet(OUT/'panel.parquet',index=False,compression='zstd',compression_level=7)
    (OUT/'charts.json').write_text(json.dumps(charts,separators=(',',':'),allow_nan=False))
    audit={'requested_start':START,'requested_end':END,'members':len(universe['members']),'scored_symbols':f.symbol.nunique(),
           'rows':len(f),'first':f.day.min(),'last':f.day.max(),'coverage':coverage,'sources':sources,
           'calendar_sha256':sha(HISTORY/'calendar.json'),'universe_sha256':hashlib.sha256(universe_bytes).hexdigest(),
           'panel_sha256':sha(OUT/'panel.parquet'),'availability_quality':'assumed_end_plus_1s',
           'incomplete_etf_scope':not all(s['status'].startswith('full_issuer_') for s in universe['funds'].values())}
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('coverage','sources')},default=str),flush=True)

if __name__=='__main__':main()
