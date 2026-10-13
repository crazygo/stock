"""Verify merged prior histories and fixed hypothetical contracts, not a strategy."""
import datetime, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from market_extension import load, old, ROOT, CACHE, sha
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v3'))
from risk import risk_label
DATES=['2023-06-15','2023-07-03','2023-09-19','2024-04-05','2024-06-25','2024-10-11','2026-04-30']
TRADED=['SOXL','SOXS','TQQQ','SQQQ','UPRO','SPXU']
CLOCKS=['09:25:30','10:05:30','10:35:30','11:05:30','12:05:30','13:05:30','14:05:30','15:05:30']


def independent(m,s,at):
    # Raw OHLC, explicit clock bounds and split ratios, independent of econ_ columns.
    frame=m.bars[s];day=at.tz_convert('America/New_York').tz_localize(None).normalize()
    entry=(at+pd.Timedelta(seconds=30)).floor('5min')+pd.Timedelta(minutes=5)
    rows=frame[frame.start_at.eq(entry)]
    assert len(rows)==1
    first=rows.iloc[0];fill=float(first.open)*1.0005
    sf=lambda d:math.prod(float(a['value']) for a in m.actions[s] if a['type']=='Splits' and pd.Timestamp(a['date'])<=d)
    initial=fill*sf(day);target=math.ceil(fill*1.05*100-1e-10)/100*sf(day)
    expiry=max(d for d in m.sessions if d<=day+pd.Timedelta(days=13));end=pd.Timestamp(m.calendar[expiry]['close_at'])
    observed=frame[(frame.start_at>=entry)&(frame.end_at<=end)]
    result=None;path=[];factors=set()
    for d in m.sessions[(m.sessions>=day)&(m.sessions<=expiry)]:
        c=m.calendar[d];g=observed[(observed.start_at>=pd.Timestamp(c['open_at']))&(observed.end_at<=pd.Timestamp(c['close_at']))]
        for b in g.itertuples():
            factor=sf(d);factors.add(factor);path.append(b)
            stop=math.floor(initial*.94/factor*100+1e-10)/100
            limit=math.ceil(target/factor*100-1e-10)/100
            if b.low<=stop: result=('risk_stop',b.end_at,min(b.open,stop)*.9995*factor);break
            if b.high>=limit: result=('target',b.end_at,limit*factor);break
        if result:break
    full=all(m.day_complete[s].get(d,False) for d in m.sessions[(m.sessions>=day)&(m.sessions<=expiry)])
    if result is None:
        if full and path and path[-1].end_at==end:result=('expiry',end,path[-1].close*.9995*sf(expiry))
        else:return dict(reason='pending',complete_window=full,entry_at=entry,entry_fill_raw=fill)
    reason,exit_at,value=result;exitday=exit_at.tz_convert('America/New_York').tz_localize(None).normalize()
    div=sum(float(a['value'])*sf(pd.Timestamp(a['date'])) for a in m.actions[s]
            if a['type']=='CashDividends' and day<pd.Timestamp(a['date'])<=exitday)
    return dict(reason=reason,complete_window=full,entry_at=entry,entry_fill_raw=fill,exit_at=exit_at,
        exit_fill_economic=value,dividend_per_economic_unit=div,net_return=(value*.9995+div)/(initial*1.0005)-1,
        split_crossed=len(factors)>1)


def main():
    m=load(for_audit=True);base=old.load_market();values=0;results=[];oldchecks=0
    for s in m.bars:
        after=m.bars[s].loc[base.bars[s].index]
        cols=['time_key','start_at','end_at','available_at','open','high','low','close','volume','turnover']
        assert after[cols].equals(base.bars[s][cols])
        oldchecks+=len(after)*len(cols)
    for s in TRADED:
        for day in DATES:
            for clock in CLOCKS:
                at=pd.Timestamp(day+' '+clock,tz='America/New_York').tz_convert('UTC')
                if at>=pd.Timestamp(m.calendar[pd.Timestamp(day)]['close_at']):continue
                actual=risk_label(m,s,at,{'id':'fixed6','stop_fraction':.06,'breakeven_trigger':None})
                expected=independent(m,s,at)
                for k in ['reason','entry_at','complete_window']:
                    assert actual[k]==expected[k],(s,at,k,actual[k],expected[k]);values+=1
                for k in ['entry_fill_raw','exit_at','exit_fill_economic','dividend_per_economic_unit','net_return']:
                    if k not in expected:continue
                    if k=='exit_at':assert actual[k]==expected[k]
                    else:assert math.isclose(float(actual[k]),float(expected[k]),rel_tol=1e-11,abs_tol=1e-11),(s,at,k)
                    values+=1
                results.append(dict(symbol=s,at=str(at),**expected))
    # Verify old labels survive the arbitrary economic-unit change caused by earlier splits.
    oldlabels=0
    for s in TRADED:
        for day in ['2024-11-06','2025-03-25','2025-11-19','2026-04-30']:
            for clock in CLOCKS:
                at=pd.Timestamp(day+' '+clock,tz='America/New_York').tz_convert('UTC')
                args={'id':'fixed6','stop_fraction':.06,'breakeven_trigger':None}
                a=risk_label(m,s,at,args);b=risk_label(base,s,at,args)
                for k in ['reason','entry_at','exit_at','complete_window','entry_fill_raw']:
                    assert a.get(k)==b.get(k),(s,at,k)
                av,bv=a.get('net_return',np.nan),b.get('net_return',np.nan)
                assert (pd.isna(av) and pd.isna(bv)) or math.isclose(av,bv,rel_tol=1e-11,abs_tol=1e-11)
                oldlabels+=1
    report=dict(passed=True,raw_prior_values_preserved=oldchecks,fixed_contracts=len(results),numeric_and_state_comparisons=values,
        old_contracts_preserved=oldlabels,contracts=results,dates=DATES,
        new_model_trained=False,new_strategy_evaluated=False,goal_complete=False,
        source_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(__file__).with_name('market_extension.py')),
        data_audit_sha256=sha(CACHE/'data_audit.json'),created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (CACHE/'extension_audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print({k:v for k,v in report.items() if k!='contracts'},flush=True)


if __name__=='__main__':main()
