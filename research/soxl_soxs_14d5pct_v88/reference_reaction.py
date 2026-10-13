"""Scalar reconstruction from clipped native rows, independent of reaction_inputs."""
from pathlib import Path
from datetime import datetime,timedelta
from functools import lru_cache
import json,math,hashlib
import numpy as np
import pandas as pd

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[1]
SYMBOLS=['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']
ET='America/New_York'


@lru_cache(maxsize=1)
def raw_sources():
    manifest=json.loads((SOURCE/'evidence/price_inputs.json').read_text());raw={};actions={}
    for s,r in manifest['symbols'].items():
        d=pd.read_parquet(ROOT/r['daily_path']);raw[s]={x['time_key'][:10]:x for x in d.to_dict('records')}
        actions[s]=pd.read_parquet(ROOT/r['actions_path']).to_dict('records')
    return raw,actions,manifest['calendar']['sessions']


def day_return(symbol,day,previous,raw,actions):
    q=raw[symbol];a=q.get(previous);b=q.get(day)
    if a is None or b is None:raise ValueError('Missing mature close')
    if not math.isfinite(a['close']) or not math.isfinite(b['close']) or min(a['close'],b['close'])<=0:
        raise ValueError('Invalid mature close')
    shares=1.;cash=0.
    for event in actions[symbol]:
        if event['ex_div_date']!=day:continue
        ratio=event.get('split_ratio');dividend=event.get('per_cash_div')
        if pd.notna(ratio):shares/=float(ratio)
        if pd.notna(dividend):cash+=float(dividend)
    return math.log((shares*b['close']+cash)/a['close'])


def economic_volume(symbol,day,raw,actions):
    row=raw[symbol].get(day)
    if row is None or not math.isfinite(row['volume']) or row['volume']<=0:
        raise ValueError('Missing mature volume')
    v=float(row['volume'])
    for a in actions[symbol]:
        if a['ex_div_date']<=day and pd.notna(a.get('split_ratio')):v*=float(a['split_ratio'])
    return v


def reference(previous_days,at,raw,actions,sessions):
    now=pd.Timestamp(at).tz_convert(ET);dates=[x['session_date'] for x in sessions]
    observed=[];pending=[]
    for symbol,event in previous_days:
        later=[(i,d) for i,d in enumerate(dates) if d>event]
        if not later:raise ValueError('Missing next calendar session')
        pos,day=later[0];available=pd.Timestamp(str(datetime.strptime(day,'%Y-%m-%d').date()+timedelta(days=1)),tz=ET)
        if now<available:
            pending.append([symbol,event,day]);continue
        if pos<20:raise ValueError('Missing past twenty sessions')
        prior=dates[pos-1];history=dates[pos-20:pos]
        market=day_return('SPY',day,prior,raw,actions)
        own=day_return(symbol,day,prior,raw,actions)-market
        peer=sum(day_return(s,day,prior,raw,actions)-market for s in SYMBOLS if s!=symbol)/7
        denominator=sum(economic_volume(symbol,d,raw,actions) for d in history)/20
        volume=math.log(economic_volume(symbol,day,raw,actions)/denominator)
        observed.append(dict(symbol=symbol,planned_date=event,reaction_date=day,base_date=prior,
            available_at=available.isoformat(),volume_history=history,issuer_excess=own,peer_excess=peer,log_volume_ratio=volume))
    n=len(observed);values=[n]+[sum(x[key] for x in observed)/n if n else 0. for key in ['issuer_excess','peer_excess','log_volume_ratio']]
    return dict(values=values,observed=observed,pending=pending)


def verify_reaction_table(table,panel):
    raw,actions,sessions=raw_sources();numeric=0;event_refs=0;unique=set();pending_refs=0;first=None;last=None
    fields=['reaction_count','issuer_reaction','peer_reaction','volume_reaction']
    for r in table.itertuples():
        at=pd.Timestamp(r.at);expected=reference(json.loads(r.previous_days),at,raw,actions,sessions)
        np.testing.assert_allclose([getattr(r,k) for k in fields],expected['values'],rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(panel.loc[at,fields].to_numpy(float),expected['values'],rtol=1e-11,atol=1e-13)
        actual=json.loads(r.reaction_observed);assert len(actual)==len(expected['observed'])
        assert json.loads(r.reaction_pending)==expected['pending'] and r.reaction_pending_count==len(expected['pending'])
        pending_refs+=len(expected['pending']);numeric+=8
        for a,b in zip(actual,expected['observed']):
            for k in ['symbol','planned_date','reaction_date','base_date','available_at','volume_history']:assert a[k]==b[k],(at,k)
            for k in ['issuer_excess','peer_excess','log_volume_ratio']:np.testing.assert_allclose(a[k],b[k],rtol=1e-11,atol=1e-13);numeric+=1
            assert pd.Timestamp(b['available_at'])<=at
            unique.add((b['symbol'],b['planned_date']));event_refs+=1
        first=at if first is None else first;last=at
    return dict(queries=len(table),numeric_checks=numeric,observed_event_references=event_refs,
        unique_observed_issuer_days=len(unique),pending_event_references=pending_refs,
        first_query=str(first),last_query=str(last),no_future_response_used=True)
