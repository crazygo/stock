"""Same-clock, locally weighted competing risks from then-visible path prefixes.

No pending trade is called a failure. Price-only learning omits dividend income;
the capital ledger retains actual dividends, costs and settlement.
"""
from pathlib import Path
import sys, math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v27'))
from clock_reliability import (core,np,pd,read,dump,sha,labels,prepare_inputs,base_forecast,
    day_cutoff,restrict_market,development_gate,CLOCKS,tag)

ROUTES=['censored_band','censored_point','terminal_only_band','premarket_only','source_v27','cash','double_cost']
WINDOW=252
MINIMUM=60
REPLICATES=199
BLOCK=10


def tape(market,symbol,at):
    """Index by expected official bars, never bridge a missing/late price bar."""
    day=pd.Timestamp(at.tz_convert(core.ET).date());expiry=core.expiry_day(market,day)
    dates=market.sessions[(market.sessions>=day)&(market.sessions<=expiry)]
    grids=[pd.date_range(market.calendar[d]['open_at'],market.calendar[d]['close_at'],freq='5min',inclusive='right') for d in dates]
    ends=grids[0].append(grids[1:]);ends=ends[ends-pd.Timedelta(minutes=5)>at+pd.Timedelta(seconds=30)]
    frame=market.bars[symbol].reindex(ends)
    valid=(frame.start_at==ends-pd.Timedelta(minutes=5)) & (frame.end_at==ends) & frame.session_type.eq('regular')
    valid &= frame.available_at.notna() & (frame.available_at>=frame.end_at)
    valid &= np.isfinite(frame[['open','high','low','close','share_factor']]).all(axis=1)
    valid &= (frame[['open','high','low','close','share_factor']]>0).all(axis=1)
    assert len(ends)>0 and (ends[0]-pd.Timedelta(minutes=5)-at).total_seconds()<=360
    reasons=np.zeros(len(frame),dtype=int);logs=np.zeros(len(frame))
    if valid.iloc[0]:
        entry=float(frame.open.iloc[0])*(1+core.SLIP)*float(frame.share_factor.iloc[0])
        target=math.ceil(float(frame.open.iloc[0])*(1+core.SLIP)*1.05*100-1e-10)/100*float(frame.share_factor.iloc[0])
        factors=frame.share_factor.to_numpy();stop=np.floor(entry*.94/factors*100+1e-10)/100
        limit=np.ceil(target/factors*100-1e-10)/100
        stopped=valid.to_numpy()&(frame.low.to_numpy()<=stop)
        hit=valid.to_numpy()&~stopped&(frame.high.to_numpy()>=limit)
        reasons[stopped]=2;reasons[hit]=1
        if valid.iloc[-1] and not reasons[-1]:reasons[-1]=3
        fill=np.where(stopped,np.minimum(frame.open.to_numpy(),stop)*(1-core.SLIP),
            np.where(hit,limit,frame.close.to_numpy()*(1-core.SLIP)))*factors
        good=valid.to_numpy();logs[good]=np.log(fill[good]*(1-core.FEE)/(entry*(1+core.FEE)))
    return {'ends':ends,'available':pd.DatetimeIndex(frame.available_at),'valid':valid.to_numpy(),
        'events':reasons,'logs':logs,'expiry':expiry,'day':day,'symbol':symbol}


def observe(t,cutoff):
    ready=t['valid'] & (t['ends']<cutoff) & (t['available']<cutoff)
    gaps=np.flatnonzero(~ready);n=int(gaps[0]) if len(gaps) else len(ready)
    hits=np.flatnonzero(t['events'][:n]);event=int(t['events'][hits[0]]) if len(hits) else 0
    used=int(hits[0])+1 if len(hits) else n
    missing=not event and n<len(ready) and t['ends'][n]<cutoff
    return {'duration':used/len(ready),'event':event,'price_log_return':float(t['logs'][used-1]) if event else None,
        'observed_bars':used,'expected_bars':len(ready),'last_end':t['ends'][used-1] if used else None,
        'release':max(t['available'][:used]) if used else None,
        'censor_reason':None if event else ('missing_or_late_prefix' if missing else 'administrative_cutoff'),
        'expired_missing':bool(missing and t['expiry']<pd.Timestamp(cutoff.tz_convert(core.ET).date()))}


def aj(observations,weights):
    """Batched weighted Aalen-Johansen product-integral with exact tied groups.

    Events at a duration precede censoring at that duration. Distinct causes are
    grouped together, not jittered or incorrectly censored. Returns competing
    cumulative incidences, remaining survival and price-only expected logreturn.
    """
    durations=np.array([o['duration'] for o in observations]);events=np.array([o['event'] for o in observations])
    payoff=np.array([o['price_log_return'] if o['event'] else 0. for o in observations])
    times=np.unique(durations[events>0]);weights=np.atleast_2d(weights)
    if not len(times):return np.zeros((len(weights),3)),np.ones(len(weights)),np.zeros(len(weights))
    risk=weights@(durations[:,None]>=times[None,:]).astype(float)
    at_event=(durations[:,None]==times[None,:])&(events[:,None]>0)
    death=weights@at_event.astype(float)
    hazard=np.divide(death,risk,out=np.zeros_like(death),where=risk>0)
    assert (hazard>=-1e-12).all() and (hazard<=1+1e-12).all()
    survival=np.cumprod(1-np.clip(hazard,0,1),axis=1)
    prior=np.c_[np.ones(len(weights)),survival[:,:-1]]
    scale=np.divide(prior,risk,out=np.zeros_like(risk),where=risk>0)
    cif=np.stack([(scale*(weights@(at_event&(events[:,None]==k)).astype(float))).sum(axis=1) for k in (1,2,3)],axis=1)
    expected=(scale*(weights@(at_event*payoff[:,None]))).sum(axis=1)
    return cif,survival[:,-1],expected


def occurrence_counts(n,at):
    rng=np.random.default_rng(828+at.tz_convert(core.ET).date().toordinal());out=[]
    for _ in range(REPLICATES):
        starts=rng.integers(0,n,size=math.ceil(n/BLOCK))
        indexes=np.concatenate([(s+np.arange(BLOCK))%n for s in starts])[:n]
        out.append(np.bincount(indexes,minlength=n))
    return np.array(out)


def estimate(history,tapes,at,raw,terminal_only=False):
    cutoff=day_cutoff(at)
    ids=[t for t,v in history.items() if t<cutoff and v is not None][-WINDOW:]
    out={'members':ids,'cutoff':cutoff,'terminal_only':terminal_only,'ready':False,'observations':{}}
    if raw is None or len(ids)<MINIMUM:return out
    assert len({t.tz_convert(core.ET).date() for t in ids})==len(ids)
    assert all(t.tz_convert(core.ET).time()==at.tz_convert(core.ET).time() for t in ids)
    X=np.array([history[t] for t in ids]);age=np.arange(len(ids)-1,-1,-1);recency=.5**(age/63.)
    # Kernel coordinates and scale use original emissions and prior dates only.
    mu=np.average(X,axis=0,weights=recency);scale=np.sqrt(np.average((X-mu)**2,axis=0,weights=recency));scale=np.maximum(scale,.001)
    distance=np.sum(((X-np.asarray(raw))/scale)**2,axis=1)
    weights=recency*np.exp(-.5*np.minimum(distance,1400))
    counts=occurrence_counts(len(ids),at);W=np.r_[weights[None,:],counts*weights]
    values=[];risks={};expired=False
    for symbol in core.ETFS:
        obs=[observe(tapes[(t,symbol)],cutoff) for t in ids];out['observations'][symbol]=obs
        expired |= any(o['expired_missing'] for o in obs)
        mask=np.array([bool(o['event']) if terminal_only else True for o in obs]);w=W*mask
        cif,tail,expected=aj(obs,w);effective=w[0].sum()**2/max(float(np.square(w[0]).sum()),1e-300)
        supported=(tail<=1e-10)&(w.sum(axis=1)>1e-20)
        risks[symbol]={'cif':cif[0].tolist(),'unidentified_mass':float(tail[0]),'effective_dates':effective,
            'known_terminals':sum(o['event']>0 for o in obs),'right_censored':sum(o['event']==0 for o in obs),
            'bootstrap_supported':int(supported[1:].sum())}
        if effective<30 or not supported.all():return {**out,'risks':risks,'reason':'insufficient_local_or_horizon_support'}
        values.append(expected)
    values=np.array(values).T;point=values[0];margin=max(0.,float(np.quantile(np.max(values[1:]-point,axis=1),.9,method='higher')))
    out.update(ready=not expired,reason='expired_missing_prefix' if expired else None,
        point=point.tolist(),margin=margin,lower=(point-margin).tolist(),risks=risks,
        kernel_scale=scale.tolist(),weights=weights.tolist(),bootstrap_values=values[1:].tolist())
    return out


def decide(info,band=True):
    vals=info.get('lower' if band else 'point');ready=info['ready']
    j=int(np.argmax(vals)) if ready else 0
    action=core.ETFS[j] if ready and vals[j]>1e-12 else 'CASH'
    return {'ready':ready,'action':action,'allocation':float(action!='CASH'),'values':vals if ready else None}


def events(at,date,clock,estimates,old):
    band=decide(estimates['censored']);point=decide(estimates['censored'],False);terminal=decide(estimates['terminal'],True)
    cash={'ready':True,'action':'CASH','allocation':0.,'values':None}
    items={'censored_band':band,'censored_point':point,'terminal_only_band':terminal,
        'premarket_only':band if clock==CLOCKS[0] else cash,'source_v27':old,'cash':cash,'double_cost':band}
    return {n:{'at':at,'date':date,**e} for n,e in items.items()}
