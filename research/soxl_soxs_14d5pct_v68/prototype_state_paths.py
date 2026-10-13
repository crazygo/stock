"""Numerical prototype only: roll current reference-price history along joint blocks.

Not part of v68 forecasts or its frozen capital replay. No return labels or ledger read.
"""
from path_learning import np,pd,CLOCKS,PRIOR,core,kernel,price_known_pool,first_kernels,load_panel,PANEL_ROOT,read,sha,restrict_market,quote,horizon


def roll_state(history):
    """history: [path,21,symbol], symbols SOXX,QQQ,then8 fixed reference issuers."""
    values=np.asarray(history,float)
    if values.ndim!=3 or values.shape[1:]!=(21,10) or not np.isfinite(values).all() or not (values>0).all():raise ValueError('Complete positive21x10 history required')
    returns=values[:,-1]/values[:,-6]-1
    changes=values[:,1:,0]/values[:,:-1,0]-1
    volatility=np.maximum(np.std(changes,axis=1,ddof=1),1e-4)
    return np.column_stack((np.clip((returns[:,0]-returns[:,1])/(volatility*np.sqrt(5)),-5,5),2*np.mean(returns[:,2:]>0,axis=1)-1,np.log(volatility)))


def query_kernels(source,states):
    """Exactly the original conditional kernel, vectorized over path states."""
    source=np.asarray(source,float);states=np.atleast_2d(states);n=len(source)
    recency=np.exp2((np.arange(n)+1-n)/126);recency/=recency.sum()
    center=np.average(source,axis=0,weights=recency);scale=np.sqrt(np.average((source-center)**2,axis=0,weights=recency));scale[scale<1e-12]=1.
    distance=np.sqrt(np.sum(((source[None,:,:]-states[:,None,:])/scale)**2,axis=2));bandwidth=[]
    for row in distance:
        positive=row[row>1e-12];bandwidth.append(float(np.quantile(positive,.4)) if len(positive) else 1.)
    probability=recency[None,:]*np.exp(-.5*(distance/np.array(bandwidth)[:,None])**2)
    total=probability.sum(axis=1)
    if not np.isfinite(total).all() or not (total>0).all():raise ValueError('Unsupported path state')
    probability/=total[:,None];effective=1/np.sum(probability**2,axis=1)
    if not np.isfinite(probability).all() or not (effective>=20).all():raise ValueError('A simulated state lacks required effective support')
    return probability,effective


def advance(history,selected,block_number,reference,price_bank,clock,quotes,plan):
    """Append actual-template closes, preserving current rolling21day history."""
    values=history.copy();n=len(values);c=CLOCKS.index(clock);daily=[]
    for offset in range(2):
        day=block_number*2+offset
        if day>=len(plan['dates']):break
        ids=price_bank['blocks'][selected,offset];last=plan['bar_counts'][day]-1
        anchor=reference['quotes'][ids,c] if day==0 else reference['previous_close'][ids]
        known=quotes if day==0 else values[:,-1]
        closing=reference['regular_close'][ids,last]*(known/anchor)
        if not np.isfinite(closing).all() or not (closing>0).all():raise ValueError('Invalid generated reference price')
        values=np.concatenate((values[:,1:],closing[:,None,:]),axis=1);daily.append(closing)
    return values,daily


def dynamic_blocks(members,first,uniforms,initial,quotes,reference,price_bank,clock,plan):
    n=len(uniforms);steps=(len(plan['dates'])+1)//2;assert uniforms.shape==(n,steps)
    history=np.broadcast_to(initial,(n,21,10)).copy();q=np.broadcast_to(quotes,(n,10));states=[];effective=[];selected=[];closes=[];local=[]
    source=price_bank['states'][price_bank['blocks'][members,0],0,:3]
    for step in range(steps):
        state=roll_state(history);states.append(state)
        if step==0:probability=np.broadcast_to(first,(n,len(members)));eff=np.full(n,1/np.sum(np.asarray(first)**2))
        else:probability,eff=query_kernels(source,state)
        cdf=np.cumsum(probability,axis=1);cdf[:,-1]=1.;index=np.sum(cdf<=uniforms[:,step,None],axis=1);chosen=np.asarray(members)[index]
        selected.append(chosen);local.append(index);effective.append(eff);history,dayclose=advance(history,chosen,step,reference,price_bank,clock,q,plan);closes.extend(dayclose)
    return {'selected_blocks':np.stack(selected,axis=1),'local_indices':np.stack(local,axis=1),'states_before_blocks':np.stack(states,axis=1),'effective_dates':np.stack(effective,axis=1),'generated_daily_closes':np.stack(closes,axis=1),'final_history':history}


def reference_inputs(market,bank,panel,cutoff):
    symbols=['SOXX','QQQ',*read(core.ROOT/'research/soxl_soxs_14d5pct_v68/protocol.json')['panel']]
    leader_bars,_,manifest=load_panel(market.calendar,PANEL_ROOT)
    known_manifest=read(core.ROOT/'.cache/soxl_soxs_14d5pct_v55/runs/development_20261013/input_manifest.json')['panel'];assert manifest==known_manifest
    bars={s:(market.bars[s] if s in ['SOXX','QQQ'] else leader_bars[s]) for s in symbols};daily={};groups={}
    for s,f in bars.items():
        f=f[(f.end_at<cutoff)&(f.available_at<cutoff)];bars[s]=f;groups[s]={d:g for d,g in f.groupby('date')};daily[s]={}
        for d,g in groups[s].items():
            cal=market.calendar.get(d)
            if cal is None:continue
            expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right');part=g[g.index.isin(expected)]
            if part.index.equals(expected) and part.available_at.notna().all() and part.available_at.ge(part.end_at).all() and np.isfinite(part.econ_close).all() and part.econ_close.gt(0).all():daily[s][d]=(float(part.econ_close.iloc[-1]),part.available_at.max())
    n=len(bank['date_ns']);reference={'regular_close':np.full((n,78,10),np.nan),'quotes':np.full((n,8,10),np.nan),'previous_close':np.full((n,10),np.nan),'release_ns':np.full(n,pd.NaT.value,dtype='int64')};sessions=list(market.sessions);positions={d:i for i,d in enumerate(sessions)}
    for i,ns in enumerate(bank['date_ns']):
        day=pd.Timestamp(ns);j=positions[day];prior=sessions[j-1] if j else None;cal=market.calendar[day];expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right')
        if len(expected)!=78:continue
        releases=[]
        for k,s in enumerate(symbols):
            if day not in daily[s] or prior not in daily[s]:break
            g=groups[s][day];reference['regular_close'][i,:,k]=g.loc[expected,'econ_close'];reference['previous_close'][i,k]=daily[s][prior][0];releases.extend([daily[s][day][1],daily[s][prior][1]])
            for c,clock in enumerate(CLOCKS):
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
                if end in g.index:
                    z=g.loc[end]
                    if pd.notna(z.available_at) and end<=z.available_at<=at and z.start_at==end-pd.Timedelta(minutes=5):reference['quotes'][i,c,k]=z.econ_close;releases.append(z.available_at)
        if len(releases) and np.isfinite(reference['regular_close'][i]).all() and np.isfinite(reference['previous_close'][i]).all():reference['release_ns'][i]=max(releases).value
    return reference,bars,daily,symbols


def current_history(market,bars,daily,symbols,at):
    day=pd.Timestamp(at.tz_convert(core.ET).date());prior=list(market.sessions[market.sessions<day][-21:]);cutoff=day.tz_localize(core.ET).tz_convert('UTC');end=at-pd.Timedelta(seconds=30)
    assert len(prior)==21 and all(d in daily[s] and daily[s][d][1]<cutoff for s in symbols for d in prior)
    history=np.array([[daily[s][d][0] for s in symbols] for d in prior]);quotes=[]
    for s in symbols:
        z=bars[s].loc[end];assert pd.notna(z.available_at) and end<=z.available_at<=at and z.start_at==end-pd.Timedelta(minutes=5);quotes.append(float(z.econ_close))
    return history,np.array(quotes)
