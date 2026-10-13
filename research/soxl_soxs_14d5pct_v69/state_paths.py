"""Roll current known reference history along jointly sampled source blocks.

No actual future return labels or capital outcomes are read.
"""
from path_learning import np,pd,CLOCKS,PRIOR,core,SOURCE,kernel,price_known_pool,first_kernels,load_panel,PANEL_ROOT,read,sha,restrict_market,quote,horizon


class StateUnavailable(ValueError):pass


def roll_state(history):
    """history: [path,21,symbol], symbols SOXX,QQQ,then8 fixed reference issuers."""
    values=np.asarray(history,float)
    if values.ndim!=3 or values.shape[1:]!=(21,10) or not np.isfinite(values).all() or not (values>0).all():raise StateUnavailable('Complete positive21x10 history required')
    returns=values[:,-1]/values[:,-6]-1
    changes=values[:,1:,0]/values[:,:-1,0]-1
    volatility=np.maximum(np.std(changes,axis=1,ddof=1),1e-4)
    return np.column_stack((np.clip((returns[:,0]-returns[:,1])/(volatility*np.sqrt(5)),-5,5),2*np.mean(returns[:,2:]>0,axis=1)-1,np.log(volatility)))


def query_kernels(source,states):
    """Exactly the original conditional kernel, vectorized over path states."""
    source=np.asarray(source,float);states=np.atleast_2d(states);n=len(source)
    recency=np.exp2((np.arange(n)+1-n)/126);recency/=recency.sum()
    center=np.average(source,axis=0,weights=recency);scale=np.sqrt(np.average((source-center)**2,axis=0,weights=recency));scale[scale<1e-12]=1.
    distance=np.sqrt(np.sum(((source[None,:,:]-states[:,None,:])/scale)**2,axis=2))
    if (distance>1e-12).all():bandwidth=np.quantile(distance,.4,axis=1)
    else:
        bandwidth=[]
        for row in distance:
            positive=row[row>1e-12];bandwidth.append(float(np.quantile(positive,.4)) if len(positive) else 1.)
    probability=recency[None,:]*np.exp(-.5*(distance/np.array(bandwidth)[:,None])**2)
    total=probability.sum(axis=1)
    if not np.isfinite(total).all() or not (total>0).all():raise StateUnavailable('Unsupported path state')
    probability/=total[:,None];effective=1/np.sum(probability**2,axis=1)
    if not np.isfinite(probability).all() or not (effective>=20).all():raise StateUnavailable('A simulated state lacks required effective support')
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
        if not np.isfinite(closing).all() or not (closing>0).all():raise StateUnavailable('Invalid generated reference price')
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
    symbols=['SOXX','QQQ',*read(SOURCE/'protocol.json')['panel']]
    leader_bars,_,manifest=load_panel(market.calendar,PANEL_ROOT)
    known_manifest=read(core.ROOT/'.cache/soxl_soxs_14d5pct_v55/runs/development_20261013/input_manifest.json')['panel'];assert manifest==known_manifest
    bars={s:(market.bars[s] if s in ['SOXX','QQQ'] else leader_bars[s]) for s in symbols};daily={};groups={}
    for s,f in bars.items():
        f=f[(f.end_at<cutoff)&(f.available_at<cutoff)];bars[s]=f;groups[s]={d:g for d,g in f.groupby('date')};daily[s]={}
        for d,g in groups[s].items():
            cal=market.calendar.get(d)
            if cal is None:continue
            expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right');part=g[g.index.isin(expected)]
            if part.index.equals(expected) and part.available_at.notna().all() and part.available_at.ge(part.end_at).all() and part.start_at.eq(part.end_at-pd.Timedelta(minutes=5)).all() and np.isfinite(part.econ_close).all() and part.econ_close.gt(0).all():daily[s][d]=(float(part.econ_close.iloc[-1]),part.available_at.max())
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


def prepare_context(output,market,bank,panel,cutoff):
    reference,bars,daily,symbols=reference_inputs(market,bank,panel,cutoff)
    n=len(panel);queries={'at_ns':panel.index.asi8.copy(),'history':np.full((n,21,10),np.nan),'quotes':np.full((n,10),np.nan),'ready':np.zeros(n,bool),'history_release_ns':np.full(n,pd.NaT.value,dtype='int64'),'quote_release_ns':np.full(n,pd.NaT.value,dtype='int64')}
    for i,at in enumerate(panel.index):
        try:h,q=current_history(market,bars,daily,symbols,at)
        except (AssertionError,KeyError):continue
        if not (np.isfinite(h).all() and (h>0).all() and np.isfinite(q).all() and (q>0).all()):continue
        day=pd.Timestamp(at.tz_convert(core.ET).date());prior=market.sessions[market.sessions<day][-21:];end=at-pd.Timedelta(seconds=30)
        queries['history'][i]=h;queries['quotes'][i]=q;queries['ready'][i]=True
        queries['history_release_ns'][i]=max(daily[s][d][1] for s in symbols for d in prior).value
        queries['quote_release_ns'][i]=max(bars[s].loc[end,'available_at'] for s in symbols).value
    np.savez_compressed(output/'reference_bank.npz',**reference);np.savez_compressed(output/'reference_queries.npz',**queries)


def load_context(folder):
    from pathlib import Path
    folder=Path(folder);reference=dict(np.load(folder/'reference_bank.npz'));queries=dict(np.load(folder/'reference_queries.npz'))
    return {'reference':reference,'queries':queries,'positions':{int(ns):i for i,ns in enumerate(queries['at_ns'])}}


def input_status(context,bank,members,at,clock):
    from path_learning import day_cutoff
    if context is None or at.value not in context['positions']:return {'ready':False,'reason':'missing_reference_context'}
    ref=context['reference'];q=context['queries'];i=context['positions'][at.value];ids=np.unique(bank['blocks'][members]);first=bank['blocks'][members,0];cut=day_cutoff(at).value;c=CLOCKS.index(clock)
    if not q['ready'][i] or q['history_release_ns'][i]==pd.NaT.value or q['history_release_ns'][i]>=cut or q['quote_release_ns'][i]==pd.NaT.value or q['quote_release_ns'][i]>at.value:return {'ready':False,'reason':'incomplete_current_history_or_quote'}
    arrays=[q['history'][i],q['quotes'][i],ref['regular_close'][ids],ref['previous_close'][ids],ref['quotes'][first,c]]
    if not all(np.isfinite(a).all() and (a>0).all() for a in arrays) or not np.all((ref['release_ns'][ids]!=pd.NaT.value)&(ref['release_ns'][ids]<cut)):return {'ready':False,'reason':'reference_support_missing'}
    return {'ready':True,'query_index':i,'source_days':ids.tolist(),'history_release_ns':int(q['history_release_ns'][i]),'quote_release_ns':int(q['quote_release_ns'][i]),'latest_reference_release_ns':int(ref['release_ns'][ids].max())}


def integrate_dynamic(bank,members,first,at,clock,current_raw,plan,context,count=2048,replication=0):
    from path_learning import simulate,payoff_moments,summarize_moments
    from sampling import uniforms
    if len(plan['dates'])<=2:raise ValueError('Original horizon must extend beyond the partition window')
    if count%8 or count//8<2 or (count//8)&(count//8-1):raise ValueError('Eight power-of-two scrambles required')
    weights=np.asarray(first,float);short_plan={**plan,'dates':plan['dates'][:2],'bar_counts':plan['bar_counts'][:2],'expiry_at':None}
    short=simulate(bank,members[:,None],clock,current_raw,short_plan);resolved=np.all(short['reason']!=3,axis=1);unresolved=~resolved
    exact=weights[resolved]@payoff_moments(short['log_returns'][resolved]);mass=float(weights[unresolved].sum()) if resolved.any() else 1.
    assert abs(float(weights.sum())-1)<1e-12 and -1e-12<=mass<=1+1e-12
    seed=1062+pd.Timestamp(at.tz_convert(core.ET).date()).toordinal()*16+CLOCKS.index(clock)
    if mass>0:
        conditioned=np.where(unresolved,weights/mass,0.);points=uniforms(seed,(len(plan['dates'])+1)//2,count,replication);i=context['positions'][at.value];q=context['queries']
        trace=dynamic_blocks(members,conditioned,points,q['history'][i],q['quotes'][i],context['reference'],bank,clock,plan)
        selected=trace['selected_blocks'];state_path={k:trace[k] for k in ['states_before_blocks','effective_dates']}
        tail=simulate(bank,selected,clock,current_raw,plan);mean=exact+mass*payoff_moments(tail['log_returns']).mean(axis=0)
        means=exact[None,:2]+mass*tail['log_returns'].reshape(8,count//8,2).mean(axis=1);covariance=np.cov(means,rowvar=False,ddof=1)/8
    else:selected=np.empty((0,(len(plan['dates'])+1)//2),int);tail=None;state_path=None;mean=exact;means=np.tile(exact[:2],(8,1));covariance=np.zeros((2,2))
    raw=summarize_moments(mean);frequencies={str(i):(weights[resolved]@np.equal(short['reason'][resolved],i).astype(float)+mass*(np.mean(tail['reason']==i,axis=0) if tail is not None else np.zeros(2))).tolist() for i in [1,2,3]}
    integration={'method':'exact_absorption_plus_dynamic_eight_scrambles','replicates':8,'points_per_replicate':len(selected)//8,'replicate_means':means.tolist(),'mean_covariance':covariance.tolist(),'standard_error':np.sqrt(np.diag(covariance)).tolist()}
    part={'seed':seed,'enumerated_source_blocks':len(members),'resolved_members':members[resolved].tolist(),'unresolved_members':members[~resolved].tolist(),'unresolved_probability':mass,'exact_moments':exact.tolist(),'estimated_moments':mean.tolist()}
    return {'selected_blocks':selected,'result':tail,'partition_result':short,'resolved':resolved,'state_path':state_path},part,raw,integration,frequencies
