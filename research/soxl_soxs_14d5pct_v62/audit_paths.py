"""Independent closed-form path scaling, earliest barrier and cash prefix valuation."""
from path_learning import core,np,pd,CLOCKS,FIELDS,PRIOR,DRAWS,day_cutoff
from verify_reliability import close
from scipy.stats import qmc
from functools import lru_cache


def reference_values(bank,selected,clock,current_raw,plan):
    n=len(selected);c=CLOCKS.index(clock);nd=len(plan['dates']);p=np.arange(n)[:,None];s=np.arange(2)[None,:]
    prior=np.broadcast_to(current_raw,(n,2));factors=np.ones((n,2));dayhit=[];fills=[];types=[];offsets=[];cashdays=[];entry=None
    # Compute every day's candidate independently of whether earlier barriers hit.
    for d,length in enumerate(plan['bar_counts']):
        ids=bank['blocks'][selected[:,d//2],d%2];begin=plan['first_offset'] if d==0 else 0
        # Preserve the simulator's canonical float64 price construction.
        # Dividing each OHLC first can move an exact cent barrier one ULP below it.
        denominator=bank['quotes'][ids,c] if d==0 else bank['previous_close'][ids]
        bars=bank['ohlc'][ids,begin:length]*(prior/denominator)[:,None,:,None]
        cashdays.append(np.zeros((n,2)) if d==0 else bank['dividend_ratio'][ids]*prior)
        if d==0:entry=bars[:,0,:,0]*1.0005;fixed_target=np.ceil(entry*1.05*100-1e-10)/100
        else:factors=factors*bank['split_ratio'][ids]
        threshold=np.floor((entry*.94/factors)*100+1e-10)/100*factors
        target=np.ceil((fixed_target/factors)*100-1e-10)/100*factors
        stops=bars[:,:,:,2]<=threshold[:,None,:];targets=bars[:,:,:,1]>=target[:,None,:]
        idx=np.min(np.where(stops|targets,np.arange(length-begin)[None,:,None],length-begin),axis=1)
        exists=idx<length-begin;safe=np.minimum(idx,length-begin-1);st=stops[p,safe,s]
        price=np.where(st,np.minimum(bars[p,safe,s,0],threshold)*.9995,target)
        dayhit.append(exists);fills.append(price);types.append(np.where(st,2,1));offsets.append(idx+begin);prior=bars[:,-1,:,3]
    occurred=np.stack(dayhit);first=np.min(np.where(occurred,np.arange(nd)[:,None,None],nd),axis=0);anyhit=first<nd;j=np.minimum(first,nd-1)
    fill=np.where(anyhit,np.stack(fills)[j,p,s],prior*.9995);reason=np.where(anyhit,np.stack(types)[j,p,s],3);bar=np.where(anyhit,np.stack(offsets)[j,p,s],plan['bar_counts'][-1]-1)
    cash=(np.stack(cashdays)*(np.arange(nd)[:,None,None]<=j[None,:,:])).sum(axis=0)
    net=(fill*.9995+cash)/(entry*1.0005)-1
    return {'net':net,'log_returns':np.log1p(net),'entry':entry,'fill':fill,'cash':cash,'reason':reason,'exit_day':j,'exit_bar':bar}


def reference_kernel(x,current,conditional):
    n=len(x);w=np.exp2((np.arange(n)+1-n)/126);w=w/sum(w);mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd=np.where(sd<1e-12,1.,sd)
    dist=np.linalg.norm((x-current)/sd,axis=1);positive=sorted(dist[dist>1e-12]);h=float(np.quantile(positive,.4)) if positive else 1.
    q=w*np.exp(-dist**2/(2*h*h)) if conditional else w;q=q/sum(q)
    return {'ready':bool(1/sum(q*q)>=20),'center':mu,'scale':sd,'distance':dist,'bandwidth':h,'recency':w,'probability':q,'effective_dates':1/sum(q*q)}


@lru_cache(maxsize=16)
def reference_transition(xbytes,ybytes):
    x=np.frombuffer(xbytes,dtype='float64').reshape(-1,3);y=np.frombuffer(ybytes,dtype='float64').reshape(-1,3)
    records=[reference_kernel(x,z,True) for z in y]
    return np.array([z['probability'] for z in records]),np.array([z['effective_dates'] for z in records]),all(z['ready'] for z in records)


def check_forecast(bank,market,panel,at,kind,info,model,count=DRAWS):
    c=CLOCKS.index(panel.loc[at,'clock']);members=[];cutoff=day_cutoff(at);today=pd.Timestamp(at.tz_convert(core.ET).date())
    for i,(a,b) in enumerate(bank['blocks']):
        if max(bank['release_ns'][a],bank['release_ns'][b])>=cutoff.value or bank['date_ns'][b]>=today.value:continue
        if not (np.isfinite(bank['quotes'][a,c]).all() and np.isfinite(bank['states'][a,c]).all() and np.isfinite(bank['states'][a,0,:3]).all()):continue
        if not np.isfinite(bank['successor_states'][i]).all() or bank['successor_at_ns'][i]==pd.NaT.value or bank['successor_at_ns'][i]>=cutoff.value:continue
        members.append(i)
    assert info['members']==members and pd.Timestamp(info['cutoff'])==cutoff
    assert list(pd.to_datetime(info['dates']))==[pd.Timestamp(bank['date_ns'][bank['blocks'][i,0]]) for i in members]
    raw=[];qrec=[];end=at-pd.Timedelta(seconds=30)
    for symbol in core.ETFS:
        b=market.bars[symbol]
        if end not in b.index:break
        z=b.loc[end]
        if not (pd.notna(z.available_at) and end<=z.available_at<=at and z.start_at==end-pd.Timedelta(minutes=5) and np.isfinite(z.close) and z.close>0):break
        raw.append(z.close);qrec.append(z)
    deadline=today+pd.Timedelta(days=13);dates=[d for d in sorted(market.calendar) if today<=d<=deadline];first=market.calendar[today]['open_at'];allstarts=pd.date_range(first,market.calendar[today]['close_at'],freq='5min',inclusive='left');nextstarts=allstarts[allstarts>at+pd.Timedelta(seconds=30)];plan=None
    if len(nextstarts) and (nextstarts[0]-at).total_seconds()<=360:
        plan={'dates':dates,'bar_counts':[int((pd.Timestamp(market.calendar[d]['close_at'])-pd.Timestamp(market.calendar[d]['open_at'])).total_seconds()/300) for d in dates],'first_offset':list(allstarts).index(nextstarts[0]),'entry_at':nextstarts[0],'expiry_at':pd.Timestamp(market.calendar[dates[-1]]['close_at'])}
    if plan is None:assert info['plan'] is None
    else:
        assert list(pd.to_datetime(info['plan']['dates']))==plan['dates'] and info['plan']['bar_counts']==plan['bar_counts'] and info['plan']['first_offset']==plan['first_offset']
        for f in ['entry_at','expiry_at']:assert pd.Timestamp(info['plan'][f])==plan[f]
    if len(raw)<2:assert info['query_quote'] is None
    else:
        for j,z in enumerate(qrec):
            rec=info['query_quote'][j];assert rec['symbol']==core.ETFS[j] and pd.Timestamp(rec['at'])==end and pd.Timestamp(rec['available_at'])==z.available_at and rec['raw_close']==z.close and rec['economic_close']==z.econ_close
    ready=len(members)>=120 and len(raw)==2 and plan is not None and panel.loc[at,'feature_ready'] and np.isfinite(panel.loc[at,FIELDS].to_numpy(float)).all()
    if not ready:
        assert not info['ready'] and model is None and info['raw']=={};return None,0
    ids=bank['blocks'][members,0];ks=[reference_kernel(bank['states'][ids,c],panel.loc[at,FIELDS].to_numpy(float),kind!='unconditional'),reference_kernel(bank['states'][ids,0,:3],panel.loc[at,PRIOR].to_numpy(float),kind!='unconditional')]
    for name,k in zip(['first','later'],ks):
        assert info['kernels'][name]['ready']==k['ready']
        for f,v in k.items():
            if f!='ready':close(info['kernels'][name][f],v)
    matrix,effective,ok=reference_transition(bank['states'][ids,0,:3].astype('float64').tobytes(),bank['successor_states'][members].astype('float64').tobytes())
    saved=info['transition'];assert saved['ready']==ok;close(saved['minimum_effective_dates'],min(effective));close(saved['next_state_point'],np.array(ks[0]['probability'])@bank['successor_states'][members])
    if not all(k['ready'] for k in ks) or not ok:assert not info['ready'] and model is None;return None,0
    assert info['ready'] and info['paths']==count
    seed=1062+today.toordinal()*16+c;assert info['seed']==seed
    assert info['replication']==0
    children=np.random.SeedSequence([seed,0]).spawn(8);m=int(np.log2(count//8));assert 8*2**m==count
    u=np.vstack([qmc.Sobol(d=(len(dates)+1)//2,scramble=True,bits=30,rng=np.random.default_rng(child)).random_base2(m) for child in children]);draws=[]
    cdf=np.cumsum(ks[0]['probability']);cdf[-1]=1.;local=np.searchsorted(cdf,u[:,0],side='right');draws=[np.array(members)[local]]
    for j in range(1,u.shape[1]):
        if kind=='conditional':
            nxt=np.empty(count,dtype=int)
            for previous in np.unique(local):
                indices=np.flatnonzero(local==previous);cdf=np.cumsum(matrix[previous]);cdf[-1]=1.
                nxt[indices]=np.searchsorted(cdf,u[indices,j],side='right')
            local=nxt
        else:
            cdf=np.cumsum(ks[1]['probability']);cdf[-1]=1.;local=np.searchsorted(cdf,u[:,j],side='right')
        draws.append(np.array(members)[local])
    selected=np.array(draws).T;np.testing.assert_array_equal(model['selected_blocks'],selected)
    values=reference_values(bank,selected,CLOCKS[c],np.array(raw),plan)
    for f,v in values.items():
        if f in ['reason','exit_day','exit_bar']:np.testing.assert_array_equal(model['result'][f],v)
        else:np.testing.assert_allclose(model['result'][f],v,rtol=1e-10,atol=1e-11)
    logs=values['log_returns'];out=logs.mean(axis=0);direction=(logs[:,0]-logs[:,1])/2;m=np.mean(abs(direction));q=sum(abs(direction[direction>0]))/sum(abs(direction)) if m>0 else .5
    expected={'C':np.mean(out),'D':(out[0]-out[1])/2,'M':m,'q':q,'SOXL':out[0],'SOXS':out[1]}
    for f,v in expected.items():close(info['raw'][f],v)
    repmeans=np.array([logs[i*count//8:(i+1)*count//8].mean(axis=0) for i in range(8)]);center=repmeans.mean(axis=0);cov=sum(np.outer(v-center,v-center) for v in repmeans)/(8*7)
    z=info['integration'];assert z['method']=='eight_independent_scrambles' and z['replicates']==8 and z['points_per_replicate']==count//8
    close(z['replicate_means'],repmeans);close(z['mean_covariance'],cov);close(z['standard_error'],np.sqrt(np.diag(cov)));close(info['monte_carlo_se'],z['standard_error'])
    for i in [1,2,3]:close(info['reason_frequencies'][str(i)],np.mean(values['reason']==i,axis=0))
    return out,count*2
