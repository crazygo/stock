"""Conditional paired two-session price blocks and original contract valuation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v59'))
import innovation_learning as previous
from innovation_learning import (core,np,pd,read,dump,sha,labels,day_cutoff,restrict_market,development_gate,tag,registry,fit_day,issue,decide,CLOCKS,OLD_CLOCKS,load_sources,load_panel,PANEL_ROOT,original_fit,old55_columns)
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
PRIOR=['relative_5','breadth_5','log_sector_volatility'];FIELDS=PRIOR+['sector_premarket'];ABSOLUTE='sector_5'
VARIANTS=['conditional','fixed','unconditional'];ROUTES=['conditional_band','conditional_point','unconditional_band','fixed_band','source_v56','source_v55','premarket_only','cash','double_cost']
DRAWS=2048
from sampling import uniforms,integration_statistics
from functools import lru_cache


def columns(clock):return previous.columns(clock)


def verified_source():
    run,source_sha,_=previous.verified_source();path=core.ROOT/'research/soxl_soxs_14d5pct_v59/evidence/delivery_manifest.json';m=read(path)
    for f,h in m['delivery_files_sha256'].items():assert sha(path.parents[1]/f)==h,f
    v=read(path.parent/'verification.json');assert v['passed'] and v['results_sha256']==sha(path.parent/'results.json')
    return run,source_sha,sha(path)


def quote(market,at):
    end=at-pd.Timedelta(seconds=30);values=[];records=[]
    for s in core.ETFS:
        f=market.bars[s]
        if end not in f.index:return None,None
        row=f.loc[end]
        if pd.isna(row.available_at) or not end<=row.available_at<=at or row.start_at!=end-pd.Timedelta(minutes=5) or not np.isfinite(row.close) or row.close<=0:return None,None
        values.append(float(row.close));records.append({'symbol':s,'at':end,'available_at':row.available_at,'raw_close':float(row.close),'economic_close':float(row.econ_close)})
    return np.array(values),records


def build_bank(market,panel):
    rows=[];prices=[];refs=[];prevs=[];splits=[];divs=[];states=[]
    sessions=list(market.sessions);positions={d:i for i,d in enumerate(sessions)};last=panel.date.max()
    for day in sessions:
        if day>last:break
        if day<panel.date.min():continue
        j=positions[day];prev=sessions[j-1] if j else None;cal=market.calendar[day]
        expected=pd.date_range(cal['open_at'],cal['close_at'],freq='5min',inclusive='right');ok=len(expected)==78 and prev is not None
        blocks=[];before=[];ratios=[];cash=[];release=[];reasons=[];meta=[]
        for s in core.ETFS:
            f=market.bars[s];g=f[f.date.eq(day)&f.session_type.eq('regular')];prior=market.daily[s]
            good=bool(ok and g.index.equals(expected) and g.available_at.notna().all() and g.available_at.ge(g.end_at).all() and g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and prev in prior.index and prior.loc[prev,'complete'])
            if not good:reasons.append(s+':incomplete_full_day_or_previous_close');break
            vals=g[['econ_open','econ_high','econ_low','econ_close']].to_numpy(float)
            if not (np.isfinite(vals).all() and (vals>0).all() and (vals[:,1]>=vals[:,[0,3]].max(axis=1)).all() and (vals[:,2]<=vals[:,[0,3]].min(axis=1)).all()):reasons.append(s+':invalid_ohlc');break
            pc=float(prior.loc[prev,'close']);pf=float(prior.loc[prev,'share_factor']);factor=float(g.share_factor.iloc[0]);assert g.share_factor.eq(factor).all()
            amount=0.;events=[];times=[g.available_at.max(),prior.loc[prev,'available_at']]
            for a in market.actions[s]:
                if a['type']=='CashDividends' and pd.Timestamp(a['date'])==day:
                    pay=pd.to_datetime(a.get('payment_date'),errors='coerce');value=float(a['value'])
                    if pd.isna(pay) or value<0 or not np.isfinite(value):reasons.append(s+':unknown_cash_metadata');break
                    known=(pd.Timestamp(pay.date(),tz=core.ET)+pd.Timedelta(days=1)).tz_convert('UTC');times.append(known);amount+=value*factor;events.append({'ex':day,'pay':pay,'economic_amount':value*factor,'assumed_available_at':known})
            if reasons:break
            blocks.append(vals);before.append(pc);ratios.append(factor/pf);cash.append(amount/pc);release+=times;meta.append({'symbol':s,'previous_close':pc,'previous_factor':pf,'factor':factor,'dividends':events})
        good=len(blocks)==2 and not reasons
        qs=np.full((8,2),np.nan);ss=np.full((8,4),np.nan);qmeta=[]
        if good:
            for c,clock in enumerate(CLOCKS):
                at=pd.Timestamp(str(day.date())+' '+clock,tz=core.ET).tz_convert('UTC');_,rec=quote(market,at)
                if rec is not None:qs[c]=[x['economic_close'] for x in rec];release+=[x['available_at'] for x in rec]
                if at in panel.index:ss[c]=panel.loc[at,FIELDS].to_numpy(float)
                qmeta.append(rec)
        idx=len(rows);rows.append({'index':idx,'date':day,'previous_date':prev,'complete':good,'reason':reasons if reasons else ([] if good else ['not_full_session']),'release':max(release) if good else None,'metadata':meta,'quotes':qmeta})
        prices.append(np.stack(blocks,axis=1) if good else np.full((78,2,4),np.nan));refs.append(qs);prevs.append(before if good else [np.nan]*2);splits.append(ratios if good else [np.nan]*2);divs.append(cash if good else [np.nan]*2);states.append(ss)
    lookup={r['date']:i for i,r in enumerate(rows)};pairs=[]
    for i,row in enumerate(rows):
        j=positions[row['date']]
        if j+1>=len(sessions) or sessions[j+1] not in lookup:continue
        k=lookup[sessions[j+1]]
        if row['complete'] and rows[k]['complete']:pairs.append([i,k])
    bank={'ohlc':np.array(prices),'quotes':np.array(refs),'previous_close':np.array(prevs),'split_ratio':np.array(splits),'dividend_ratio':np.array(divs),'states':np.array(states),'blocks':np.array(pairs,dtype=int).reshape(-1,2),'date_ns':np.array([r['date'].value for r in rows],dtype=np.int64),'release_ns':np.array([pd.Timestamp(r['release']).value if r['complete'] else pd.NaT.value for r in rows],dtype=np.int64)}
    add_successors(bank,market.sessions,panel)
    add_absolute(bank,panel)
    return bank,rows


def horizon(calendar,at):
    day=pd.Timestamp(at.tz_convert(core.ET).date());deadline=day+pd.Timedelta(days=13);dates=sorted(d for d in calendar if day<=d<=deadline)
    assert dates and dates[0]==day and max(calendar)>=deadline
    lengths=[int((pd.Timestamp(calendar[d]['close_at'])-pd.Timestamp(calendar[d]['open_at'])).total_seconds()/300) for d in dates]
    opening=pd.Timestamp(calendar[day]['open_at']);start=opening
    while start<=at+pd.Timedelta(seconds=30):start+=pd.Timedelta(minutes=5)
    offset=int((start-opening).total_seconds()/300)
    if offset>=lengths[0] or (start-at).total_seconds()>360:return None
    assert all(0<n<=78 for n in lengths)
    return {'dates':dates,'bar_counts':lengths,'first_offset':offset,'entry_at':start,'expiry_at':pd.Timestamp(calendar[dates[-1]]['close_at'])}


def source_pool(bank,at,clock):
    c=CLOCKS.index(clock);blocks=bank['blocks'];first=blocks[:,0];last=blocks[:,1];cutoff=day_cutoff(at).value
    good=(bank['release_ns'][first]<cutoff)&(bank['release_ns'][last]<cutoff)&(bank['date_ns'][last]<pd.Timestamp(at.tz_convert(core.ET).date()).value)
    good&=np.isfinite(bank['quotes'][first,c]).all(axis=1)&np.isfinite(bank['states'][first,c]).all(axis=1)&np.isfinite(bank['states'][first,0,:3]).all(axis=1)
    good&=np.isfinite(bank['successor_states']).all(axis=1)&(bank['successor_at_ns']!=pd.NaT.value)&(bank['successor_at_ns']<cutoff)
    good&=np.isfinite(bank['absolute_states'][first,c])
    return np.flatnonzero(good)


def kernel(x,current,kind):
    n=len(x);w=2.**(-np.arange(n)[::-1]/126.);w/=w.sum();mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    distance=np.sqrt(np.sum(((x-current)/scale)**2,axis=1));positive=distance[distance>1e-12];h=float(np.quantile(positive,.4)) if len(positive) else 1.
    probability=w*np.exp(-.5*(distance/h)**2) if kind=='conditional' else w.copy();total=probability.sum()
    if not np.isfinite(total) or total<=0:return {'ready':False}
    probability/=total;effective=float(1/(probability@probability))
    return {'ready':effective>=20,'center':mean.tolist(),'scale':scale.tolist(),'distance':distance.tolist(),'bandwidth':h,'recency':w.tolist(),'probability':probability.tolist(),'effective_dates':effective}


def add_absolute(bank,panel):
    values=np.full((len(bank['date_ns']),len(CLOCKS)),np.nan)
    for i,ns in enumerate(bank['date_ns']):
        for c,clock in enumerate(CLOCKS):
            at=pd.Timestamp(str(pd.Timestamp(ns).date())+' '+clock,tz=core.ET).tz_convert('UTC')
            if at in panel.index:values[i,c]=panel.loc[at,ABSOLUTE]
    bank['absolute_states']=values


def first_kernels(bank,ids,clock,row):
    c=CLOCKS.index(clock);x=bank['states'][ids,c];current=row[FIELDS].to_numpy(float)
    old=kernel(x,current,'conditional')
    absolute=kernel(np.column_stack([x,bank['absolute_states'][ids,c]]),np.r_[current,float(row[ABSOLUTE])],'conditional')
    return old,absolute


def add_successors(bank,sessions,panel):
    positions={d:i for i,d in enumerate(sessions)};values=[];times=[]
    for _,last in bank['blocks']:
        day=pd.Timestamp(bank['date_ns'][last]);j=positions[day]+1
        if j>=len(sessions):values.append([np.nan]*3);times.append(pd.NaT.value);continue
        at=pd.Timestamp(str(sessions[j].date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC')
        values.append(panel.loc[at,PRIOR].to_numpy(float) if at in panel.index else [np.nan]*3);times.append(at.value)
    bank['successor_states']=np.array(values,float).reshape(-1,3);bank['successor_at_ns']=np.array(times,dtype=np.int64)


@lru_cache(maxsize=16)
def transition_matrix(source_bytes,next_bytes):
    x=np.frombuffer(source_bytes,dtype=np.float64).reshape(-1,3);y=np.frombuffer(next_bytes,dtype=np.float64).reshape(-1,3)
    records=[kernel(x,state,'conditional') for state in y]
    matrix=np.array([row.get('probability',np.full(len(x),np.nan)) for row in records]);effective=np.array([row.get('effective_dates',0.) for row in records])
    matrix.flags.writeable=False;effective.flags.writeable=False
    return matrix,effective,bool(all(row['ready'] for row in records))


def draw_blocks(members,first,later,at,clock,days,count=DRAWS,replication=0,transition=None):
    seed=1062+pd.Timestamp(at.tz_convert(core.ET).date()).toordinal()*16+CLOCKS.index(clock);u=uniforms(seed,(days+1)//2,count,replication);local=np.zeros(u.shape,dtype=int)
    cdf=np.cumsum(first);cdf[-1]=1.;local[:,0]=np.searchsorted(cdf,u[:,0],side='right')
    if transition is not None:
        tcdf=np.cumsum(transition,axis=1);tcdf[:,-1]=1.
    else:
        cdf=np.cumsum(later);cdf[-1]=1.
    for k in range(1,u.shape[1]):
        if transition is None:local[:,k]=np.searchsorted(cdf,u[:,k],side='right')
        else:local[:,k]=np.sum(tcdf[local[:,k-1]]<=u[:,k,None],axis=1)
    return np.asarray(members)[local],seed


def simulate(bank,selected,clock,current_raw,plan):
    n=len(selected);alive=np.ones((n,2),bool);factor=np.ones((n,2));prior=np.broadcast_to(current_raw,(n,2)).copy();cash=np.zeros((n,2));entry=np.zeros((n,2));target=np.zeros((n,2));stop=np.zeros((n,2));fills=np.zeros((n,2));reasons=np.zeros((n,2),int);exit_day=np.full((n,2),-1,int);exit_bar=exit_day.copy();ar=np.arange(n);c=CLOCKS.index(clock)
    for day,count in enumerate(plan['bar_counts']):
        ids=bank['blocks'][selected[:,day//2],day%2];first=plan['first_offset'] if day==0 else 0
        scale=prior/(bank['quotes'][ids,c] if day==0 else bank['previous_close'][ids])
        path=bank['ohlc'][ids,first:count]*scale[:,None,:,None]
        if day==0:
            entry=path[:,0,:,0]*(1+core.SLIP);target=np.ceil(entry*1.05*100-1e-10)/100;stop=entry*.94
        else:
            factor*=bank['split_ratio'][ids];cash+=alive*prior*bank['dividend_ratio'][ids]
        low=np.floor(stop/factor*100+1e-10)/100*factor;high=np.ceil(target/factor*100-1e-10)/100*factor
        sl=path[:,:,:,2]<=low[:,None,:];tp=path[:,:,:,1]>=high[:,None,:];hit=sl|tp;where=np.argmax(hit,axis=1);seen=hit.any(axis=1)&alive
        for s in range(2):
            j=where[:,s];isstop=sl[ar,j,s];fill=np.where(isstop,np.minimum(path[ar,j,s,0],low[:,s])*(1-core.SLIP),high[:,s]);mask=seen[:,s]
            fills[mask,s]=fill[mask];reasons[mask,s]=np.where(isstop[mask],2,1);exit_day[mask,s]=day;exit_bar[mask,s]=j[mask]+first
        alive&=~seen;prior=path[:,-1,:,3]
        if day==len(plan['bar_counts'])-1:
            fills[alive]=prior[alive]*(1-core.SLIP);reasons[alive]=3;exit_day[alive]=day;exit_bar[alive]=count-1;alive[:]=False
        if not alive.any():break
    assert not alive.any() and np.isfinite(fills).all() and (fills>0).all()
    net=(fills*(1-core.FEE)+cash)/(entry*(1+core.FEE))-1;assert (net>-1).all()
    return {'net':net,'log_returns':np.log1p(net),'entry':entry,'fill':fills,'cash':cash,'reason':reasons,'exit_day':exit_day,'exit_bar':exit_bar}


def summarize(result):
    logs=result['log_returns'];c=logs.mean(axis=1);d=(logs[:,0]-logs[:,1])/2;m=abs(d);C=float(c.mean());D=float(d.mean());M=float(m.mean());q=float(m[d>0].sum()/m.sum()) if m.sum()>0 else .5
    return {'C':C,'D':D,'M':M,'q':q,'SOXL':C+D,'SOXS':C-D}


def forecast(bank,market,panel,at,kind,count=DRAWS,replication=0):
    row=panel.loc[at];members=source_pool(bank,at,row.clock);rawquote,qr=quote(market,at);plan=horizon(market.calendar,at)
    info={'members':members.tolist(),'dates':[pd.Timestamp(bank['date_ns'][bank['blocks'][i,0]]) for i in members],'cutoff':day_cutoff(at),'query_quote':qr,'plan':plan,'ready':False,'kernels':{},'raw':{}}
    if len(members)<120 or rawquote is None or plan is None or not row.feature_ready or not np.isfinite(row[FIELDS+[ABSOLUTE]].to_numpy(float)).all():return None,info,None
    ids=bank['blocks'][members,0];old_first,absolute_first=first_kernels(bank,ids,row.clock,row)
    a=absolute_first if kind=='conditional' else old_first if kind=='fixed' else kernel(bank['states'][ids,CLOCKS.index(row.clock)],row[FIELDS].to_numpy(float),'unconditional')
    later_conditional=kernel(bank['states'][ids,0,:3],row[PRIOR].to_numpy(float),'conditional')
    b=kernel(bank['states'][ids,0,:3],row[PRIOR].to_numpy(float),'unconditional') if kind=='unconditional' else later_conditional
    info['kernels']={'first':a,'later':b}
    info['absolute_first_kernel']=absolute_first
    info['shared_ready']=all(z['ready'] for z in [old_first,absolute_first,later_conditional])
    matrix,effective,ready=transition_matrix(bank['states'][ids,0,:3].astype('float64').tobytes(),bank['successor_states'][members].astype('float64').tobytes())
    info['transition']={'ready':ready,'minimum_effective_dates':float(min(effective)),'next_state_point':np.array(a['probability'])@bank['successor_states'][members]}
    if not a['ready'] or not b['ready'] or not ready or not info['shared_ready']:return None,info,None
    selected,seed=draw_blocks(members,a['probability'],b['probability'],at,row.clock,len(plan['dates']),count,replication,transition=None);result=simulate(bank,selected,row.clock,rawquote,plan);raw=summarize(result)
    integration=integration_statistics(result['log_returns'])
    info.update(ready=True,seed=seed,replication=replication,paths=count,raw=raw,integration=integration,monte_carlo_se=integration['standard_error'],reason_frequencies={str(i):np.mean(result['reason']==i,axis=0).tolist() for i in [1,2,3]})
    model={'selected_blocks':selected,'result':result};return model,info,np.array([raw[s] for s in core.ETFS])


def events(at,date,clock,estimates,daily,old56,old55):
    primary=decide(estimates['conditional'],daily['conditional']['feedback'],True);cash={'ready':True,'action':'CASH','allocation':0.}
    choices={'conditional_band':primary,'conditional_point':decide(estimates['conditional'],daily['conditional']['feedback'],False),'unconditional_band':decide(estimates['unconditional'],daily['unconditional']['feedback'],True),'fixed_band':decide(estimates['fixed'],daily['fixed']['feedback'],True),'source_v56':old56,'source_v55':old55,'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in choices.items()}
