"""Independent dense design, outer-product penalty and all base normal equations."""
from scipy.special import expit
from shared_learning import core,np,pd,CLOCKS,columns
from verify_reliability import close
from verify_continuation import near


def reference_members(panel,market,at):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC');today=pd.Timestamp(at.tz_convert(core.ET).date());prior=[]
    for day,g in panel.groupby('date'):
        if day>=today:break
        required=[pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[day]['close_at'])]
        if list(g.index)!=required or not all(g.feature_ready):continue
        prior.append({'date':day,'members':required,'feature_ready':True,'release':max(g.terminal_release) if g.terminal_release.notna().all() else None})
    parents=[]
    for r in prior:
        if r['release'] is None or r['release']>=cutoff:break
        parents.append(r)
    first=prior[len(parents)] if len(parents)<len(prior) else None
    fb={'prior_feature_ready_dates':len(prior),'selected_parent_dates':len(parents),'first_pending_date':first['date'] if first else None,'expired_unresolved_prefix':bool(first and first['date']+pd.Timedelta(days=14)<=today)}
    return parents,fb


def reference_layout(cols,kind):
    names=[];index=[]
    for c,fields in enumerate(cols):
        loc=[]
        for f in fields:
            name=f if kind=='shared' else (c,f)
            if name not in names:names.append(name)
            loc.append(names.index(name))
        index.append(loc)
    return index,len(names)


def reference_penalty(cols,eta,kind,strength):
    index,size=reference_layout(cols,kind);out=np.zeros((size,size))
    for f in sorted(set(f for fields in cols for f in fields)):
        active=[c for c,fields in enumerate(cols) if f in fields];total=sum(eta[c] for c in active)
        if kind=='shared':
            v=np.zeros(size);v[index[active[0]][cols[active[0]].index(f)]]=1;out+=strength*total*np.outer(v,v)
        else:
            mean=np.zeros(size)
            for c in active:mean[index[c][cols[c].index(f)]]=eta[c]/total
            out+=strength*total*np.outer(mean,mean)
            for c in active:
                dev=-mean.copy();dev[index[c][cols[c].index(f)]]+=1
                out+=strength*(4 if kind=='partial' else 1)*eta[c]*np.outer(dev,dev)
    return out


def reference_arrays(panel,parents):
    ids=[t for p in parents for t in p['members']];n=len(parents)
    dw=np.array([2.**((i-n+1)/126.) for i in range(n)]);dw=dw*n/dw.sum() if n else dw
    rw=np.array([dw[j]/len(p['members']) for j,p in enumerate(parents) for _ in p['members']])
    frame=panel.loc[ids];counts=[int(sum(frame.clock==c)) for c in CLOCKS];ready=n>=120 and min(counts)>=120
    meta={'members':ids,'dates':[p['date'] for p in parents],'N':n,'parent_weights':dw,'row_weights':rw,'clock_rows':counts,'ready':ready}
    if not ready:return meta,None
    cols=[columns(c) for c in CLOCKS];centers=[];scales=[];zs=[];eta=[];weights=[];cs=[];ds=[]
    for c,fields in zip(CLOCKS,cols):
        mask=frame.clock.eq(c).to_numpy();x=frame.loc[mask,fields].to_numpy(float);w=rw[mask]
        mean=np.sum(w[:,None]*x,axis=0)/w.sum();scale=np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0)/w.sum());scale[scale<1e-12]=1.
        centers.append(mean);scales.append(scale);zs.append((x-mean)/scale);eta.append(w.sum()/n);weights.append(w);cs.append(frame.loc[mask,'C'].to_numpy(float));ds.append(frame.loc[mask,'D'].to_numpy(float))
    meta.update(columns=cols,center=centers,scale=scales,eta=eta)
    return meta,{'Z':zs,'weights':weights,'C':cs,'D':ds,'columns':cols,'eta':eta,'N':n}


def check_model(arrays,kind,saved):
    cols=arrays['columns'];index,size=reference_layout(cols,kind);clocks=len(cols);rows=sum(len(z) for z in arrays['Z'])
    design=np.zeros((rows,clocks+size));cursor=0
    for c,z in enumerate(arrays['Z']):
        sl=slice(cursor,cursor+len(z));design[sl,c]=1;design[sl,clocks+np.array(index[c])]=z;cursor+=len(z)
    rw=np.concatenate(arrays['weights']);P=reference_penalty(cols,arrays['eta'],kind,100.);pen=np.zeros((clocks+size,clocks+size));pen[clocks:,clocks:]=P
    truth=np.column_stack([np.concatenate(arrays['C']),abs(np.concatenate(arrays['D']))]);beta=np.linalg.solve(design.T@(rw[:,None]*design)+pen,design.T@(rw[:,None]*truth))
    assert saved['kind']==kind
    for j,key in enumerate(['C','M']):close(saved[key]['intercepts'],beta[:clocks,j]);close(saved[key]['coefficients'],beta[clocks:,j])
    q=saved['q'];w=[];active=[];fallback=[]
    for c,(d,v) in enumerate(zip(arrays['D'],arrays['weights'])):
        a=np.array([v[i]*abs(d[i]) for i in range(len(d))]);mass=a.sum()
        if mass:a=a*v.sum()/mass
        positive=a[d>0].sum();value=positive/a.sum() if a.sum() else .5
        mixed=bool(positive>0 and a[d<0].sum()>0)
        active.append(mixed);fallback.append(value);w.append(a);close(q['classification_weights'][c],a)
    assert q['mixed_classes']==active;close(q['fallback_q'],fallback)
    coef=np.array(q['coefficients']);intercepts=np.array(q['intercepts']);p=reference_penalty(cols,arrays['eta'],kind,10.)
    fulltheta=np.r_[intercepts,coef];y=np.concatenate([d>0 for d in arrays['D']]).astype(float)
    cw=np.concatenate([a if active[c] else np.zeros_like(a) for c,a in enumerate(w)])
    linear=design@fulltheta;prob=expit(linear);gradient=design.T@(cw*(prob-y));gradient[clocks:]+=p@coef
    for c,enabled in enumerate(active):
        if not enabled:assert intercepts[c]==0
    value=float(cw@(np.logaddexp(0,linear)-y*linear)+.5*coef@p@coef);near(q['objective'],value)
    maximum=float(max(abs(gradient)));near(q['gradient_max_abs'],maximum);assert maximum<1e-7*max(1,arrays['N'])
    return 3


def check_day(panel,market,at,saved):
    from run_planning import json_ready
    import json
    canon=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    parents,fb=reference_members(panel,market,at);assert saved['parents']==canon(parents) and saved['feedback']==canon(fb)
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    meta,arrays=reference_arrays(panel,parents)
    for k,v in meta.items():
        if k in ['parent_weights','row_weights','center','scale','eta']:
            if k in ['center','scale']:
                for a,b in zip(saved['prepared'][k],v):close(a,b)
            else:close(saved['prepared'][k],v)
        else:assert saved['prepared'][k]==canon(v),(k,saved['prepared'][k],canon(v))
    if not meta['ready']:assert saved['models']=={};return 0
    return sum(check_model(arrays,k,saved['models'][k]) for k in ['partial','independent','shared'])


def predict_ref(panel,at,saved,kind):
    m=saved['prepared']
    if not m['ready'] or not panel.loc[at,'feature_ready']:return {},None
    clock=CLOCKS.index(panel.loc[at,'clock']);index,_=reference_layout(m['columns'],kind);ids=index[clock];model=saved['models'][kind]
    z=[(float(panel.loc[at,f])-m['center'][clock][j])/m['scale'][clock][j] for j,f in enumerate(m['columns'][clock])]
    c=model['C']['intercepts'][clock]+sum(x*model['C']['coefficients'][j] for x,j in zip(z,ids))
    magnitude=max(0.,model['M']['intercepts'][clock]+sum(x*model['M']['coefficients'][j] for x,j in zip(z,ids)))
    q=model['q']['fallback_q'][clock]
    if model['q']['mixed_classes'][clock]:q=float(expit(model['q']['intercepts'][clock]+sum(x*model['q']['coefficients'][j] for x,j in zip(z,ids))))
    direction=magnitude*(2*q-1);raw={'C':c,'M':magnitude,'D':direction,'q':q,'SOXL':c+direction,'SOXS':c-direction}
    return raw,np.array([raw[s] for s in core.ETFS])
