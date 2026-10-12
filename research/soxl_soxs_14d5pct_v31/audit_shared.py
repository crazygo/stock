"""Explicit date resampling and independent normal-equation model verification."""
from shared_error import core,np,pd,CLOCKS,STATE_COLUMNS
from verify_reliability import close
from verify_continuation import near


def reference_pool(panel,market,history,at):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC');today=pd.Timestamp(at.tz_convert(core.ET).date())
    prefix=[];prior=[]
    for day in sorted(set(panel.date)):
        if day>=today:break
        frame=panel[panel.date==day];cl=pd.Timestamp(market.calendar[day]['close_at'])
        required=[pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<cl]
        if list(frame.index)!=required or not frame.feature_ready.all():continue
        prior.append({'date':day,'members':required,'feature_ready':True,
            'release':max(frame.terminal_release) if frame.terminal_release.notna().all() else None})
    for r in prior:
        if r['release'] is None or r['release']>=cutoff:break
        prefix.append(r)
    pending=prior[len(prefix)] if len(prefix)<len(prior) else None
    selected=[r for r in prefix if all(t in history and history[t] is not None for t in r['members'])][-120:]
    fb={'prior_dates':len(prior),'selected_early_dates':len(prefix),'first_pending_date':pending['date'] if pending else None,
        'expired_unresolved_prefix':pending is not None and pending['date']+pd.Timedelta(days=14)<=today,
        'eligible_forecast_dates':len(selected),'candidate_rows':sum(len(r['members']) for r in selected)}
    return selected,fb


def input_rows(panel,history,ids):
    return np.array([[*history[t],*[panel.loc[t,c] for c in STATE_COLUMNS],*[int(panel.loc[t,'clock']==c) for c in CLOCKS[1:]]] for t in ids],float)


def basis(raw,kind):
    # Loop expansion is independent of training's broadcast/reshape ordering.
    result=[]
    for row in raw:
        values=[*row[:2],*row[8:]]
        if kind!='shared':values.extend(row[2:8])
        if kind=='interaction':values.extend(row[a]*row[b] for a in range(2) for b in range(2,8))
        result.append(values)
    return np.array(result)


def solve(X,Y,w,kind,n):
    w=w*n/w.sum();mx=np.average(X,axis=0,weights=w);sx=np.sqrt(np.average((X-mx)**2,axis=0,weights=w));sx[sx<1e-12]=1.
    A=basis((X-mx)/sx,kind);ma=np.average(A,axis=0,weights=w);sa=np.sqrt(np.average((A-ma)**2,axis=0,weights=w));sa[sa<1e-12]=1.
    Z=(A-ma)/sa;mz=np.average(Z,axis=0,weights=w);my=np.average(Y,axis=0,weights=w);R=Z-mz
    coef=np.linalg.solve(R.T@(w[:,None]*R)+20*np.eye(Z.shape[1]),R.T@(w[:,None]*(Y-my)));intercept=my-mz@coef
    return {'kind':kind,'input_center':mx,'input_scale':sx,'basis_center':ma,'basis_scale':sa,'coef':coef.T,'intercept':intercept}


def model_check(saved,want):
    assert saved['kind']==want['kind']
    for k in want:
        if k!='kind':close(saved[k],want[k])


def predict(model,x):
    x=np.atleast_2d(x);A=basis((x-model['input_center'])/model['input_scale'],model['kind'])
    return ((A-model['basis_center'])/model['basis_scale'])@np.asarray(model['coef']).T+model['intercept']


def daily_check(panel,market,history,at,saved):
    from run_planning import json_ready
    import json
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    parents,fb=reference_pool(panel,market,history,at);assert canonical(saved['parents'])==canonical(parents) and canonical(saved['feedback'])==canonical(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['variants']=={};return {},0
    dw=np.array([2.**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum()
    rw=np.array([dw[i]/len(p['members']) for i,p in enumerate(parents) for _ in p['members']]);close(saved['day_weights'],dw);close(saved['row_weights'],rw)
    groups=[i for i,p in enumerate(parents) for _ in p['members']];assert saved['row_groups']==groups
    X=input_rows(panel,history,ids);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    bydate=[np.flatnonzero(np.asarray(groups)==j) for j in range(n)];models={};checks=0
    for kind in ['interaction','additive','shared']:
        point=solve(X,Y,rw,kind,n);model_check(saved['variants'][kind]['point_model'],point);samples=[];checks+=2
        for b,draw in enumerate(starts):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
            ref=solve(X[indices],Y[indices],rw[indices],kind,n);model_check(saved['variants'][kind]['bootstrap_models'][b],ref);samples.append(ref);checks+=2
        models[kind]={'point_model':point,'bootstrap_models':samples}
    return models,checks
