"""Independent joint intercept normal equations with explicit repeated dates."""
from short_learning import core,np,pd,CLOCKS
from verify_reliability import close


def reference_pool(panel,market,history,at,clocks):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC');today=pd.Timestamp(at.tz_convert(core.ET).date());prefix=[];prior=[]
    for day in sorted(set(panel.date)):
        if day>=today:break
        frame=panel[panel.date==day];cl=pd.Timestamp(market.calendar[day]['close_at'])
        required=[pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in clocks if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<cl]
        if list(frame.index)!=required or not frame.feature_ready.all():continue
        prior.append({'date':day,'members':required,'feature_ready':True,'release':max(frame.terminal_release) if frame.terminal_release.notna().all() else None})
    for r in prior:
        if r['release'] is None or r['release']>=cutoff:break
        prefix.append(r)
    pending=prior[len(prefix)] if len(prefix)<len(prior) else None
    selected=[r for r in prefix if all(t in history and history[t] is not None for t in r['members'])][-120:]
    fb={'prior_dates':len(prior),'selected_early_dates':len(prefix),'first_pending_date':pending['date'] if pending else None,
        'expired_unresolved_prefix':pending is not None and pending['date']+pd.Timedelta(days=14)<=today,'eligible_forecast_dates':len(selected),'candidate_rows':sum(len(r['members']) for r in selected)}
    return selected,fb


def reference_penalty(k):
    matrix=np.zeros((2*k,2*k))
    for j in range(2):
        mean=np.zeros(2*k);mean[j::2]=1/k;matrix+=20*np.outer(mean,mean)
        for c in range(k):
            v=np.zeros(2*k);v[2*c+j]=1;v-=mean;matrix+=(80./k)*np.outer(v,v)
    return matrix


def solve(X,Y,w,n,k):
    w=np.array(w)*n/np.sum(w);clock=np.array(X[:,2],int);means=[];scales=[];Z=np.zeros((len(X),2))
    for c in range(k):
        ids=np.flatnonzero(clock==c);a=X[ids,:2];v=w[ids];assert sum(v)>0;m=np.sum(v[:,None]*a,axis=0)/sum(v)
        sd=np.sqrt(np.sum(v[:,None]*(a-m)**2,axis=0)/sum(v));sd[sd<1e-12]=1;means.append(m);scales.append(sd);Z[ids]=(a-m)/sd
    A=np.array([[*[float(c==j) for j in range(k)],*[Z[i,j] if c==l else 0. for l in range(k) for j in range(2)]] for i,c in enumerate(clock)])
    P=np.zeros((3*k,3*k));P[k:,k:]=reference_penalty(k);beta=np.linalg.solve(A.T@(w[:,None]*A)+P,A.T@(w[:,None]*Y))
    return {'kind':'partial','input_center':means,'input_scale':scales,'intercepts':beta[:k],'coefficients':beta[k:].reshape(k,2,2)}


def model_check(saved,want):
    assert saved['kind']==want['kind']
    for key in ['input_center','input_scale','intercepts','coefficients']:close(saved[key],want[key])


def predict(model,x):
    out=[]
    for row in np.atleast_2d(x):
        c=int(row[2]);z=[(row[j]-model['input_center'][c][j])/model['input_scale'][c][j] for j in range(2)]
        out.append([model['intercepts'][c][a]+sum(z[j]*model['coefficients'][c][j][a] for j in range(2)) for a in range(2)])
    return np.array(out)


def input_rows(panel,history,ids,clocks):return np.array([[history[t][0],history[t][1],clocks.index(panel.loc[t,'clock'])] for t in ids])


def daily_check(panel,market,history,at,saved,clocks=CLOCKS):
    from run_planning import json_ready
    import json
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    parents,fb=reference_pool(panel,market,history,at,clocks);assert saved['clocks']==clocks
    assert canonical(saved['parents'])==canonical(parents) and canonical(saved['feedback'])==canonical(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['model']=={};return {},0
    dw=np.array([2.**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum();rw=np.array([dw[i]/len(p['members']) for i,p in enumerate(parents) for _ in p['members']])
    close(saved['day_weights'],dw);close(saved['row_weights'],rw);groups=[i for i,p in enumerate(parents) for _ in p['members']];assert saved['row_groups']==groups
    X=input_rows(panel,history,ids,clocks);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed;starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    bydate=[np.flatnonzero(np.array(groups)==j) for j in range(n)];point=solve(X,Y,rw,n,len(clocks));model_check(saved['model']['point_model'],point);samples=[]
    for b,draw in enumerate(starts):
        days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days]);ref=solve(X[indices],Y[indices],rw[indices],n,len(clocks))
        model_check(saved['model']['bootstrap_models'][b],ref);samples.append(ref)
    return {'point_model':point,'bootstrap_models':samples},400
