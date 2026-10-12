"""Independent joint intercept/slope equations and explicit repeated-date bootstrap."""
from hierarchical_error import core,np,pd,CLOCKS
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near


def input_rows(panel,history,ids):
    return np.array([[history[t][0],history[t][1],list(CLOCKS).index(panel.loc[t,'clock'])] for t in ids],float)


def reference_penalty(kind):
    if kind=='shared':return np.diag([20.,20.])
    matrix=np.zeros((8,8));lam=20. if kind=='partial' else 5.
    for k in range(2):
        mean=np.zeros(8);mean[k::2]=.25;matrix+=20*np.outer(mean,mean)
        for c in range(4):
            v=np.zeros(8);v[2*c+k]=1;v-=mean;matrix+=lam*np.outer(v,v)
    return matrix


def solve(X,Y,w,kind,n):
    w=np.array(w)*n/np.sum(w);clock=np.array(X[:,2],int);means=[];scales=[];Z=np.zeros((len(X),2))
    for c in range(4):
        ids=np.flatnonzero(clock==c);a=X[ids,:2];v=w[ids];m=np.sum(v[:,None]*a,axis=0)/sum(v)
        sd=np.sqrt(np.sum(v[:,None]*(a-m)**2,axis=0)/sum(v));sd[sd<1e-12]=1
        means.append(m);scales.append(sd);Z[ids]=(a-m)/sd
    # Fit intercepts jointly instead of residualizing each clock's targets.
    rows=[]
    for i,c in enumerate(clock):
        values=[float(c==j) for j in range(4)]
        if kind=='shared':values.extend(Z[i])
        else:values.extend(Z[i,k] if c==j else 0. for j in range(4) for k in range(2))
        rows.append(values)
    A=np.array(rows);P=np.zeros((A.shape[1],A.shape[1]));P[4:,4:]=reference_penalty(kind)
    beta=np.linalg.solve(A.T@(w[:,None]*A)+P,A.T@(w[:,None]*Y))
    coeff=np.array([beta[4:]]*4) if kind=='shared' else beta[4:].reshape(4,2,2)
    return {'kind':kind,'input_center':means,'input_scale':scales,'intercepts':beta[:4],'coefficients':coeff}


def model_check(saved,want):
    assert saved['kind']==want['kind']
    for k in ['input_center','input_scale','intercepts','coefficients']:close(saved[k],want[k])


def predict(model,x):
    out=[]
    for row in np.atleast_2d(x):
        c=int(row[2]);normalized=[(row[j]-model['input_center'][c][j])/model['input_scale'][c][j] for j in range(2)]
        out.append([model['intercepts'][c][a]+sum(normalized[j]*model['coefficients'][c][j][a] for j in range(2)) for a in range(2)])
    return np.array(out)


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
    for kind in ['partial','independent','shared']:
        point=solve(X,Y,rw,kind,n);model_check(saved['variants'][kind]['point_model'],point);samples=[];checks+=2
        for b,draw in enumerate(starts):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
            ref=solve(X[indices],Y[indices],rw[indices],kind,n);model_check(saved['variants'][kind]['bootstrap_models'][b],ref);samples.append(ref);checks+=2
        models[kind]={'point_model':point,'bootstrap_models':samples}
    return models,checks
