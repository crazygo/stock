"""Independent joint normal equations, explicit date resampling and input reconstruction."""
from conditional_calibration import core,np,pd,CLOCKS,MACRO,VARIANTS
from audit_shared import reference_pool
from verify_reliability import close


def input_rows(panel,history,ids,kind):
    cols=MACRO if kind in ['macro_context','base_context'] else []
    return np.array([[history[t][0],history[t][1],*[panel.loc[t,c] for c in cols],list(CLOCKS).index(panel.loc[t,'clock'])] for t in ids],float)

def reference_penalty(p):
    matrix=np.zeros((4*p,4*p))
    for k in range(p):
        mean=np.zeros(4*p);mean[k::p]=.25;matrix+=20*np.outer(mean,mean)
        for c in range(4):
            v=np.zeros(4*p);v[c*p+k]=1;v-=mean;matrix+=20*np.outer(v,v)
    return matrix

def solve(X,Y,w,n):
    w=np.asarray(w)*n/np.sum(w);keep=w>0;X=X[keep];Y=Y[keep];w=w[keep];p=X.shape[1]-1
    clock=np.array(X[:,-1],int);means=[];scales=[];Z=np.zeros((len(X),p))
    assert np.array_equal(clock,X[:,-1]) and set(clock)==set(range(4))
    for c in range(4):
        ids=np.flatnonzero(clock==c);a=X[ids,:p];v=w[ids];m=np.sum(v[:,None]*a,axis=0)/sum(v)
        sd=np.sqrt(np.sum(v[:,None]*(a-m)**2,axis=0)/sum(v));sd[sd<1e-12]=1
        means.append(m);scales.append(sd);Z[ids]=(a-m)/sd
    rows=[]
    for i,c in enumerate(clock):rows.append([float(c==j) for j in range(4)]+[Z[i,k] if c==j else 0. for j in range(4) for k in range(p)])
    A=np.array(rows);P=np.zeros((A.shape[1],A.shape[1]));P[4:,4:]=reference_penalty(p)
    beta=np.linalg.solve(A.T@(w[:,None]*A)+P,A.T@(w[:,None]*Y))
    return {'kind':'macro_partial' if p==10 else 'partial','input_center':means,'input_scale':scales,'intercepts':beta[:4],'coefficients':beta[4:].reshape(4,p,2)}

def model_check(saved,want):
    assert saved['kind']==want['kind']
    for k in ['input_center','input_scale','intercepts','coefficients']:close(saved[k],want[k])

def predict(model,x):
    out=[];p=len(model['input_center'][0])
    for row in np.atleast_2d(x):
        c=int(row[-1]);assert c==row[-1] and c in range(4)
        z=[(row[j]-model['input_center'][c][j])/model['input_scale'][c][j] for j in range(p)]
        out.append([model['intercepts'][c][a]+sum(z[j]*model['coefficients'][c][j][a] for j in range(p)) for a in range(2)])
    return np.array(out)

def daily_check(panel,market,histories,at,saved):
    from run_planning import json_ready
    import json
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    common={t:histories['macro_plain'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['macro_plain'])}
    parents,fb=reference_pool(panel,market,common,at);assert canonical(saved['parents'])==canonical(parents) and canonical(saved['feedback'])==canonical(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['variants']=={};return {},0
    dw=np.array([2.**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum()
    rw=np.array([dw[i]/len(p['members']) for i,p in enumerate(parents) for _ in p['members']]);close(saved['day_weights'],dw);close(saved['row_weights'],rw)
    groups=[i for i,p in enumerate(parents) for _ in p['members']];assert saved['row_groups']==groups
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    bydate=[np.flatnonzero(np.asarray(groups)==j) for j in range(n)];models={};checks=0
    for kind in VARIANTS:
        X=input_rows(panel,histories[kind],ids,kind);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
        point=solve(X,Y,rw,n);model_check(saved['variants'][kind]['point_model'],point);samples=[];checks+=2
        for b,draw in enumerate(starts):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
            ref=solve(X[indices],Y[indices],rw[indices],n);model_check(saved['variants'][kind]['bootstrap_models'][b],ref);samples.append(ref);checks+=2
        models[kind]={'point_model':point,'bootstrap_models':samples}
    return models,checks
