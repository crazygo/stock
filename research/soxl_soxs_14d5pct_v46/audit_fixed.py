"""Independent daily member reconstruction and fixed-penalty normal equations."""
from fixed_candidate import core,np,pd,CLOCKS,MACRO,CANDIDATES
from audit_shared import reference_pool
from audit_penalty import solve,model_check,reference_predict
from verify_reliability import close


def daily_check(panel,market,history,at,saved):
    from run_planning import json_ready
    import json
    wire=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    parents,fb=reference_pool(panel,market,history,at);ids=[t for r in parents for t in r['members']];n=len(parents)
    for f,want in [('parents',parents),('members',ids),('feedback',fb)]:assert saved[f]==wire(want),f
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC') and saved['ready']==(n>=60)
    if n<60:assert saved['models']=={};return {},0
    dw=np.array([2.**((j-n+1)/63) for j in range(n)]);dw*=n/dw.sum();w=np.array([dw[j]/len(r['members']) for j,r in enumerate(parents) for _ in r['members']]);groups=np.repeat(np.arange(n),[len(r['members']) for r in parents])
    close(saved['day_weights'],dw);close(saved['row_weights'],w);close(saved['row_groups'],groups)
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    X=np.array([[*history[t],*[panel.loc[t,c] for c in MACRO],CLOCKS.index(panel.loc[t,'clock'])] for t in ids]);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
    counts=[]
    for draw in starts:
        days=[(int(s)+j)%n for s in draw for j in range(10)][:n];counts.append(np.array([days.count(j) for j in range(n)]))
    models={};checks=0
    for k in CANDIDATES:
        m=solve(X,Y,w,n,k);model_check(saved['models'][k]['point_model'],m);samples=[];checks+=2
        for i,c in enumerate(counts):
            b=solve(X,Y,w*c[groups],n,k);model_check(saved['models'][k]['bootstrap_models'][i],b);samples.append(b);checks+=2
        models[k]={'point_model':m,'bootstrap_models':samples}
    return models,checks
