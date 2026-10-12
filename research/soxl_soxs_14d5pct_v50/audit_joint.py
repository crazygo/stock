"""Independent joint intercept/slope equations and explicit repeated-date bootstrap."""
from confidence_shrinkage import core,np,pd,CLOCKS
from audit_hierarchical import input_rows,solve,model_check,predict
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near
from audit_shrinkage import check_shrinkage,input_rows as offset_rows
from confidence_shrinkage import correction_columns,CORRECTIONS,VARIANTS


def daily_check(panel,market,records,histories,at,saved):
    from run_planning import json_ready
    import json
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    common={t:histories['baseline'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['baseline'])}
    parents,fb=reference_pool(panel,market,common,at);assert canonical(saved['parents'])==canonical(parents) and canonical(saved['feedback'])==canonical(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['variants']=={} and saved['offsets']=={};return {},{},0,0
    dw=np.array([2.**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum()
    rw=np.array([dw[i]/len(p['members']) for i,p in enumerate(parents) for _ in p['members']]);close(saved['day_weights'],dw);close(saved['row_weights'],rw)
    groups=[i for i,p in enumerate(parents) for _ in p['members']];assert saved['row_groups']==groups
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    bydate=[np.flatnonzero(np.asarray(groups)==j) for j in range(n)];models={};checks=0
    offsets={};offset_checks=0
    for kind in CORRECTIONS:
        cols=correction_columns(kind);assert saved['offsets'][kind]['columns']==cols
        X,q=offset_rows(panel,records,ids,cols);D=panel.loc[ids,'D'].to_numpy(float)
        point,c=check_shrinkage(X,q,D,rw,n,kind,saved['offsets'][kind]['point_model']);samples=[];offset_checks+=c
        for b,draw in enumerate(starts):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n]
            indices=np.concatenate([bydate[j] for j in days])
            model,c=check_shrinkage(X[indices],q[indices],D[indices],rw[indices],n,kind,saved['offsets'][kind]['bootstrap_models'][b])
            samples.append(model);offset_checks+=c
        offsets[kind]={'point_model':point,'bootstrap_models':samples}
    for kind in VARIANTS:
        X=input_rows(panel,histories[kind],ids);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
        point=solve(X,Y,rw,'partial',n);model_check(saved['variants'][kind]['point_model'],point);samples=[];checks+=2
        for b,draw in enumerate(starts):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
            ref=solve(X[indices],Y[indices],rw[indices],'partial',n);model_check(saved['variants'][kind]['bootstrap_models'][b],ref);samples.append(ref);checks+=2
        models[kind]={'point_model':point,'bootstrap_models':samples}
    return models,offsets,checks,offset_checks
