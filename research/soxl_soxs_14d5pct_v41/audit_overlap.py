"""Independent explicit indicator masks and summation, no prefix-array reuse."""
from overlap_learning import np,pd,core,CLOCKS,day_cutoff,VARIANTS
from audit_hierarchical import input_rows,solve,model_check,predict
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near


def reference_uniqueness(starts,ends,saved):
    a=np.array(starts,dtype=int);b=np.array(ends,dtype=int)
    if not len(a):assert saved=={'offset':None,'count_runs':[],'uniqueness':[]};return np.array([])
    assert a.min()>=0 and np.all(a<=b)
    times=np.arange(a.min(),b.max()+1)
    mask=(a[:,None]<=times)&(b[:,None]>=times);counts=mask.sum(axis=0)
    observed=np.repeat([r[0] for r in saved['count_runs']],[r[1] for r in saved['count_runs']]);assert saved['offset']==int(a.min())
    np.testing.assert_array_equal(observed,counts)
    inv=np.zeros(len(times));good=counts>0;inv[good]=1./counts[good]
    u=np.array([np.mean(inv[row]) for row in mask]);close(saved['uniqueness'],u)
    return u


def base_check(panel,ids,at,model,rec,cols,kind):
    w=np.array([2.**((j-len(ids)+1)/126) for j in range(len(ids))])
    if kind in ['unique','base_only']:
        u=reference_uniqueness(panel.loc[ids,'span_start'],panel.loc[ids,'span_end'],rec['overlap']);w*=u
    if len(w):w*=len(w)/w.sum()
    info=rec['fit'];close(info['parent_weights'],w);near(info['weight_sum'],w.sum())
    near(info['parent_effective_n'],w.sum()**2/(w@w) if len(w) else 0.)
    return weighted_equations(panel,ids,at,model,info,cols,w)


def weighted_equations(panel,ids,at,model,info,columns,w):
    from scipy.special import expit
    assert info['ready']==(len(ids)>=120)
    if len(ids)<120:assert model is None;return {},0
    X=panel.loc[ids,columns].to_numpy(float);C=panel.loc[ids,'C'].to_numpy(float);D=panel.loc[ids,'D'].to_numpy(float);M=abs(D)
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1
    close(model['scaler'].mean_,center);close(model['scaler'].scale_,scale);close(info['center'],center);close(info['scale'],scale)
    near(info['means']['C'],np.average(C,weights=w));near(info['means']['D'],np.average(D,weights=w))
    Z=(X-center)/scale;heads=model['heads'];checks=0
    for target,y in [('C',C),('M',M)]:
        zm=np.average(Z,axis=0,weights=w);ym=np.average(y,weights=w);A=Z-zm
        coef=np.linalg.solve(A.T@(w[:,None]*A)+100*np.eye(len(columns)),A.T@(w*(y-ym)));intercept=ym-zm@coef
        close(heads[target].coef_,coef);near(heads[target].intercept_,intercept);close(info['heads'][target]['coefficients'],coef);near(info['heads'][target]['intercept'],intercept);checks+=1
    cw=w*M
    if cw.sum()>0:cw*=len(w)/cw.sum()
    close(info['classification_weights'],cw);q=float(np.average(D>0,weights=cw)) if cw.sum()>0 else .5
    near(model['q'],q);near(info['fallback_q'],q)
    if len(np.unique(D[cw>0]>0))==2:
        h=heads['q'];residual=cw*(expit(Z@h.coef_[0]+h.intercept_[0])-(D>0));gradient=Z.T@residual+h.coef_[0]/.1
        assert max(abs(gradient).max(),abs(residual.sum()))<1e-6*max(1,len(w));close(info['heads']['q']['coefficients'],h.coef_);close(info['heads']['q']['intercept'],h.intercept_);checks+=1
    else:assert 'q' not in heads
    if not panel.loc[at,'feature_ready']:return {},checks
    x=(panel.loc[at,columns].to_numpy(float)-center)/scale;c=float(x@heads['C'].coef_+heads['C'].intercept_);m=max(0.,float(x@heads['M'].coef_+heads['M'].intercept_))
    if 'q' in heads:q=float(expit(x@heads['q'].coef_[0]+heads['q'].intercept_[0]))
    d=m*(2*q-1);return {'C':c,'D':d,'M':m,'q':q,'SOXL':c+d,'SOXS':c-d},checks


def daily_check(panel,market,histories,at,saved):
    from run_planning import json_ready
    import json
    canon=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    common={t:histories['baseline'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['baseline'])}
    parents,fb=reference_pool(panel,market,common,at);assert canon(saved['parents'])==canon(parents) and canon(saved['feedback'])==canon(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==day_cutoff(at);n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['variants']=={};return {},0
    starts=[min(int(panel.loc[t,'span_start']) for t in p['members']) for p in parents];ends=[max(int(panel.loc[t,'span_end']) for t in p['members']) for p in parents]
    assert saved['span_starts']==starts and saved['span_ends']==ends;u=reference_uniqueness(starts,ends,saved['overlap'])
    groups=[j for j,p in enumerate(parents) for t in p['members']];assert saved['row_groups']==groups
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    draws=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==draws.tolist()
    bydate=[np.flatnonzero(np.array(groups)==j) for j in range(n)];models={};checks=0
    for k in VARIANTS:
        dw=np.array([2.**((j-n+1)/63) for j in range(n)])
        if k in ['unique','cal_only']:dw*=u
        dw*=n/dw.sum();rw=np.array([dw[j]/len(p['members']) for j,p in enumerate(parents) for t in p['members']])
        close(saved['variants'][k]['day_weights'],dw);close(saved['variants'][k]['row_weights'],rw)
        X=input_rows(panel,histories[k],ids);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
        point=solve(X,Y,rw,'partial',n);model_check(saved['variants'][k]['point_model'],point);samples=[];checks+=2
        for i,draw in enumerate(draws):
            days=[(int(t)+j)%n for t in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
            ref=solve(X[indices],Y[indices],rw[indices],'partial',n);model_check(saved['variants'][k]['bootstrap_models'][i],ref);samples.append(ref);checks+=2
        models[k]={'point_model':point,'bootstrap_models':samples}
    return models,checks
