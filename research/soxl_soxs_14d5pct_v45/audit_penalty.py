"""Independent joint normal equations and chronological selector reconstruction."""
from adaptive_penalty import core,np,pd,CLOCKS,MACRO,CANDIDATES,STRENGTH
from audit_shared import reference_pool
from audit_condition_calibration import predict as reference_predict,model_check
from verify_reliability import close


def solve(X,Y,w,n,kind):
    if kind=='plain':X=X[:,[0,1,10]]
    w=np.asarray(w)*n/np.sum(w);ok=w>0;X=X[ok];Y=Y[ok];w=w[ok];p=X.shape[1]-1;c=X[:,-1].astype(int);Z=np.zeros((len(X),p));means=[];scales=[]
    for j in range(4):
        ix=np.flatnonzero(c==j);a=X[ix,:p];v=w[ix];mean=np.sum(a*v[:,None],axis=0)/sum(v);sd=np.sqrt(np.sum((a-mean)**2*v[:,None],axis=0)/sum(v));sd[sd<1e-12]=1;means.append(mean);scales.append(sd);Z[ix]=(a-mean)/sd
    A=np.array([[float(clock==j) for j in range(4)]+[Z[i,k] if clock==j else 0. for j in range(4) for k in range(p)] for i,clock in enumerate(c)])
    P=np.zeros((len(A[0]),len(A[0])))
    for k in range(p):
        strength=20. if k<2 else STRENGTH[kind];mean=np.zeros(len(P));mean[4+k::p]=.25;P+=strength*np.outer(mean,mean)
        for j in range(4):
            v=-mean.copy();v[4+j*p+k]+=1;P+=strength*np.outer(v,v)
    beta=np.linalg.solve(A.T@(w[:,None]*A)+P,A.T@(w[:,None]*Y))
    return {'kind':'partial' if kind=='plain' else 'macro_partial','input_center':means,'input_scale':scales,'intercepts':beta[:4],'coefficients':beta[4:].reshape(4,p,2)}


def select_reference(panel,history,parents,counts=None,precomputed=None):
    if len(parents)<60:return 'plain',None
    n=len(parents);w=np.array([2.**((j-n+1)/63) for j in range(n)]);w=w if counts is None else w*counts
    if sum(w)==0:return 'plain',{k:None for k in CANDIDATES}
    errors={k:[] for k in CANDIDATES} if precomputed is None else precomputed
    for row in parents if precomputed is None else []:
        daily={k:[] for k in CANDIDATES}
        for t in row['members']:
            truth=[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']]
            for k in CANDIDATES:daily[k].append(sum((history[t][k][j]-truth[j])**2 for j in range(2))/2)
        for k in CANDIDATES:errors[k].append(sum(daily[k])/len(daily[k]))
    loss={k:float(sum(w*np.array(errors[k]))/sum(w)) for k in CANDIDATES};return min(CANDIDATES,key=loss.get),loss


def daily_check(panel,market,raw_history,history,at,saved):
    from run_planning import json_ready
    import json
    wire=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    parents,fb=reference_pool(panel,market,raw_history,at);eligible={t:1 if h is not None and all(h[k] is not None for k in CANDIDATES) else None for t,h in history.items()};selected,sfb=reference_pool(panel,market,eligible,at)
    for field,want in [('parents',parents),('feedback',fb),('selector_parents',selected),('selector_feedback',sfb)]:assert saved[field]==wire(want),field
    assert pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    ids=[t for r in parents for t in r['members']];sids=[t for r in selected for t in r['members']];assert saved['members']==wire(ids) and saved['selector_members']==wire(sids)
    assert saved['ready']==(len(parents)>=60) and saved['selector_ready']==(len(selected)>=60)
    if not saved['ready']:assert saved['models']=={} and saved['selected']=='plain';return {},0,0
    n=len(parents);wday=np.array([2.**((j-n+1)/63) for j in range(n)]);wday*=n/wday.sum();w=np.array([wday[j]/len(r['members']) for j,r in enumerate(parents) for _ in r['members']]);close(saved['day_weights'],wday);close(saved['row_weights'],w)
    groups=np.repeat(np.arange(n),[len(r['members']) for r in parents]);close(saved['row_groups'],groups)
    union=sorted(set(r['date'] for r in parents+selected));assert saved['union_dates']==wire(union);seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,len(union),size=(199,(len(union)+9)//10));assert saved['block_starts']==starts.tolist()
    counts=[]
    for draws in starts:
        days=[(int(start)+j)%len(union) for start in draws for j in range(10)][:len(union)];counts.append(np.array([days.count(j) for j in range(len(union))]))
    pick,loss=select_reference(panel,history,selected);assert saved['selected']==pick
    if loss is None:assert saved['losses'] is None
    else:
        for k in CANDIDATES:close(saved['losses'][k],loss[k])
        ns=len(selected);dw=np.array([2.**((j-ns+1)/63) for j in range(ns)]);dw*=ns/dw.sum();close(saved['selector_day_weights'],dw);close(saved['selector_row_weights'],[dw[j]/len(r['members']) for j,r in enumerate(selected) for _ in r['members']]);close(saved['selector_groups'],np.repeat(np.arange(ns),[len(r['members']) for r in selected]))
    assert set(r['date'] for r in selected)<=set(r['date'] for r in parents)
    s_ix=[union.index(r['date']) for r in selected];precomputed={k:[] for k in CANDIDATES}
    if len(selected)>=60:
        for r in selected:
            for k in CANDIDATES:
                loss=[]
                for t in r['members']:
                    truth=[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']]
                    loss.append(sum((history[t][k][j]-truth[j])**2 for j in range(2))/2)
                precomputed[k].append(sum(loss)/len(loss))
    for i,c in enumerate(counts):assert saved['bootstrap_selected'][i]==select_reference(panel,history,selected,c[s_ix],precomputed)[0]
    X=np.array([[*raw_history[t],*[panel.loc[t,c] for c in MACRO],CLOCKS.index(panel.loc[t,'clock'])] for t in ids]);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids])-X[:,:2]
    ix=np.array([union.index(r['date']) for r in parents]);models={};checks=0
    for k in CANDIDATES:
        model=solve(X,Y,w,n,k);model_check(saved['models'][k]['point_model'],model);samples=[];checks+=2
        for i,c in enumerate(counts):
            m=solve(X,Y,w*c[ix][groups],n,k);model_check(saved['models'][k]['bootstrap_models'][i],m);samples.append(m);checks+=2
        models[k]={'point_model':model,'bootstrap_models':samples}
    return models,checks,200
