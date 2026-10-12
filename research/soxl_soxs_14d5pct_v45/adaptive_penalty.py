"""Daily macro-block regularization chosen only from mature as-issued errors."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v44'))
from conditional_calibration import *
from conditional_calibration import fit_condition as original_fit
SOURCE=Path(__file__).resolve().parent
CANDIDATES=['plain','strong','medium','weak']
STRENGTH={'plain':None,'strong':2000.,'medium':200.,'weak':20.}
ROUTES=['adaptive_band','adaptive_point','plain_band','strong_band','medium_band','weak_band','premarket_only','cash','double_cost']


def fit_candidate(X,Y,w,n,kind):
    if kind=='plain':return original_fit(X[:,[0,1,10]],Y,w,n)
    if kind=='weak':return original_fit(X,Y,w,n)
    w=np.asarray(w,float);w=w*n/w.sum();keep=w>0;X=X[keep];Y=Y[keep];w=w[keep]
    p=10;clocks=X[:,-1].astype(int);assert np.array_equal(clocks,X[:,-1]) and set(clocks)==set(range(4))
    means=[];scales=[];intercepts=[];Z=np.zeros((len(X),p));R=np.zeros_like(Y)
    for c in range(4):
        mask=clocks==c;v=w[mask];x=X[mask,:p];y=Y[mask];center=np.average(x,axis=0,weights=v)
        scale=np.sqrt(np.average((x-center)**2,axis=0,weights=v));scale[scale<1e-12]=1
        intercept=np.average(y,axis=0,weights=v);Z[mask]=(x-center)/scale;R[mask]=y-intercept
        means.append(center);scales.append(scale);intercepts.append(intercept)
    A=(np.eye(4)[clocks,:,None]*Z[:,None,:]).reshape(len(Z),4*p)
    P=np.kron(np.eye(4)-.1875*np.ones((4,4)),np.diag([20.,20.]+[STRENGTH[kind]]*8))
    beta=np.linalg.lstsq(np.vstack((A*np.sqrt(w)[:,None],np.linalg.cholesky(P).T)),np.vstack((R*np.sqrt(w)[:,None],np.zeros((len(P),2)))),rcond=None)[0]
    return {'kind':'macro_partial','input_center':np.array(means).tolist(),'input_scale':np.array(scales).tolist(),'intercepts':np.array(intercepts).tolist(),'coefficients':beta.reshape(4,p,2).tolist()}


def candidate_prediction(model,x):
    return forecast_condition(model,np.atleast_2d(x)[:,[0,1,10]] if model['kind']=='partial' else x)


def date_weights(parents):
    n=len(parents);dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(p['members']) for p in parents]);w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    return dw,w,groups


def choose(errors,w):
    # One choice across the two ETF outputs and all clocks. Strongest wins exact ties.
    assert errors.shape[1:]==(4,2) and np.isfinite(errors).all();w=np.asarray(w,float);assert np.all(w>=0)
    if w.sum()==0:return 'plain',{k:None for k in CANDIDATES}
    loss=np.sum(w[:,None]*np.mean(errors**2,axis=2),axis=0)/w.sum();j=int(np.argmin(loss))
    return CANDIDATES[j],dict(zip(CANDIDATES,loss.tolist()))


def fit_day(panel,raw_history,prediction_history,reg,at):
    parents,fb=pool(reg,raw_history,at);ids=[t for r in parents for t in r['members']];n=len(parents)
    eligible={t:1 if h is not None and all(h[k] is not None for k in CANDIDATES) else None for t,h in prediction_history.items()}
    selected,sfb=pool(reg,eligible,at);sids=[t for r in selected for t in r['members']];ready=n>=60;sr=len(selected)>=60
    d={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'members':ids,'feedback':fb,'selector_parents':selected,'selector_members':sids,'selector_feedback':sfb,'selector_ready':sr,'ready':ready,'models':{},'selected':'plain','losses':None,'bootstrap_selected':[]}
    if not ready:return d
    dw,w,groups=date_weights(parents);X=np.array([[*raw_history[t],*[panel.loc[t,c] for c in MACRO],CLOCKS.index(panel.loc[t,'clock'])] for t in ids],float)
    truth=panel.loc[ids,['C','D']].to_numpy(float);Y=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-X[:,:2]
    assert {r['date'] for r in selected}<={r['date'] for r in parents},'Selector requires complete candidate-history suffix'
    union=sorted({r['date'] for r in parents+selected});lookup={day:i for i,day in enumerate(union)};counts,starts,seed=block_counts(len(union),at,10)
    date_union=np.array([lookup[r['date']] for r in parents]);d.update(day_weights=dw,row_weights=w,row_groups=groups,union_dates=union,block_starts=starts,seed=seed)
    if sr:
        sdw,sw,sg=date_weights(selected);sd_union=np.array([lookup[r['date']] for r in selected]);t=panel.loc[sids,['C','D']].to_numpy(float);st=np.column_stack((t[:,0]+t[:,1],t[:,0]-t[:,1]))
        errors=np.array([[prediction_history[t][k] for k in CANDIDATES] for t in sids])-st[:,None,:]
        d['selected'],d['losses']=choose(errors,sw);d.update(selector_day_weights=sdw,selector_row_weights=sw,selector_groups=sg)
        d['bootstrap_selected']=[choose(errors,sw*c[sd_union][sg])[0] for c in counts]
    else:d['bootstrap_selected']=['plain']*199
    for kind in CANDIDATES:
        d['models'][kind]={'point_model':fit_candidate(X,Y,w,n,kind),'bootstrap_models':[fit_candidate(X,Y,w*c[date_union][groups],n,kind) for c in counts]}
    return d


def issue(panel,at,raw,daily):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={};x=None
    if ready:x=np.r_[raw,[panel.loc[at,c] for c in MACRO],CLOCKS.index(panel.loc[at,'clock'])]
    for kind in CANDIDATES:
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            models=daily['models'][kind];point=np.array(raw)+candidate_prediction(models['point_model'],x)[0];draws=np.array([raw+candidate_prediction(m,x)[0] for m in models['bootstrap_models']]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind]=info
    adaptive={'ready':bool(ready),'point':None,'lower':None}
    if ready:
        point=np.array(out[daily['selected']]['point']);draws=np.array([out[k]['bootstrap_predictions'][i] for i,k in enumerate(daily['bootstrap_selected'])]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
        adaptive.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
    out['adaptive']=adaptive;return out


def events(at,date,clock,estimates,feedback):
    primary=decide(estimates['adaptive'],feedback,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'adaptive_band':primary,'adaptive_point':decide(estimates['adaptive'],feedback,False),**{k+'_band':decide(estimates[k],feedback,True) for k in CANDIDATES},'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}
