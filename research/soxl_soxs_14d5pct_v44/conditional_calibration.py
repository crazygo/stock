"""Factorial macro information in the base model and in daily residual calibration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v42'))
from macro_learning import *
from macro_learning import columns as old_columns,fit_map as old_fit_map,forecast as old_forecast
SOURCE=Path(__file__).resolve().parent
VARIANTS=['macro_context','macro_plain','base_context','base_plain']
ROUTES=['macro_context_band','macro_context_point','macro_plain_band','base_context_band','base_plain_band','source_v35','premarket_only','cash','double_cost']

def columns(clock,kind):return old_columns(clock,'macro' if kind.startswith('macro_') else 'baseline')

def context_columns(kind):return MACRO if kind.endswith('_context') else []

def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));models={};fits={};raws={}
    cache={}
    for k in ['macro','base']:
        m,f=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],k+'_plain'));r=predict(m,same.loc[[at]]);cache[k]=(m,f,r)
    for k in VARIANTS:
        m,f,r=cache[k.split('_')[0]];models[k]=m;fits[k]={'fit':f,'feedback':fb,'raw':r};raws[k]=np.array([r[s] for s in core.ETFS]) if r else None
    return models,fits,raws

def encode_condition(panel,history,ids,kind):
    cols=context_columns(kind)
    return np.asarray([[*history[t],*(float(panel.loc[t,c]) for c in cols),CLOCKS.index(panel.loc[t,'clock'])] for t in ids],float)

def fit_condition(X,Y,w,n_dates):
    p=X.shape[1]-1
    if p==2:return old_fit_map(X,Y,w,'partial',n_dates)
    assert p==10
    w=np.asarray(w,float);w=w*n_dates/w.sum();keep=w>0;X=X[keep];Y=Y[keep];w=w[keep]
    assert np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(w).all()
    clocks=X[:,-1].astype(int);assert np.array_equal(clocks,X[:,-1]) and set(clocks)==set(range(4))
    means=[];scales=[];intercepts=[];Z=np.zeros((len(X),p));R=np.zeros_like(Y)
    for c in range(4):
        mask=clocks==c;v=w[mask];x=X[mask,:p];y=Y[mask]
        center=np.average(x,axis=0,weights=v);scale=np.sqrt(np.average((x-center)**2,axis=0,weights=v));scale[scale<1e-12]=1.
        intercept=np.average(y,axis=0,weights=v);Z[mask]=(x-center)/scale;R[mask]=y-intercept
        means.append(center);scales.append(scale);intercepts.append(intercept)
    A=(np.eye(4)[clocks,:,None]*Z[:,None,:]).reshape(len(Z),4*p)
    P=np.kron(20.*np.eye(4)-3.75*np.ones((4,4)),np.eye(p));L=np.linalg.cholesky(P)
    beta=np.linalg.lstsq(np.vstack((A*np.sqrt(w)[:,None],L.T)),np.vstack((R*np.sqrt(w)[:,None],np.zeros((len(P),2)))),rcond=None)[0]
    return {'kind':'macro_partial','input_center':np.array(means).tolist(),'input_scale':np.array(scales).tolist(),
        'intercepts':np.array(intercepts).tolist(),'coefficients':beta.reshape(4,p,2).tolist()}

def forecast_condition(model,x):
    if model['kind']=='partial':return old_forecast(model,x)
    out=[]
    for row in np.atleast_2d(x):
        c=int(row[-1]);assert c==row[-1] and c in range(4)
        z=(row[:-1]-model['input_center'][c])/model['input_scale'][c]
        out.append(z@np.array(model['coefficients'][c])+model['intercepts'][c])
    return np.array(out)

def common_history(histories):
    return {t:histories['macro_plain'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['macro_plain'])}

def fit_calibration(panel,histories,reg,at):
    parents,fb=pool(reg,common_history(histories),at);ids=[t for p in parents for t in p['members']];n=len(parents)
    d={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,'ready':n>=60,'variants':{}}
    if n<60:return d
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    truth=panel.loc[ids,['C','D']].to_numpy(float);Ytrue=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])
    counts,starts,seed=block_counts(n,at,10);d.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in VARIANTS:
        X=encode_condition(panel,histories[kind],ids,kind);Y=Ytrue-X[:,:2]
        d['variants'][kind]={'point_model':fit_condition(X,Y,w,n),
            'bootstrap_models':[fit_condition(X,Y,w*c[groups],n) for c in counts]}
    return d

def issue(panel,at,raws,daily):
    out={}
    for kind in VARIANTS:
        raw=raws[kind];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode_condition(panel,{at:raw},[at],kind)[0];m=daily['variants'][kind];point=raw+forecast_condition(m['point_model'],x)[0]
            draws=np.array([raw+forecast_condition(z,x)[0] for z in m['bootstrap_models']]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind]=info
    return out

def events(at,date,clock,estimates,fb,old):
    primary=decide(estimates['macro_context'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'macro_context_band':primary,'macro_context_point':decide(estimates['macro_context'],fb,False),
        'macro_plain_band':decide(estimates['macro_plain'],fb,True),'base_context_band':decide(estimates['base_context'],fb,True),
        'base_plain_band':decide(estimates['base_plain'],fb,True),'source_v35':old,
        'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}


def verified_v42():
    root=core.ROOT/'.cache/soxl_soxs_14d5pct_v42/runs/macro_20261013'
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v42/evidence/delivery_manifest.json')
    want={f:h for f,h in delivery['cache_outputs_sha256'].items()}
    for folder in ['training','daily']:
        want.update({str(Path(folder)/f):h for f,h in delivery[folder+'_sha256'].items()})
    for f,h in want.items():assert sha(root/f)==h,f
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root,want
