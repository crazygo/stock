"""Clock slopes shrink toward their common mean while dates retain unit weight."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v31'))
from shared_error import (core,np,pd,read,dump,sha,labels,prepare_inputs,base_forecast,day_cutoff,
    restrict_market,development_gate,CLOCKS,tag,columns,issue,registry,pool,block_counts,decide)

VARIANTS=['partial','independent','shared']
ROUTES=['partial_band','partial_point','independent_band','shared_band','premarket_only','source_v27','cash','double_cost']


def encode(panel,history,ids):
    return np.asarray([[*history[t],CLOCKS.index(panel.loc[t,'clock'])] for t in ids],float)


def penalty(kind):
    if kind=='shared':return 20.*np.eye(2)
    if kind=='independent':return 5.*np.eye(8)
    assert kind=='partial'
    # 20||mean(beta)||^2 + 20*sum||beta_c-mean(beta)||^2.
    return np.kron(20.*np.eye(4)-3.75*np.ones((4,4)),np.eye(2))


def fit_map(X,Y,w,kind,n_dates):
    w=np.asarray(w,float);w=w*n_dates/w.sum();keep=w>0;X=X[keep];Y=Y[keep];w=w[keep]
    clocks=X[:,2].astype(int);assert np.array_equal(clocks,X[:,2]) and set(clocks)==set(range(4))
    means=[];scales=[];intercepts=[];Z=np.zeros((len(X),2));R=np.zeros_like(Y)
    for c in range(4):
        mask=clocks==c;v=w[mask];x=X[mask,:2];y=Y[mask]
        center=np.average(x,axis=0,weights=v);scale=np.sqrt(np.average((x-center)**2,axis=0,weights=v));scale[scale<1e-12]=1.
        intercept=np.average(y,axis=0,weights=v);Z[mask]=(x-center)/scale;R[mask]=y-intercept
        means.append(center);scales.append(scale);intercepts.append(intercept)
    A=Z if kind=='shared' else (np.eye(4)[clocks,:,None]*Z[:,None,:]).reshape(len(Z),8)
    P=penalty(kind);L=np.linalg.cholesky(P);augmented=np.vstack((A*np.sqrt(w)[:,None],L.T))
    target=np.vstack((R*np.sqrt(w)[:,None],np.zeros((len(P),2))))
    beta=np.linalg.lstsq(augmented,target,rcond=None)[0]
    coefficient=np.repeat(beta[None,:,:],4,axis=0) if kind=='shared' else beta.reshape(4,2,2)
    return {'kind':kind,'input_center':np.array(means).tolist(),'input_scale':np.array(scales).tolist(),
        'intercepts':np.array(intercepts).tolist(),'coefficients':coefficient.tolist()}


def forecast(model,x):
    x=np.atleast_2d(x);out=[]
    for row in x:
        c=int(row[2]);assert c==row[2] and c in range(4)
        z=(row[:2]-model['input_center'][c])/model['input_scale'][c]
        out.append(z@np.array(model['coefficients'][c])+model['intercepts'][c])
    return np.array(out)


def fit_day(panel,history,reg,at):
    parents,fb=pool(reg,history,at);ids=[t for r in parents for t in r['members']];n=len(parents)
    receipt={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,
        'ready':n>=60,'variants':{}}
    if n<60:return receipt
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(r['members']) for r in parents])
    w=np.concatenate([np.repeat(dw[j]/len(r['members']),len(r['members'])) for j,r in enumerate(parents)])
    X=encode(panel,history,ids);raw=X[:,:2];truth=panel.loc[ids,['C','D']].to_numpy(float)
    Y=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-raw
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    counts,starts,seed=block_counts(n,at,10);receipt.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in VARIANTS:
        receipt['variants'][kind]={'point_model':fit_map(X,Y,w,kind,n),
            'bootstrap_models':[fit_map(X,Y,w*c[groups],kind,n) for c in counts]}
    return receipt


def issue_hierarchical(panel,at,raw,daily):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={}
    for kind in VARIANTS:
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(panel,{at:raw},[at])[0];models=daily['variants'][kind]
            point=raw+forecast(models['point_model'],x)[0]
            samples=np.asarray([raw+forecast(m,x)[0] for m in models['bootstrap_models']])
            radius=max(0.,float(np.quantile(np.max(samples-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=samples.tolist())
        out[kind]=info
    return out


def events(at,date,clock,estimates,feedback,old):
    primary=decide(estimates['partial'],feedback,True);point=decide(estimates['partial'],feedback,False)
    cash={'ready':True,'action':'CASH','allocation':0.}
    out={'partial_band':primary,'partial_point':point,'independent_band':decide(estimates['independent'],feedback,True),
        'shared_band':decide(estimates['shared'],feedback,True),'premarket_only':primary if clock==CLOCKS[0] else cash,
        'source_v27':old,'cash':cash,'double_cost':primary}
    return {n:{**e,'at':at,'date':date} for n,e in out.items()}
