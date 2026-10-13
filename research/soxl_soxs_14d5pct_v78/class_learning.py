"""Gap-aware multinomial contract forecasts and causal expert probability mixtures."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,expit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v77'))
from history_learning import *
import history_learning as parent
SOURCE=Path(__file__).resolve().parent;CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v78'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v77/runs/development'
SYMBOLS=[s for pair in PAIRS.values() for s in pair[:2]]
PRICE_IDS=[i for i in range(18) if i not in range(11,16)]
DIRECTIONAL={0,1,2,5,6,7,8,9,10}
MODES=['gap','merged'];CLASSES={'gap':4,'merged':3}
CONTROL_MAP={'source_long_price':'long_price_band','source_short_price':'short_price_band',
             'source_long_full':'long_full_band','source_long_mean':'long_mean_band'}
ROUTES=[k+'_'+s for k in MODES for s in ['blend','raw','prior','equal']]+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def verify_parent():
    parent.verify_parents();folder=ROOT/'research/soxl_soxs_14d5pct_v77'
    manifest=read(folder/'evidence/delivery_manifest.json')
    for f,h in manifest['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in manifest['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    for f,h in manifest['input_files_sha256'].items():assert sha(ROOT/manifest['input_root']/f)==h,f
    assert read(PARENT/'verification.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')


def signed_price(X,side):
    X=np.asarray(X,float)
    if X.shape[-1]!=18 or not np.isfinite(X).all() or side not in [-1,1]:raise ValueError('Complete original coordinates and explicit side required')
    return X[...,PRICE_IDS]*np.array([side if i in DIRECTIONAL else 1 for i in PRICE_IDS])


def objective(theta,A,y,w,offset):
    B=theta.reshape(A.shape[1],len(offset));scores=A@B+offset;logs=scores-logsumexp(scores,axis=1)[:,None]
    prob=np.exp(logs);prob[np.arange(len(y)),y]-=1
    return float(-w@logs[np.arange(len(y)),y]+5*np.sum(B*B)),(A.T@(w[:,None]*prob)+10*B).ravel()


def hessian(theta,A,w,offset):
    B=theta.reshape(A.shape[1],len(offset));scores=A@B+offset
    prob=np.exp(scores-logsumexp(scores,axis=1)[:,None]);width,k=B.shape
    covariance=-prob[:,:,None]*prob[:,None,:]
    covariance[:,np.arange(k),np.arange(k)]+=prob
    return np.einsum('ni,nj,nab,n->iajb',A,A,covariance,w,optimize=True).reshape(width*k,width*k)+10*np.eye(width*k)


def fit_distribution(X,y,returns,groups,date_weights,classes):
    X=np.asarray(X,float);y=np.asarray(y,int);returns=np.asarray(returns,float);groups=np.asarray(groups,int)
    dw=np.asarray(date_weights,float)
    if dw.ndim!=1 or not len(dw) or not np.isfinite(dw).all() or (dw<=0).any():raise ValueError('Finite positive date weights required')
    n=len(dw);ess=dw.sum()**2/(dw@dw);dw=dw*(ess/dw.sum());w=np.tile(dw/6,6)
    if X.shape!=(n*6,13) or any(len(a)!=n*6 for a in [y,returns,groups]):raise ValueError('Six rows per real date required')
    if not np.isfinite(X).all() or not np.isfinite(returns).all() or (dw<=0).any() or not set(y)<=set(range(classes)):
        raise ValueError('Finite mature labels and positive weights required')
    assert np.array_equal(groups,np.repeat(np.arange(6),n))
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1
    A=np.column_stack([np.eye(6)[groups],(X-center)/scale])
    pooled=np.bincount(y,weights=w,minlength=classes)/ess
    pooled_sum=np.bincount(y,weights=w*returns,minlength=classes)/ess
    active=np.flatnonzero(pooled>0);offset=np.log(pooled[active]);mapped=np.searchsorted(active,y)
    result=minimize(objective,np.zeros(A.shape[1]*len(active)),args=(A,mapped,w,offset),jac=True,
        method='L-BFGS-B',options={'maxiter':2000,'gtol':1e-9,'ftol':1e-15,'maxls':50})
    theta=result.x.copy();loss,gradient=objective(theta,A,mapped,w,offset);refinements=0
    for _ in range(5):
        if np.max(np.abs(gradient))<=1e-9:break
        step=np.linalg.solve(hessian(theta,A,w,offset),gradient);alpha=1.;oldnorm=np.max(np.abs(gradient))
        for _ in range(40):
            candidate=theta-alpha*step;value,newgrad=objective(candidate,A,mapped,w,offset)
            if value<=loss+64*np.finfo(float).eps*max(1.,abs(loss)) and np.max(np.abs(newgrad))<oldnorm:
                theta,loss,gradient=candidate,value,newgrad;refinements+=1;break
            alpha*=.5
        else:break
    if not np.isfinite(theta).all() or np.max(np.abs(gradient))>1e-6:
        raise ValueError(f'Multinomial optimizer did not converge: {result.message}; gradient={np.max(np.abs(gradient))}')
    priors=[];means=[];masses=[];sums=[]
    for g in range(6):
        keep=groups==g;mass=np.bincount(y[keep],weights=dw,minlength=classes)
        totals=np.bincount(y[keep],weights=dw*returns[keep],minlength=classes)
        denominator=mass+20*pooled;value=np.zeros(classes)
        np.divide(totals+20*pooled_sum,denominator,out=value,where=denominator>0)
        priors.append(denominator/(ess+20));means.append(value);masses.append(mass);sums.append(totals)
    return {'real_dates':n,'date_ess':ess,'date_weights':dw,'weights':w,'center':center,'scale':scale,
        'classes':classes,'active':active,'offset':offset,'coefficients':theta.reshape(A.shape[1],len(active)),
        'pooled_prior':pooled,'pooled_payoff_sum':pooled_sum,'prior':np.array(priors),'payoff_means':np.array(means),
        'own_class_mass':np.array(masses),'own_payoff_sum':np.array(sums),'objective':loss,
        'gradient_inf':float(np.max(np.abs(gradient))),'iterations':int(result.nit),'newton_refinements':refinements,'lbfgs_success':bool(result.success)}


def fit_clock(panels,parents,clock):
    ids=[t for p in parents for t in p['members'] if panels['semiconductor'].loc[t,'clock']==clock] if len(parents)>=120 else []
    if len(ids)<120:return {},ids
    n=len(ids);X=[];ys={m:[] for m in MODES};returns=[]
    for pair in PAIRS:
        rows=panels[pair].loc[ids]
        for side,sign in [('L',1),('S',-1)]:
            X.append(signed_price(rows[FIELDS].to_numpy(float),sign));returns.append(rows[side+'_log'].to_numpy(float))
            for mode in MODES:ys[mode].append(rows[side+'_'+mode].to_numpy(int))
    dw=2.**(-np.arange(n)[::-1]/126)
    models={m:fit_distribution(np.concatenate(X),np.concatenate(ys[m]),np.concatenate(returns),np.repeat(np.arange(6),n),dw,CLASSES[m]) for m in MODES}
    return models,ids


def predict_distribution(model,row):
    if model is None or not row.feature_ready:return None
    probs=[]
    for g,sign in [(0,1),(1,-1)]:
        x=signed_price(row[FIELDS].to_numpy(float),sign)
        a=np.r_[np.eye(6)[g],(x-model['center'])/model['scale']];logits=a@model['coefficients']+model['offset']
        p=np.zeros(model['classes']);p[np.array(model['active'],int)]=np.exp(logits-logsumexp(logits));probs.append(p)
    return {'raw':np.array(probs),'prior':np.array(model['prior'])[:2],
            'payoff_means':np.array(model['payoff_means'])[:2]}


def fit_mixture(panel,history,reg,at,mode):
    parents,feedback=mature_prefix(reg,at)
    selected=[p for p in parents if all(history.get(t) is not None for t in p['members'])][-120:]
    n=len(selected);out={'ready':n>=60,'cutoff':day_cutoff(at),'feedback':feedback,'parents':selected,'alpha':None}
    if not n:return out
    dw=2.**(-np.arange(n)[::-1]/63);dw*=n/dw.sum();scores=[]
    for p in selected:
        delta=[]
        for t in p['members']:
            rec=history[t]
            for j,side in enumerate(['L','S']):
                y=int(panel.loc[t,side+'_'+mode]);assert y>=0
                delta.append(np.log(max(rec['raw'][j][y],1e-8))-np.log(max(rec['prior'][j][y],1e-8)))
        scores.append(float(np.mean(delta)))
    odds=float(.1*np.dot(dw,scores));out.update(date_weights=dw,date_logscore_differences=scores,logodds=odds,
        alpha=float(expit(odds)) if n>=60 else None)
    return out


def issue_distribution(raw,daily,variant):
    if raw is None or not daily['ready'] or daily['feedback']['expired_unresolved_prefix']:
        return {'ready':False,'probabilities':None,'values':None}
    alpha={'blend':daily['alpha'],'raw':1.,'prior':0.,'equal':.5}[variant]
    prob=alpha*np.asarray(raw['raw'])+(1-alpha)*np.asarray(raw['prior'])
    values=np.sum(prob*np.asarray(raw['payoff_means']),axis=1)
    return {'ready':True,'probabilities':prob,'values':values,'alpha':alpha}


def choose(info):
    out={'ready':info['ready'],'action':'CASH','allocation':0.,'values':info['values']}
    if info['ready']:
        v=info['values'];j=int(v[1]>v[0]);buy=v[j]>1e-12
        out.update(action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),top_symbol=['SOXL','SOXS'][j],buy_advantage=float(v[j]))
    return out
