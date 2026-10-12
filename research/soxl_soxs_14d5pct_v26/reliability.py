"""As-issued residual learning with paired chronological block resampling."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v25'))
from planning import core,day_cutoff,restrict_market,development_gate,WeightedScale,weights
from early_training import training_dates
from adaptive_blend import STATE_COLUMNS

REPLICATES=199;BLOCK=10;ALPHA=20.;QUANTILE=.9
ROUTES=['context_band','context_point','score_band','score_point','intercept_band','context_iid_band',
        'source_early','source_no_risk','cash','double_cost']
VARIANTS={'context':(True,True,10),'score':(True,False,10),'intercept':(False,False,10),'context_iid':(True,True,1)}


def raw_vector(record):
    raw=record['forecasts']['early_base_cal']['event']['raw']
    if not raw:return None
    out=np.array([raw[s] for s in core.ETFS],float)
    return out if np.isfinite(out).all() else None


def rows(panel,history,at):
    dates,feedback=training_dates(panel,day_cutoff(at));eligible=set(dates['early'])
    ids=[t for t in sorted(history) if t<day_cutoff(at) and panel.loc[t,'date'] in eligible and history[t] is not None][-120:]
    return ids,feedback


def design(panel,history,ids,score=True,context=True):
    parts=[]
    if score:parts.append(np.array([history[t] for t in ids]))
    if context:parts.append(panel.loc[ids,STATE_COLUMNS].to_numpy(float))
    return np.ascontiguousarray(np.column_stack(parts)) if parts else np.empty((len(ids),0))


def fit_error(X,Y,w):
    w=np.asarray(w,float);w=w*len(X)/w.sum()
    if X.shape[1]:
        scale=WeightedScale(X,w);Z=scale.transform(X)
        model=Ridge(alpha=ALPHA,solver='svd').fit(Z,Y,sample_weight=w)
        return {'center':scale.mean_.tolist(),'scale':scale.scale_.tolist(),
            'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
    return {'center':[],'scale':[],'coef':[[],[]],'intercept':np.average(Y,axis=0,weights=w).tolist()}


def predict_error(model,x):
    if not len(x):return np.array(model['intercept'])
    return (np.asarray(x)-model['center'])/model['scale']@np.array(model['coef']).T+model['intercept']


def block_counts(n,at,length):
    seed=826+at.tz_convert(core.ET).date().toordinal();rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(REPLICATES,(n+length-1)//length))
    indices=((starts[:,:,None]+np.arange(length))%n).reshape(REPLICATES,-1)[:,:n]
    return np.array([np.bincount(i,minlength=n) for i in indices]),starts,seed


def estimate(panel,history,at,raw,ids,kind):
    score,context,length=VARIANTS[kind];info={'ready':False,'members':ids,'variant':kind,'point':None,'lower':None}
    if raw is None or len(ids)<60 or not panel.loc[at,'feature_ready']:return info
    X=design(panel,history,ids,score,context);now={at:raw};x=design(panel,now,[at],score,context)[0]
    base=np.array([history[t] for t in ids]);truth=panel.loc[ids,['C','D']].to_numpy(float)
    Y=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-base
    assert np.isfinite(X).all() and np.isfinite(x).all() and np.isfinite(Y).all()
    w=weights(len(ids),63);model=fit_error(X,Y,w);point=raw+predict_error(model,x)
    counts,starts,seed=block_counts(len(ids),at,length);samples=[]
    for count in counts:samples.append(raw+predict_error(fit_error(X,Y,w*count),x))
    samples=np.array(samples);deviation=np.max(samples-point,axis=1)
    radius=max(0.,float(np.quantile(deviation,QUANTILE,method='higher')))
    info.update(ready=True,weights=w.tolist(),point_model=model,point=point.tolist(),
        bootstrap_predictions=samples.tolist(),block_starts=starts.tolist(),seed=seed,
        radius=radius,lower=(point-radius).tolist(),length=length)
    return info


def decide(info,feedback,use_band=True):
    e={'ready':False,'action':'CASH','allocation':0.,'values':None}
    if not info['ready'] or feedback['expired_unresolved_prefix']:return e
    v=np.array(info['lower' if use_band else 'point']);j=0 if v[0]>=v[1] else 1
    e.update(ready=True,action=core.ETFS[j] if v[j]>1e-12 else 'CASH',allocation=float(v[j]>1e-12),
        values=v.tolist(),top_symbol=core.ETFS[j],buy_advantage=float(v[j]))
    return e
