"""Paired profit/loss classes, as-issued probability calibration and joint utility."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v21'))
from early_training import core,training_dates,day_cutoff,restrict_market,development_gate,BASE_COLUMNS,REGIME_COLUMNS
from adaptive_blend import STATE_COLUMNS

COLUMNS=BASE_COLUMNS+REGIME_COLUMNS
PARAMS=dict(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,learning_rate=.05,l2_regularization=5,early_stopping=False,random_state=814)
ROUTES=['joint_hgb','joint_logit','joint_raw','joint_global','prior_only','full_size','risk_neutral','source_early','cash','double_cost']
FRACTIONS=[0.,.25,.5,.75,1.]


def paired_panel(source):
    p=source.copy();p['rL']=np.expm1(p.C+p.D);p['rS']=np.expm1(p.C-p.D)
    known=np.isfinite(p[['rL','rS']]).all(axis=1)
    p['joint_class']=-1
    p.loc[known,'joint_class']=(2*p.loc[known,'rL'].gt(0).astype(int)+p.loc[known,'rS'].gt(0).astype(int))
    return p


def weights(n,half):
    w=2.**(-np.arange(n)[::-1]/half)
    return w*n/w.sum() if n else w


class WeightedScale:
    def __init__(self,X,w):
        self.mean_=np.average(X,axis=0,weights=w)
        self.scale_=np.sqrt(np.average((X-self.mean_)**2,axis=0,weights=w))
        self.scale_[self.scale_<1e-12]=1.

    def transform(self,X):
        return (X-self.mean_)/self.scale_


def fit_classifier(X,y,w,kind,minimum):
    info={'ready':len(y)>=minimum,'rows':len(y),'weights':w.tolist(),'classes':sorted(map(int,np.unique(y))),
          'kind':kind,'center':None,'scale':None}
    if not info['ready']:return None,info
    assert np.isfinite(X).all() and np.isfinite(w).all() and (w>0).all() and ((0<=y)&(y<4)).all()
    counts=np.bincount(y,weights=w,minlength=4);prior=counts/counts.sum();scaler=None;model=None
    if kind=='logit':
        scaler=WeightedScale(X,w);X=scaler.transform(X)
        info.update(center=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
    if len(np.unique(y))>1:
        model=(HistGradientBoostingClassifier(**PARAMS) if kind=='hgb' else
               LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,tol=1e-10))
        model.fit(X,y,sample_weight=w)
        if kind=='logit':assert model.n_iter_.max()<2000
    info['prior']=prior.tolist()
    if kind=='logit' and model is not None:info.update(coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist())
    return {'kind':kind,'estimator':model,'scaler':scaler,'prior':prior},info


def predict(model,X):
    if model is None:return None
    if model['scaler'] is not None:X=model['scaler'].transform(X)
    if model['estimator'] is None:return model['prior'].copy()
    values=np.zeros(4);e=model['estimator'];values[e.classes_.astype(int)]=e.predict_proba(X)[0]
    assert np.isfinite(values).all() and abs(values.sum()-1)<1e-10
    return values


def fit_daily(panel,at,history,kind):
    admitted,feedback=training_dates(panel,day_cutoff(at));dates=admitted['early'];rows=panel[panel.date.isin(dates)]
    ids=list(rows.index);y=rows.joint_class.to_numpy(int);X=rows[COLUMNS].to_numpy(float)
    model,info=fit_classifier(X,y,weights(len(rows),126),'hgb' if kind=='hgb' else 'logit',120)
    info.update(dates=list(rows.date),members=ids,columns=COLUMNS)
    raw=predict(model,panel.loc[[at],COLUMNS].to_numpy(float)) if panel.loc[at,'feature_ready'] else None
    eligible=set(ids);old=[h for h in history if h['at'] in eligible and h['raw'] is not None][-120:]
    calX=np.log(np.maximum(np.array([h['raw'] for h in old]),1e-6)) if old else np.empty((0,4))
    caly=panel.loc[[h['at'] for h in old],'joint_class'].to_numpy(int)
    calibrator,cal=fit_classifier(calX,caly,weights(len(old),63),'logit',60)
    cal.update(members=[h['at'] for h in old],dates=list(panel.loc[[h['at'] for h in old],'date']),columns=['log_p0','log_p1','log_p2','log_p3'])
    adjusted=predict(calibrator,np.log(np.maximum(raw,1e-6))[None,:]) if raw is not None else None
    ready=raw is not None and adjusted is not None and not feedback['expired_unresolved_prefix']
    record={'feedback':feedback,'fit':info,'calibration':cal,'raw':raw,'adjusted':adjusted,'ready':bool(ready)}
    return {'base':model,'calibration':calibrator},record,ids


def scenarios(panel,ids,at,probabilities,mode='local'):
    ids=ids[-120:];pool=panel.loc[ids];info={'members':ids,'dates':list(pool.date),'ready':False,
        'reason':'insufficient_history','prior':None,'posterior':None}
    if len(pool)<60 or not panel.loc[at,'feature_ready']:return info
    b=weights(len(pool),63);b/=b.sum();w=b.copy();kernel=None
    if mode!='global':
        X=pool[STATE_COLUMNS].to_numpy(float);Q=panel.loc[at,STATE_COLUMNS].to_numpy(float)
        center=np.average(X,axis=0,weights=b);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=b));scale[scale<1e-12]=1.
        distance=np.mean(((X-Q)/scale)**2,axis=1);bandwidth=max(np.sort(distance)[39],1e-12)
        logw=np.log(b)-distance/(2*bandwidth);w=np.exp(logw-logw.max());w/=w.sum()
        kernel={'center':center.tolist(),'scale':scale.tolist(),'distance':distance.tolist(),'bandwidth':float(bandwidth)}
    info.update(prior=w.tolist(),base_weights=b.tolist(),kernel=kernel,prior_concentration=float(1/(w@w)))
    if 1/(w@w)<20:info['reason']='insufficient_prior_concentration';return info
    y=pool.joint_class.to_numpy(int);mass=np.bincount(y,weights=w,minlength=4)
    p=mass if mode=='prior' else np.asarray(probabilities,float)
    assert p.shape==(4,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-10
    info.update(class_prior=mass.tolist(),probabilities=p.tolist())
    if any(p[k]>0 and mass[k]<=0 for k in range(4)):
        info['reason']='predicted_class_without_observed_payoffs';return info
    multipliers=np.divide(p,mass,out=np.zeros(4),where=mass>0);q=w*multipliers[y]
    assert np.isfinite(q).all() and abs(q.sum()-1)<1e-10
    info.update(posterior=q.tolist(),posterior_concentration=float(1/(q@q)),returns=pool[['rL','rS']].to_numpy(float).tolist())
    if 1/(q@q)<20:info['reason']='insufficient_posterior_concentration';return info
    info.update(ready=True,reason='joint_distribution_ready');return info


def choose(distribution,fractions=FRACTIONS,neutral=False):
    out={'ready':False,'action':'CASH','allocation':0.,'reason':distribution['reason'],'utility':None}
    if not distribution['ready']:return out
    q=np.asarray(distribution['posterior']);r=np.asarray(distribution['returns'])
    choices=[('CASH',0.)]+[(s,f) for f in fractions if f>0 for s in core.ETFS];scores=[0.]
    for s,f in choices[1:]:
        wealth=1+f*r[:,core.ETFS.index(s)];assert np.isfinite(wealth).all() and (wealth>0).all()
        scores.append(float(q@(wealth-1 if neutral else (wealth**-4-1)/-4)))
    best=next(i for i,x in enumerate(scores) if x>=max(scores)-1e-12);symbol,fraction=choices[best]
    out.update(ready=True,action=symbol,allocation=fraction,reason='positive_utility' if fraction else 'cash_best',
               utility={'choices':choices,'scores':scores,'selected_index':best,'neutral':neutral})
    return out


def capital(panel,ids,at,forecast,mode='local',raw=False,fractions=FRACTIONS,neutral=False):
    if not forecast['ready']:return {'ready':False,'action':'CASH','allocation':0.,'reason':'forecast_unready','utility':None,'distribution':None}
    d=scenarios(panel,ids,at,forecast['raw'] if raw else forecast['adjusted'],mode)
    return {**choose(d,fractions,neutral),'distribution':d}
