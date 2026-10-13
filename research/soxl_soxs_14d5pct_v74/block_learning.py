"""Preserve original-field regularization while normalizing the quadratic block."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v73'))
from normalized_learning import (core,pd,read,dump,sha,labels,day_cutoff,registry,fit_day,
    issue,decide,development_gate,CLOCKS,tag,PAIRS,FIELDS,load_market,matched_registry,
    mature_prefix,training_arrays,polynomial,forecast)
from quadratic_learning import normalize
SOURCE=Path(__file__).resolve().parent
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v73/runs/development'
VARIANTS=['partial','independent','pooled']
CONTROL_MAP={'uniform_band':'normalized_partial_band','uniform_point':'normalized_partial_point',
             'old_quadratic_band':'old_quadratic_band','old_quadratic_point':'old_quadratic_point',
             'linear_band':'linear_band','linear_point':'linear_point'}
ROUTES=['block_'+k+'_'+s for k in VARIANTS for s in ['band','point']]+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def block_factors(widths,reference=18):
    if not widths or not isinstance(reference,int) or reference<1 or any(not isinstance(v,int) or v<1 for v in widths):
        raise ValueError('Positive integer block widths and reference required')
    return np.concatenate([np.full(width,width/reference,dtype=float) for width in widths])


def solve(grams,rhs,mass,factors):
    grams=np.asarray(grams,float);rhs=np.asarray(rhs,float);mass=np.asarray(mass,float);f=np.asarray(factors,float)
    if f.shape!=(grams.shape[1],) or not np.isfinite(f).all() or (f<=0).any():raise ValueError('Positive finite feature penalty factors required')
    # Reparameterize beta=D^-1/2 theta; the shared/own penalty in theta is scalar.
    scale=1/np.sqrt(f);g=grams*scale[None,:,None]*scale[None,None,:];b=rhs*scale[None,:,None];I=np.eye(len(f))
    inv=np.array([cho_solve(cho_factor(a+300*I,lower=True),I) for a in g])
    ib=np.einsum('kij,kjl->kil',inv,b)
    mean_matrix=I-200*np.einsum('k,kij->ij',mass,inv)
    mean=cho_solve(cho_factor(mean_matrix,lower=True),np.einsum('k,kij->ij',mass,ib))
    partial=ib+200*np.einsum('kij,jl->kil',inv,mean)
    independent=np.array([cho_solve(cho_factor(a+100*I,lower=True),v) for a,v in zip(g,b)])
    pooled=cho_solve(cho_factor(np.einsum('k,kij->ij',mass,g)+100*I,lower=True),np.einsum('k,kij->ij',mass,b))
    return {'partial':partial*scale[None,:,None],'independent':independent*scale[None,:,None],
            'pooled':np.repeat((pooled*scale[:,None])[None,:,:],len(mass),axis=0)}


def fit_family(X,Y,pair,weights,distinct_dates,degree=2,factors=None):
    X=np.asarray(X,float);Y=np.asarray(Y,float);pair=np.asarray(pair);w=np.asarray(weights,float)
    if (X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or pair.shape!=(len(X),) or w.shape!=(len(X),)
        or not len(X) or not np.isfinite(X).all() or not np.isfinite(Y).all()
        or not np.isfinite(w).all() or (w<=0).any() or not isinstance(distinct_dates,int) or distinct_dates<1):
        raise ValueError('Complete rows and positive real date weights required')
    widths=[X.shape[1]] if degree==1 else [X.shape[1],X.shape[1]*(X.shape[1]+1)//2]
    f=block_factors(widths) if factors is None else np.asarray(factors,float)
    w=w*distinct_dates/w.sum();names=np.unique(pair).tolist();mass=[];original_mean=[];original_scale=[]
    centers=[];scales=[];intercepts=[];grams=[];rhs=[]
    for name in names:
        mask=pair==name;v=w[mask];nu=v.sum()/distinct_dates
        z,mean,sd=normalize(X[mask],v);z,center,scale=normalize(polynomial(z,degree),v)
        intercept=np.average(Y[mask],axis=0,weights=v);residual=Y[mask]-intercept
        mass.append(nu);original_mean.append(mean);original_scale.append(sd)
        centers.append(center);scales.append(scale);intercepts.append(intercept)
        grams.append(z.T@(v[:,None]*z)/nu);rhs.append(z.T@(v[:,None]*residual)/nu)
    coefficients=solve(grams,rhs,mass,f)
    return {'degree':degree,'names':names,'distinct_dates':distinct_dates,'common':100*f,'deviation':300*f,
            'normalization':{'block_widths':widths,'reference_width':18,'factors':f},
            'weights':w,'pair_mass':np.array(mass),'original_center':np.array(original_mean),'original_scale':np.array(original_scale),
            'centers':np.array(centers),'scales':np.array(scales),'intercepts':np.array(intercepts),'coefficients':coefficients}


def fit_clock(panels,parents,clock):
    if len(parents)<120:return None,[]
    X,Y,names,w,ids=training_arrays(panels,parents,clock)
    if len(ids)<120:return None,ids
    assert X.shape[1]==len(FIELDS)==18
    return fit_family(X,Y,names,w,len(ids)),ids


def raw_prediction(family,row,kind):
    if family is None or not row.feature_ready:return None
    cd=forecast(family,row[FIELDS].to_numpy(float)[None,:],['semiconductor'],kind)[0]
    return np.array([cd[0]+cd[1],cd[0]-cd[1]])


def verify_parent():
    folder=ROOT/'research/soxl_soxs_14d5pct_v73';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    assert read(PARENT/'verification.json')['passed'] and read(PARENT/'input_audit.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')
