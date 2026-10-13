"""Fixed quadratic map and exact block solution of partially shared Ridge."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/soxl_soxs_14d5pct_v71'))
from transfer_learning import (core, pd, read, dump, sha, labels, day_cutoff,
    registry, fit_day, issue, decide, development_gate, CLOCKS, tag,
    PAIRS, FIELDS, load_market, matched_registry, mature_prefix, training_arrays)
SOURCE = Path(__file__).resolve().parent
PARENT = ROOT/'.cache/soxl_soxs_14d5pct_v71/runs/development'
VARIANTS = ['partial','independent','pooled']
ROUTES = ['quadratic_'+k+'_'+s for k in VARIANTS for s in ['band','point']] + [
    'linear_band','linear_point','premarket_only','cash','double_cost']


def polynomial(x, degree=2):
    x=np.asarray(x,float)
    if degree==1:return x.copy()
    if degree!=2 or x.ndim!=2:raise ValueError('Only registered degree1/2 arrays supported')
    left,right=np.triu_indices(x.shape[1])
    return np.column_stack([x,x[:,left]*x[:,right]])


def normalize(x,w):
    center=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    return (x-center)/scale,center,scale


def fit_family(X,Y,pair,weights,distinct_dates,degree=2,common=100.,deviation=300.):
    X=np.asarray(X,float);Y=np.asarray(Y,float);pair=np.asarray(pair);w=np.asarray(weights,float)
    if (X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or pair.shape!=(len(X),) or w.shape!=(len(X),)
        or not len(X) or not np.isfinite(X).all() or not np.isfinite(Y).all()
        or not np.isfinite(w).all() or (w<=0).any() or not isinstance(distinct_dates,int)
        or distinct_dates<1 or common<=0 or deviation<=0):raise ValueError('Complete rows and positive date/penalty weights required')
    w=w*distinct_dates/w.sum();names=np.unique(pair).tolist();mass=[];original_mean=[];original_scale=[]
    centers=[];scales=[];intercepts=[];grams=[];rhs=[]
    for name in names:
        mask=pair==name;v=w[mask];nu=v.sum()/distinct_dates
        z,mean,sd=normalize(X[mask],v);z,center,scale=normalize(polynomial(z,degree),v)
        intercept=np.average(Y[mask],axis=0,weights=v);residual=Y[mask]-intercept
        mass.append(nu);original_mean.append(mean);original_scale.append(sd)
        centers.append(center);scales.append(scale);intercepts.append(intercept)
        grams.append(z.T@(v[:,None]*z)/nu);rhs.append(z.T@(v[:,None]*residual)/nu)
    mass=np.array(mass);grams=np.array(grams);rhs=np.array(rhs);width=grams.shape[1];I=np.eye(width)
    # Each block is G_p + deviation I. Eliminating beta_p leaves one SPD mean system.
    inverses=np.array([cho_solve(cho_factor(g+deviation*I,lower=True),I) for g in grams])
    inv_rhs=np.einsum('kij,kjl->kil',inverses,rhs)
    mean_matrix=I-(deviation-common)*np.einsum('k,kij->ij',mass,inverses)
    mean_rhs=np.einsum('k,kij->ij',mass,inv_rhs)
    mean_beta=cho_solve(cho_factor(mean_matrix,lower=True),mean_rhs)
    partial=inv_rhs+(deviation-common)*np.einsum('kij,jl->kil',inverses,mean_beta)
    independent=np.array([cho_solve(cho_factor(g+common*I,lower=True),b) for g,b in zip(grams,rhs)])
    pooled=cho_solve(cho_factor(np.einsum('k,kij->ij',mass,grams)+common*I,lower=True),np.einsum('k,kij->ij',mass,rhs))
    return {'degree':degree,'names':names,'distinct_dates':distinct_dates,'common':common,'deviation':deviation,
            'weights':w,'pair_mass':mass,'original_center':np.array(original_mean),'original_scale':np.array(original_scale),
            'centers':np.array(centers),'scales':np.array(scales),'intercepts':np.array(intercepts),
            'coefficients':{'partial':partial,'independent':independent,'pooled':np.repeat(pooled[None,:,:],len(names),axis=0)}}


def forecast(family,X,names,mode):
    X=np.asarray(X,float);names=np.asarray(names)
    if X.ndim!=2 or names.shape!=(len(X),) or not np.isfinite(X).all() or mode not in VARIANTS:raise ValueError('Finite explicit query required')
    out=[]
    for row,name in zip(X,names):
        i=family['names'].index(name)
        z=(row-np.asarray(family['original_center'])[i])/np.asarray(family['original_scale'])[i]
        mapped=polynomial(z[None,:],family['degree'])[0]
        mapped=(mapped-np.asarray(family['centers'])[i])/np.asarray(family['scales'])[i]
        out.append(np.asarray(family['intercepts'])[i]+mapped@np.asarray(family['coefficients'][mode])[i])
    return np.asarray(out)


def fit_clock(panels,parents,clock):
    if len(parents)<120:return None,[]
    X,Y,names,w,ids=training_arrays(panels,parents,clock)
    if len(ids)<120:return None,ids
    return fit_family(X,Y,names,w,len(ids)),ids


def raw_prediction(family,row,kind):
    if family is None or not row.feature_ready:return None
    cd=forecast(family,row[FIELDS].to_numpy(float)[None,:],['semiconductor'],kind)[0]
    return np.array([cd[0]+cd[1],cd[0]-cd[1]])


def verify_parent():
    folder=ROOT/'research/soxl_soxs_14d5pct_v71'
    m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    assert read(PARENT/'verification.json')['passed'] and read(PARENT/'input_audit.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')
