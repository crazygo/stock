"""Factorial causal-history / price-state experiment, with loss-mass controls."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v71'))
from transfer_learning import *
import transfer_learning as old
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v76'))
from market_extension import load as extended_market
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v77'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v75/runs/development'
VARIANTS=['short_price','long_full','long_price','long_native','long_mean']
ACTIVITY_FIELDS=FIELDS[11:16]
CONTROL_MAP={'short_full_band':'linear_band','short_full_point':'linear_point',
             'short_mean_band':'mean_band','short_mean_point':'mean_point'}
ROUTES=[k+'_'+s for k in VARIANTS for s in ['band','point']]+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def verify_parents():
    folder=ROOT/'research/soxl_soxs_14d5pct_v75';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    for f,h in m['additional_cache_files_sha256'].items():assert sha(ROOT/m['additional_cache_root']/f)==h,f
    other=ROOT/'research/soxl_soxs_14d5pct_v76';d=read(other/'delivery_manifest.json')
    for f,h in d['delivery_files_sha256'].items():assert sha(ROOT/f)==h,f
    inputs=read(other/'evidence/input_manifest.json')
    for f,h in {**inputs['files_sha256'],**inputs['loaded_research_source_sha256']}.items():assert sha(ROOT/f)==h,f
    assert read(PARENT/'verification.json')['passed'] and read(PARENT/'input_audit.json')['passed']
    return {'v75':sha(folder/'evidence/delivery_manifest.json'),'v76':sha(other/'delivery_manifest.json')}


def mask_features(X,kind):
    x=np.asarray(X,float).copy()
    if x.shape[-1]!=len(FIELDS):raise ValueError('Original eighteen coordinates required')
    if not np.isfinite(x).all():raise ValueError('Missing coordinates cannot be zero-filled by the price-only mask')
    if kind in ['short_price','long_price','long_native']:x[...,11:16]=0.
    elif kind!='long_full':raise ValueError('Unknown state variant')
    return x


def fit_state(X,Y,names,w,n,reference_mass,kind):
    if not isinstance(n,int) or n<1 or not np.isfinite(reference_mass) or reference_mass<=0:
        raise ValueError('Positive real count and reference mass required')
    x=mask_features(X,kind);scale=1. if kind=='long_native' else n/reference_mass
    model=old.fit(x,Y,names,w,n,mode='partial',common=100.*scale,deviation=300.*scale)
    v=np.asarray(w,float)
    model.update(variant=kind,reference_mass=float(reference_mass),penalty_scale=float(scale),
                 date_effective_sample_size=float((v.sum()**2/(v@v))/len(np.unique(names))))
    return model


def clock_models(panels,short_panels,parents,short_parents,clock):
    def select(pp,ps):return [t for p in ps for t in p['members'] if pp['semiconductor'].loc[t,'clock']==clock] if len(ps)>=120 else []
    ids=select(panels,parents);shortids=select(short_panels,short_parents)
    reference=max(120,len(shortids));models={};membership={}
    for kind in VARIANTS:
        use=shortids if kind=='short_price' else ids
        source=short_panels if kind=='short_price' else panels
        membership[kind]=use
        if len(use)<120:continue
        n=len(use);w=2.**(-np.arange(n)[::-1]/126.);w*=n/w.sum()
        if kind=='long_mean':
            model={'variant':kind,'distinct_dates':n,'mean_cd':np.average(source['semiconductor'].loc[use,['C','D']].to_numpy(float),axis=0,weights=w),
                   'weights':w,'date_effective_sample_size':float(w.sum()**2/(w@w))}
        else:
            X=np.concatenate([p.loc[use,FIELDS].to_numpy(float) for p in source.values()])
            Y=np.concatenate([p.loc[use,['C','D']].to_numpy(float) for p in source.values()])
            model=fit_state(X,Y,np.repeat(list(source),n),np.tile(w/len(source),len(source)),n,reference,kind)
        models[kind]=model
    return models,membership


def raw_prediction(model,row,kind):
    if model is None or not row.feature_ready:return None
    if kind=='long_mean':cd=np.asarray(model['mean_cd'])
    else:cd=old.predict(model,mask_features(row[FIELDS].to_numpy(float)[None,:],kind),np.array(['semiconductor']))[0]
    return np.array([cd[0]+cd[1],cd[0]-cd[1]])
