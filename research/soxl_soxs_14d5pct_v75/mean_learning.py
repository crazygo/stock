"""Causal own-pair intercept as a full forecasting and capital baseline."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v74'))
from block_learning import (core,pd,read,dump,sha,labels,day_cutoff,registry,fit_day,
    issue,decide,development_gate,CLOCKS,tag,PAIRS,FIELDS,load_market,matched_registry,mature_prefix)
SOURCE=Path(__file__).resolve().parent
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v74/runs/development'
VARIANTS=['mean']
CONTROL_MAP={'block_band':'block_partial_band','block_point':'block_partial_point',
             'uniform_band':'uniform_band','uniform_point':'uniform_point',
             'old_quadratic_band':'old_quadratic_band','old_quadratic_point':'old_quadratic_point',
             'linear_band':'linear_band','linear_point':'linear_point'}
ROUTES=['mean_band','mean_point','mean_raw']+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def fit_mean(values):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Complete finite C/D observations required')
    n=len(values);w=2.**(-np.arange(n)[::-1]/126.);w*=n/w.sum()
    return {'real_dates':n,'weights':w,'mean_cd':np.average(values,axis=0,weights=w)}


def fit_clock(panels,parents,clock):
    if len(parents)<120:return None,[]
    panel=panels['semiconductor'];ids=[at for d in parents for at in d['members'] if panel.loc[at,'clock']==clock]
    if len(ids)<120:return None,ids
    return fit_mean(panel.loc[ids,['C','D']].to_numpy(float)),ids


def raw_prediction(model,row,kind='mean'):
    if kind!='mean':raise ValueError('Only the registered causal mean is supported')
    if model is None or not row.feature_ready:return None
    c,d=np.asarray(model['mean_cd'],float)
    return np.array([c+d,c-d])


def raw_decision(raw,estimate,feedback):
    # Match calibrated readiness so raw-vs-calibrated actions differ only in scores.
    info={'ready':estimate['ready'] and raw is not None,'point':raw,'lower':raw}
    return decide(info,feedback,False)


def verify_parent():
    folder=ROOT/'research/soxl_soxs_14d5pct_v74';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    assert read(PARENT/'verification.json')['passed'] and read(PARENT/'input_audit.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')
