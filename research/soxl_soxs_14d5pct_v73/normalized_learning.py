"""Fixed representation-width normalization of the v72 partial-sharing penalty."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/soxl_soxs_14d5pct_v72'))
from quadratic_learning import (core,pd,read,dump,sha,labels,day_cutoff,registry,fit_day,
    issue,decide,development_gate,CLOCKS,tag,PAIRS,FIELDS,load_market,matched_registry,
    mature_prefix,training_arrays,polynomial,fit_family,forecast)
SOURCE=Path(__file__).resolve().parent
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v72/runs/development'
VARIANTS=['partial','independent','pooled']
ROUTES=['normalized_'+k+'_'+s for k in VARIANTS for s in ['band','point']]+[
    'old_quadratic_band','old_quadratic_point','linear_band','linear_point','premarket_only','cash','double_cost']


def penalties(columns,reference_columns=18):
    if not isinstance(columns,int) or not isinstance(reference_columns,int) or columns<1 or reference_columns<1:
        raise ValueError('Positive registered representation widths required')
    multiplier=columns/reference_columns
    return {'multiplier':multiplier,'common':100.*multiplier,'deviation':300.*multiplier,
            'columns':columns,'reference_columns':reference_columns}


def normalized_fit(X,Y,names,w,n):
    settings=penalties(polynomial(np.zeros((1,len(FIELDS)))).shape[1])
    assert np.asarray(X).shape[1]==len(FIELDS)==18
    model=fit_family(X,Y,names,w,n,common=settings['common'],deviation=settings['deviation'])
    model['normalization']=settings
    return model


def fit_clock(panels,parents,clock):
    if len(parents)<120:return None,[]
    X,Y,names,w,ids=training_arrays(panels,parents,clock)
    if len(ids)<120:return None,ids
    return normalized_fit(X,Y,names,w,len(ids)),ids


def raw_prediction(family,row,kind):
    if family is None or not row.feature_ready:return None
    cd=forecast(family,row[FIELDS].to_numpy(float)[None,:],['semiconductor'],kind)[0]
    return np.array([cd[0]+cd[1],cd[0]-cd[1]])


def verify_parent():
    folder=ROOT/'research/soxl_soxs_14d5pct_v72';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    assert read(PARENT/'verification.json')['passed'] and read(PARENT/'input_audit.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')
