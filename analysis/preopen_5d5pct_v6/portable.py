"""Data-only latest research model artifacts. No pickle loading at inference."""
import gzip,json
import numpy as np
from scipy.special import expit,logit
from common import OUT,sha

def calibrate(cal,p):
    if cal is None:return p
    return expit(cal['intercept']+cal['coefficient']*logit(np.clip(p,1e-5,1-1e-5)))

def raw_probability(artifact,frame):
    x=frame[artifact['features']].to_numpy(copy=True)
    if x.dtype.kind!='f':x=x.astype(float)
    x[~np.isfinite(x)]=np.nan;model=artifact['estimator']
    if model['kind']=='LightGBM':
        from lightgbm import Booster
        return Booster(model_str=model['model_text']).predict(x,num_threads=1)
    impute=model['imputation'];missing=np.isnan(x);x=np.where(missing,np.asarray(impute['statistics'],dtype=x.dtype),x)
    if impute['indicators']:x=np.column_stack([x,missing[:,impute['indicators']]])
    if model['kind']=='LogisticRegression':
        x-=model['center'];x/=model['scale']
        return expit(x@np.asarray(model['coefficient'])+model['intercept'])
    if model['kind']=='ExtraTrees':
        x=x.astype(np.float32)
        out=np.zeros(len(x))
        for tree in model['trees']:
            left=np.asarray(tree['left']);right=np.asarray(tree['right']);feature=np.asarray(tree['feature']);threshold=np.asarray(tree['threshold']);value=np.asarray(tree['positive_probability']);node=np.zeros(len(x),dtype=int)
            active=left[node]>=0
            while active.any():
                ids=np.flatnonzero(active);current=node[ids];go=x[ids,feature[current]]<=threshold[current];node[ids]=np.where(go,left[current],right[current]);active=left[node]>=0
            out+=value[node]
        return out/len(model['trees'])
    raise ValueError('unsupported native estimator')

def probabilities(artifact,frame,phase,variant='top_one'):
    p=calibrate(artifact['candidate_calibrators'][phase],raw_probability(artifact,frame))
    if variant=='top_one':p=calibrate(artifact['top_calibrators'][phase],p)
    return p

def baseline_table(artifact):
    b=artifact['baseline']
    return dict(vol_edges=b['vol_edges'],minute={int(k):v for k,v in b['minute'].items()},stock_minute={(r[0],int(r[1])):r[2] for r in b['stock_minute']},matched={(r[0],int(r[1]),int(r[2])):r[3] for r in b['matched']})

def load(dest,arm):
    manifest_path=OUT/'portable_models/manifest.json'
    if not manifest_path.exists():return None
    manifest=json.loads(manifest_path.read_text());round_id='R00' if dest==OUT else dest.name
    items=[r for r in manifest['models'] if r['round']==round_id and r['arm']==arm]
    if not items:return None
    item=max(items,key=lambda r:r['month']);path=OUT/'portable_models'/item['file']
    if sha(path)!=item['sha256']:raise ValueError('portable artifact SHA mismatch')
    with gzip.open(path,'rt') as f:a=json.load(f)
    if a['schema']!='stock-v6-native-1' or a['round']!=round_id or a['arm']!=arm:raise ValueError('artifact identity mismatch')
    return a,item

def load_upstream():
    path=OUT/'portable_models/manifest.json'
    if not path.exists():return None
    item=json.loads(path.read_text()).get('upstream')
    if not item:return None
    source=OUT/'portable_models'/item['file']
    if sha(source)!=item['sha256']:raise ValueError('upstream portable SHA mismatch')
    with gzip.open(source,'rt') as f:a=json.load(f)
    if a['schema']!='stock-v6-upstream-1':raise ValueError('unknown upstream schema')
    return a,item
