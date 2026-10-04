"""Data-only native inference; no sklearn/pickle runtime needed in the cloud."""
from __future__ import annotations
import gzip,json,hashlib
from pathlib import Path
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-700,700)))

class NativeModel:
 def __init__(self,path,expected_sha256):
  if digest(path)!=expected_sha256:raise ValueError('Frozen native model SHA mismatch: '+Path(path).name)
  self.a=json.loads(gzip.decompress(Path(path).read_bytes()));a=self.a
  if a['format']!='preopen_native_v1':raise ValueError('Unknown native model format')
  self.booster=None
  if a['route']=='relative_flow':
   import lightgbm
   self.booster=lightgbm.Booster(model_str=a['learner']['model_text'])
 def predict(self,f):
  a=self.a;cols=[c for c in a['features'] if c!='symbol'];x=f[cols].to_numpy(dtype=float);x[~np.isfinite(x)]=np.nan
  if self.booster is not None:raw=self.booster.predict(x,num_threads=1)
  else:
   p=a['preprocess'];x=x.astype('float32');missing=np.isnan(x);x=np.where(missing,np.asarray(p['statistics'],dtype='float32'),x)
   if p['indicators']:x=np.concatenate([x,missing[:,p['indicators']].astype('float32')],axis=1)
   if a['route']=='hazard_linear':
    x-=np.asarray(p['center']);x/=np.asarray(p['scale']);cats=p['categories'];sym=f.symbol.to_numpy();onehot=np.column_stack([sym==c for c in cats]).astype(float);x=np.concatenate([x,onehot],axis=1);raw=sigmoid(x@np.asarray(a['learner']['coef'])+a['learner']['intercept'])
   else:
    # sklearn trees compare float32 inputs against stored float64 thresholds.
    x=x.astype('float32');raw=np.zeros(len(f));indices=np.arange(len(f))
    for t in a['learner']['trees']:
     left=np.asarray(t['left']);right=np.asarray(t['right']);feature=np.asarray(t['feature']);threshold=np.asarray(t['threshold']);node=np.zeros(len(f),dtype=int)
     for _ in range(a['learner']['max_depth']+1):
      active=left[node]!=-1
      if not active.any():break
      i=indices[active];n=node[active];node[active]=np.where(x[i,feature[n]]<=threshold[n],left[n],right[n])
     raw+=np.asarray(t['p'])[node]
    raw/=len(a['learner']['trees'])
  z=np.clip(raw,1e-5,1-1e-5);z=np.log(z/(1-z));out=np.zeros(len(f))
  for phase,cal in a['calibrators'].items():
   mask=(f.minute.to_numpy()<=570) if phase=='pre' else (f.minute.to_numpy()>570);out[mask]=sigmoid(cal['coef']*z[mask]+cal['intercept'])
  return out
