"""Data-only October horizon artifacts, SHA and independent native parity."""
import gzip,json,pickle
import numpy as np
import pandas as pd
from common import *
from horizon import DEST,TARGETS
from export_models import preprocess
from portable import NativeModel,digest
from phase import predict
from store import dump

def main():
 import pyarrow.parquet as pq
 rng=np.random.default_rng(1004);parts=[]
 for batch in pq.ParquetFile(OUT/'weekly_v1/panel.parquet').iter_batches(batch_size=32768):
  frame=batch.to_pandas();parts.append(frame.iloc[rng.choice(len(frame),min(128,len(frame)),replace=False)])
 sample=pd.concat(parts,ignore_index=True);sample['symbol']=sample.symbol.astype(str);entries=[]
 for target in TARGETS:
  dest=DEST/target/'portable_models';dest.mkdir(exist_ok=True);metas=[]
  for route in ROUTES:
   meta=json.loads((DEST/target/f'{route}_2026-10_meta.json').read_text());path=DEST/target/'models'/f'{route}_2026-10.pkl';assert digest(path)==meta['model_sha256'];a=pickle.loads(path.read_bytes());model=a['model']
   native=dict(format='preopen_native_v1',route=route,target_id=target,features=a['features'],registered=a['registered'],stock_map=a['stock_map'],calibrators={p:dict(coef=float(c.coef_[0,0]),intercept=float(c.intercept_[0])) for p,c in a['phase_calibrators'].items()},source_model_sha256=meta['model_sha256'],train_end=meta['train_end'],calibration_end=meta['calibration_end'],selection_end=meta['selection_end'],thresholds=meta['thresholds'],baseline_table=a['baseline_table'],baseline_minute=a['baseline_minute'])
   if route=='relative_flow':native['learner']=dict(model_text=model.booster_.model_to_string(),library='lightgbm',native_version=model.booster_.dump_model()['version'])
   elif route=='hazard_linear':
    number=model.named_steps['features'].named_transformers_['number'];p=preprocess(number.named_steps['impute']);p.update(center=number.named_steps['scale'].center_.tolist(),scale=number.named_steps['scale'].scale_.tolist(),categories=model.named_steps['features'].named_transformers_['stock'].categories_[0].tolist());native['preprocess']=p;learner=model.named_steps['learner'];native['learner']=dict(coef=learner.coef_[0].tolist(),intercept=float(learner.intercept_[0]))
   else:
    native['preprocess']=preprocess(model.named_steps['impute']);trees=[]
    for tree in model.named_steps['learner'].estimators_:
     t=tree.tree_;v=t.value[:,0,:];trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),p=(v[:,1]/v.sum(axis=1)).tolist()))
    native['learner']=dict(max_depth=model.named_steps['learner'].max_depth,trees=trees)
   output=dest/f'{route}_2026-10.json.gz';output.write_bytes(gzip.compress(json.dumps(clean(native),separators=(',',':'),allow_nan=False).encode(),mtime=0));f=sample[sample.symbol.isin(a['registered'])].copy();f['stock_id']=f.symbol.map(a['stock_map'])
   for c in a['features']:
    if c!='symbol':f[c]=f[c].astype('float32')
   error=float(np.max(np.abs(predict(a,f)-NativeModel(output,digest(output)).predict(f))));assert error<1e-10
   entry=dict(route=route,name=NAMES[route],algorithm=ALGORITHMS[route],file=output.name,sha256=digest(output),bytes=output.stat().st_size,source_model_sha256=meta['model_sha256'],registered=a['registered'],thresholds=meta['thresholds'],parity_rows=len(f),parity_premarket_rows=int((f.minute<=570).sum()),max_absolute_probability_error=error,train_end=meta['train_end'],selection_end=meta['selection_end']);metas.append(entry);entries.append(dict(target_id=target,**entry));print(dump(dict(target=target,route=route,parity_rows=len(f),error=error)),flush=True)
  write(dest/'manifest.json',dict(format='preopen_native_v1',target_id=target,model_month='2026-10',frozen_at=datetime.now(ET).isoformat(),admitted=False,selected_route=None,all_routes=metas,protocol_sha256=digest(OUT/'HORIZON_PROTOCOL.md'),cloud_protocol_sha256=digest(OUT/'HORIZON_CLOUD_PROTOCOL.md'),note='Research probability report only; no change to original October signals.'))
  reference=json.loads((OUT/'portable_models/reference_snapshot.json').read_text());reference['model_manifest_sha256']=digest(dest/'manifest.json');reference['target_id']=target;write(dest/'reference_snapshot.json',reference)
 write(DEST/'native_verification.json',dict(status='passed',all_models=entries,not_effectiveness_evidence=True))
if __name__=='__main__':main()
