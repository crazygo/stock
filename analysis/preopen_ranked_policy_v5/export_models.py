"""Export trusted local frozen October models as portable data and check parity."""
import gzip,json,pickle
import numpy as np
import pandas as pd
from common import *
from portable import NativeModel,digest
from phase import predict

def preprocess(imputer):return dict(statistics=imputer.statistics_.tolist(),indicators=imputer.indicator_.features_.tolist())

def main():
 dep=json.loads((OUT/'signals_deployment.json').read_text());dest=OUT/'portable_models';dest.mkdir(exist_ok=True);entries=[]
 panel=pd.read_parquet(OUT/'panel.parquet');panel['symbol']=panel.symbol.astype(str);sample=panel.groupby('symbol',observed=True).sample(n=120,random_state=1004)
 for meta in dep['all_routes']:
  path=OUT/'models'/meta['model'];assert digest(path)==meta['model_sha256'];a=pickle.loads(path.read_bytes());m=a['model'];route=a['route']
  native=dict(format='preopen_native_v1',route=route,features=a['features'],registered=a['registered'],stock_map=a['stock_map'],calibrators={p:dict(coef=float(c.coef_[0,0]),intercept=float(c.intercept_[0])) for p,c in a['phase_calibrators'].items()},source_model_sha256=meta['model_sha256'],train_end=meta['train_end'],calibration_end=meta['calibration_end'],selection_end=meta['selection_end'],thresholds=meta['thresholds'],baseline_table=a['baseline_table'],baseline_minute=a['baseline_minute'])
  if route=='relative_flow':native['learner']=dict(model_text=m.booster_.model_to_string(),library='lightgbm',native_version=m.booster_.dump_model()['version'])
  elif route=='hazard_linear':
   number=m.named_steps['features'].named_transformers_['number'];p=preprocess(number.named_steps['impute']);p.update(center=number.named_steps['scale'].center_.tolist(),scale=number.named_steps['scale'].scale_.tolist(),categories=m.named_steps['features'].named_transformers_['stock'].categories_[0].tolist());native['preprocess']=p;learner=m.named_steps['learner'];native['learner']=dict(coef=learner.coef_[0].tolist(),intercept=float(learner.intercept_[0]))
  else:
   native['preprocess']=preprocess(m.named_steps['impute']);trees=[]
   for tree in m.named_steps['learner'].estimators_:
    t=tree.tree_;v=t.value[:,0,:];trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),p=(v[:,1]/v.sum(axis=1)).tolist()))
   native['learner']=dict(max_depth=m.named_steps['learner'].max_depth,trees=trees)
  output=dest/f'{route}_2026-10.json.gz';output.write_bytes(gzip.compress(json.dumps(clean(native),separators=(',',':'),allow_nan=False).encode(),mtime=0));sha256=digest(output);f=sample[sample.symbol.isin(a['registered'])].copy()
  for c in a['features']:
   if c!='symbol':f[c]=f[c].astype('float32')
  expected=predict(a,f);actual=NativeModel(output,sha256).predict(f);error=float(np.max(np.abs(expected-actual)));assert error<1e-10,(route,error)
  entries.append(dict(route=route,name=NAMES[route],file=output.name,sha256=sha256,bytes=output.stat().st_size,source_model_sha256=meta['model_sha256'],thresholds=meta['thresholds'],registered=a['registered'],parity_rows=len(f),max_absolute_probability_error=error,train_end=meta['train_end'],selection_end=meta['selection_end']))
  print(json.dumps(entries[-1]),flush=True)
 write(dest/'manifest.json',dict(format='preopen_native_v1',frozen_at=dep['frozen_at'],model_month='2026-10',observe_from=dep['observe_from'],observe_through=dep['observe_through'],admitted=dep['admitted'],selected_route=dep['route'],all_routes=entries,signal_deployment_sha256=digest(OUT/'signals_deployment.json'),note='Top-three reference reporting does not alter frozen signal admission. Native assets contain model parameters, not raw price histories.'))
if __name__=='__main__':main()
