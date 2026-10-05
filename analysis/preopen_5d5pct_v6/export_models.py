"""Export only latest monthly artifacts and prove numeric equivalence."""
import gzip,json,pickle
import numpy as np
import pandas as pd
from common import OUT,sha,write,clean,now
from research import raw_predict,apply_cal
from portable import raw_probability,probabilities

def calibration(model):
 return None if model is None else dict(coefficient=float(model.coef_[0,0]),intercept=float(model.intercept_[0]))

def estimator(model):
 if hasattr(model,'booster_'):return dict(kind='LightGBM',model_text=model.booster_.model_to_string())
 impute=model.named_steps['impute'];final=model.named_steps['model'];result=dict(kind=type(final).__name__.removesuffix('Classifier'),imputation=dict(statistics=impute.statistics_.tolist(),indicators=impute.indicator_.features_.tolist()))
 if result['kind']=='LogisticRegression':
  scale=model.named_steps['scale'];result.update(center=scale.center_.tolist(),scale=scale.scale_.tolist(),coefficient=final.coef_[0].tolist(),intercept=float(final.intercept_[0]))
 else:
  assert result['kind']=='ExtraTrees'
  result['trees']=[]
  for tree in final.estimators_:
   t=tree.tree_;value=t.value[:,0,:];positive=value[:,1]/value.sum(1)
   result['trees'].append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),positive_probability=positive.tolist()))
 return result

def main():
 dest=OUT/'portable_models';dest.mkdir(exist_ok=True);models=[];checks=[]
 for rid,base in [('R00',OUT),('R01',OUT/'R01'),('R02',OUT/'R02'),('R03',OUT/'R03'),('R04',OUT/'R04'),('R05',OUT/'R05')]:
  results=base/'results.json'
  if not results.exists():continue
  for arm in json.loads(results.read_text())['arms']:
   paths=sorted((base/'models').glob(f"{arm['arm']}_*.pkl"))
   if not paths:continue
   p=paths[-1];m=pickle.loads(p.read_bytes());b=m['baseline']
   a=dict(schema='stock-v6-native-1',round=rid,arm=m['arm'],month=m['month'],features=m['features'],registered=m['registered'],estimator=estimator(m['model']),candidate_calibrators={k:calibration(v) for k,v in m['candidate_calibrators'].items()},top_calibrators={k:calibration(v) for k,v in m['top_calibrators'].items()},variants=m['variants'],baseline=dict(vol_edges=b['vol_edges'],minute=b['minute'],stock_minute=[[s,int(t),v] for (s,t),v in b['stock_minute'].items()],matched=[[s,int(t),int(v),p] for (s,t,v),p in b['matched'].items()]),source_pickle_sha256=sha(p),independently_admitted=False)
   metadata=json.loads((base/'cache/runs'/f"{m['arm']}_{m['month']}"/'meta.json').read_text())
   a['training_protocol']={key:metadata.get(key) for key in ['training','train_days','candidate_calibration','top_calibration','selection','target','panel_sha256','protocol_sha256']}
   a['training_protocol']['top_calibration_dedup']=metadata.get('top_calibration_dedup','phase_local_stock_day')
   a['training_protocol']['baseline_vol_edges']=metadata.get('baseline_vol_edges','training_candidate_and_top_blocks')
   file=dest/f"{rid}_{m['arm']}_{m['month']}.json.gz";file.write_bytes(gzip.compress(json.dumps(clean(a),separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
   reloaded=json.loads(gzip.decompress(file.read_bytes()));cols=['symbol','day','minute']+m['features']
   panel_file='H0_panel.parquet' if rid=='R03' and m['arm']=='H0' else 'panel.parquet'
   sample=pd.read_parquet(base/'cache'/panel_file,columns=cols,filters=[('day','>=',m['month']+'-01'),('day','<',m['month']+'-32')])
   sample=sample[sample.symbol.isin(m['registered'])].sample(min(len(sample),1000),random_state=1005).replace([np.inf,-np.inf],np.nan)
   old=raw_predict(m['model'],m['features'],sample);native=raw_probability(reloaded,sample);errors=[float(np.max(np.abs(old-native)))]
   for phase in ['pre','regular']:
    expected=apply_cal(m['top_calibrators'][phase],apply_cal(m['candidate_calibrators'][phase],old));actual=probabilities(reloaded,sample,phase);errors.append(float(np.max(np.abs(expected-actual))))
   err=max(errors);assert err<1e-10,(rid,m['arm'],err)
   models.append(dict(round=rid,arm=m['arm'],month=m['month'],algorithm=arm['algorithm'],file=file.name,sha256=sha(file),bytes=file.stat().st_size,source_pickle_sha256=sha(p)))
   checks.append(dict(round=rid,arm=m['arm'],rows=len(sample),max_error=err));print(json.dumps(checks[-1]),flush=True)
 upstream_item=None
 provenance=OUT/'R01/upstream_provenance.json'
 if provenance.exists():
  last=next(b for b in reversed(json.loads(provenance.read_text())['blocks']) if b['status']=='fitted')
  p=OUT/'R01/models'/f"daily_opportunity_block_{last['block']:03d}.pkl";model=pickle.loads(p.read_bytes())
  a=dict(schema='stock-v6-upstream-1',features=last['features'],registered=last['registered'],estimator=estimator(model),training=last['training'],max_training_label_end=last['max_training_label_end'],score_is_calibrated=False,source_pickle_sha256=sha(p))
  file=dest/'R01_daily_opportunity.json.gz';file.write_bytes(gzip.compress(json.dumps(clean(a),separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
  sample=pd.read_parquet(OUT/'R01/cache/panel.parquet',columns=['symbol','day']+last['features'],filters=[('day','>=',last['start']),('day','<=',last['end'])]).drop_duplicates(['symbol','day']).head(1000)
  err=float(np.max(np.abs(model.predict_proba(sample[last['features']])[:,1]-raw_probability(a,sample))));assert err<1e-10
  checks.append(dict(round='R01',arm='daily_opportunity',rows=len(sample),max_error=err));upstream_item=dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,source_pickle_sha256=sha(p))
 write(dest/'manifest.json',dict(schema='stock-v6-native-manifest-1',at=now(),models=models,upstream=upstream_item,admission='research_only_effective_threshold_none'))
 anchors=OUT/'R03/cache/anchors.parquet'
 if anchors.exists():
  file=OUT/'source_data/R03/mature_anchors.json.gz';file.parent.mkdir(parents=True,exist_ok=True)
  records=pd.read_parquet(anchors).to_dict('records')
  file.write_bytes(gzip.compress(json.dumps(clean(dict(source_sha256=sha(anchors),records=records)),separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
  write(OUT/'R03/mature_anchor_manifest.json',dict(file=str(file.relative_to(OUT)),sha256=sha(file),rows=len(records),availability='complete prior label_end only; current membership retrospective'))
 write(OUT/'portable_verification.json',dict(at=now(),status='passed',checks=checks,note='Numeric portability is engineering evidence, not predictive or calibration validation.'))
if __name__=='__main__':main()
