"""Incremental R06 native export; old model bytes and evidence stay intact."""
import sys,json,gzip,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import OUT,sha,write,clean,now
from export_models import estimator,calibration
from research import raw_predict,apply_cal
from portable import raw_probability,probabilities

def main():
    base=OUT/'R06';dest=OUT/'portable_models';manifest=json.loads((dest/'manifest.json').read_text());old=[m for m in manifest['models'] if m['round']!='R06'];checks=[];models=[]
    results=json.loads((base/'results.json').read_text())
    for arm in results['arms']:
        p=sorted((base/'models').glob(f"{arm['arm']}_*.pkl"))[-1];m=pickle.loads(p.read_bytes());b=m['baseline']
        meta=json.loads((base/'cache/runs'/f"{m['arm']}_{m['month']}"/'meta.json').read_text());assert sha(p)==meta['model_sha256']
        a=dict(schema='stock-v6-native-1',round='R06',arm=m['arm'],month=m['month'],features=m['features'],registered=m['registered'],estimator=estimator(m['model']),candidate_calibrators={k:calibration(v) for k,v in m['candidate_calibrators'].items()},top_calibrators={k:calibration(v) for k,v in m['top_calibrators'].items()},variants=m['variants'],baseline=dict(vol_edges=b['vol_edges'],minute=b['minute'],stock_minute=[[s,int(t),v] for (s,t),v in b['stock_minute'].items()],matched=[[s,int(t),int(v),p] for (s,t,v),p in b['matched'].items()]),source_pickle_sha256=sha(p),independently_admitted=False)
        a['training_protocol']={key:meta.get(key) for key in ['training','train_days','candidate_calibration','top_calibration','selection','target','panel_sha256','protocol_sha256','top_calibration_dedup','baseline_vol_edges']}
        file=dest/f"R06_{m['arm']}_{m['month']}.json.gz";file.write_bytes(gzip.compress(json.dumps(clean(a),separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
        reloaded=json.loads(gzip.decompress(file.read_bytes()));sample=pd.read_parquet(base/'cache/panel.parquet',columns=['symbol','day','minute']+m['features'],filters=[('day','>=',m['month']+'-01'),('day','<',m['month']+'-32')]);sample=sample[sample.symbol.isin(m['registered'])].sample(min(len(sample),1000),random_state=1005).replace([np.inf,-np.inf],np.nan)
        expected=raw_predict(m['model'],m['features'],sample);errors=[float(np.max(abs(expected-raw_probability(reloaded,sample))))]
        for phase in ['pre','regular']:
            correct=apply_cal(m['top_calibrators'][phase],apply_cal(m['candidate_calibrators'][phase],expected));errors.append(float(np.max(abs(correct-probabilities(reloaded,sample,phase)))))
        error=max(errors);assert error<1e-10
        models.append(dict(round='R06',arm=m['arm'],month=m['month'],algorithm=arm['algorithm'],file=file.name,sha256=sha(file),bytes=file.stat().st_size,source_pickle_sha256=sha(p)))
        checks.append(dict(arm=m['arm'],rows=len(sample),max_error=error));print(json.dumps(checks[-1]),flush=True)
    for m in old:assert sha(dest/m['file'])==m['sha256']
    write(dest/'manifest.json',dict(manifest,at=now(),models=old+models))
    daily=pd.read_parquet(base/'cache/daily_features.parquet');day='2026-10-02';daily=daily[daily.day==day]
    for c in daily.select_dtypes('number'):daily[c]=daily[c].astype('float32')
    file=OUT/'source_data/R06/daily_features_latest.json.gz';file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(gzip.compress(json.dumps(clean(dict(records=daily.to_dict('records'))),separators=(',',':'),allow_nan=False).encode(),compresslevel=9,mtime=0))
    write(base/'feature_snapshot_manifest.json',dict(at=now(),day=day,file=str(file.relative_to(OUT)),sha256=sha(file),rows=len(daily),source_daily_sha256=sha(base/'cache/daily_features.parquet'),availability='preregistered source-day plus 2/5-session assumptions; no original release vintages; historical reference only'))
    write(base/'portable_verification.json',dict(at=now(),status='passed',checks=checks,old_model_sha_checks=len(old),note='Numerical equivalence only, no predictive admission.'))

if __name__=='__main__':main()
