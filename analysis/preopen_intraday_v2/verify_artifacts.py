"""Audit artifact provenance and replay frozen models against stored predictions."""
from pathlib import Path
import json,hashlib,pickle,sys,importlib.metadata
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import evaluate as ev
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=json.loads((OUT/'audit.json').read_text());u=json.loads((OUT/'universe.json').read_text())
    assert sha(OUT/'universe.json')==audit['universe_sha256'],'Universe changed after build; rebuild required'
    assert sha(OUT/'panel.parquet')==audit['panel_sha256']
    results=[]
    for rfile,pfile,modeldir in [('results.json','predictions.parquet','models'),('adaptive_results.json','adaptive_predictions.parquet','models_adaptive')]:
        r=json.loads((OUT/rfile).read_text());pred=pd.read_parquet(OUT/pfile);test=pred[pred.period=='test'].copy()
        assert r['panel_sha256']==sha(OUT/'panel.parquet')
        assert r['predictions_sha256']==sha(OUT/pfile)
        assert (test.y==(test.mfe>=.03).astype(int)).all()
        assert (pd.to_datetime(pred.max_source_available)<=pd.to_datetime(pred.decision_at)).all()
        assert (pd.to_datetime(pred.feature_end)<pd.to_datetime(pred.day+' 09:30')).all()
        if modeldir=='models_adaptive':
            assert (pred.train_end<pred.day).all() and (pred.calibration_end<pred.day).all()
        for route in ['H','continuation','repair','relative']:
            assert np.isfinite(pred['score_'+route]).all()
            folds=pred.groupby('fold') if modeldir=='models_adaptive' else [('fixed',pred)]
            replayed=0
            for fold,g in folds:
                sample=g.iloc[np.linspace(0,len(g)-1,min(3,len(g))).astype(int)]
                name=f'{fold}_{route}.pkl' if modeldir=='models_adaptive' else route+'.pkl'
                with (OUT/modeldir/name).open('rb') as f:m=pickle.load(f)
                p,raw=ev.predict(m['model'],m['calibrator'],sample[m['features']])
                np.testing.assert_allclose(p,sample['score_'+route],rtol=0,atol=1e-12)
                for q,reg in zip([20,50,80],m['amplitude_models']):
                    np.testing.assert_allclose(np.maximum(0,reg.predict(sample[m['features']])),sample[f'mfe_{route}_{q}'],rtol=0,atol=1e-12)
                replayed+=len(sample)
            actual=ev.stats(test,test['score_'+route]>=r['routes'][route]['threshold'])
            for metric in ['n','tp','fp','fn','tn','days','months']:
                assert actual[metric]==r['routes'][route]['summary'][metric]
            np.testing.assert_allclose(actual['precision'],r['routes'][route]['summary']['precision'],rtol=0,atol=1e-12)
            results.append({'version':modeldir,'route':route,'replayed_rows':replayed,'n_signals':actual['n'],'precision':actual['precision']})
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['pandas','numpy','pyarrow','scikit-learn','scipy','lightgbm','futu-api']},'artifact_checks':results,'sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='run_manifest.json' and p.suffix in ('.py','.md','.json','.html','.parquet')},'source_files':len(audit['sources']),'market_sources_match':all(sha(OUT.parents[1]/s['path'])==s['sha256'] for s in audit['sources'])}
    assert manifest['market_sources_match']
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({'status':'passed','model_checks':len(results),'sources':len(audit['sources'])}))
if __name__=='__main__':main()
