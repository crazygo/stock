"""Replay ALL registered and selected artifacts; source/temporal checks."""
import json,pickle
import numpy as np
import pandas as pd
import evaluate as ev
OUT=ev.OUT

def main():
 ev.init();audit=json.loads((OUT/'audit.json').read_text());checks=[];source_files={s['path']:s['sha256'] for s in audit['sources']};registered=[]
 for resultfile,predfile,enrolledfile in [('results.json','predictions.parquet','enrolled_predictions.parquet'),('alignment_results.json','alignment_predictions.parquet','alignment_enrolled_predictions.parquet')]:
  r=json.loads((OUT/resultfile).read_text());pred=pd.read_parquet(OUT/predfile);ep=pd.read_parquet(OUT/enrolledfile)
  assert r['protocol_sha256']==ev.b.sha(OUT/'PROTOCOL.md') and r['panel_sha256']==ev.b.sha(OUT/'panel.parquet') and r['predictions_sha256']==ev.b.sha(OUT/predfile) and r['enrolled_predictions_sha256']==ev.b.sha(OUT/enrolledfile)
  assert not pred.duplicated(['symbol','day','cycle_start']).any() and not ep.duplicated(['symbol','day','cycle_start','model']).any()
  assert (pd.to_datetime(ep.max_source_available)<=pd.to_datetime(ep.decision_at)).all()
  for c in r['enrolled_combinations']:
   registered.append(c);g=ep[ep.symbol.eq(c['symbol'])&ep.cycle_start.eq(c['start'])&ep.model.eq(c['selected'])]
   assert c['train_end']<c['fit_end']<c['selection_end']<c['start'];assert c['calibration']['n']>=8 and c['calibration']['precision']>=.7 and c['calibration']['lift']>=.03
   path=OUT/'models'/c['artifact'];assert ev.b.sha(path)==c['artifact_sha256']
   with path.open('rb') as f:a=pickle.load(f)
   assert not set(a['features'])&ev.LABELS;score=ev.transform(a['model'].predict_proba(g[a['features']])[:,1],a['calibrator']);np.testing.assert_allclose(score,g.score,rtol=0,atol=1e-12)
   m=ev.stats(g,score,c['threshold']);assert m['tp']==c['metrics']['tp'] and m['n']==c['metrics']['n'];passed=bool(c['complete'] and len(g)>=14 and m['n']>=12 and m['precision']>=.7 and m['lift']>=.03 and m['cost_proxy']>0);assert passed==c['pass_research']
   primary=next(x for x in r['cells'] if x['symbol']==c['symbol'] and x['start']==c['start'])
   if primary['selected']==c['selected']:
    p=pred[pred.symbol.eq(c['symbol'])&pred.cycle_start.eq(c['start'])];assert len(p)==len(g);np.testing.assert_allclose(p.score,g.score,atol=1e-12,rtol=0)
   checks.append(dict(symbol=c['symbol'],start=c['start'],model=c['selected'],rows=len(g),score_replay_max_error=float(np.max(np.abs(score-g.score))),pass_research=passed))
 for p,digest in source_files.items():assert ev.b.sha(ev.b.ROOT/p)==digest,p
 for m in audit['macro_sources']:assert ev.b.sha(OUT/'raw'/(m['symbol']+'_History.csv'))==m['sha256']
 passed=[c for c in registered if c['pass_research']];stocks=sorted(set(c['symbol'] for c in passed));assert len(passed)>=5 and len(stocks)>=5
 for s in stocks:
  loaded,*_=ev.b.legacy.load(s);saved=pd.read_parquet(OUT/'raw'/f'{s}.parquet');pd.testing.assert_frame_equal(loaded.reset_index(drop=True),saved.drop(columns=['night_day'],errors='ignore').reset_index(drop=True))
 payload=dict(status='passed',registered_artifacts=len(checks),source_files=len(source_files),macro_sources=3,one_sample_per_stock_day_per_model_cycle=True,overlap_across_cycles_disclosed=True,causal_availability=True,all_models=checks,qualified_combinations=len(passed),qualified_stocks=stocks,files={p.name:ev.b.sha(p) for p in OUT.glob('*.py')},protocol_sha256=ev.b.sha(OUT/'PROTOCOL.md'),panel_sha256=ev.b.sha(OUT/'panel.parquet'),predictions_sha256=ev.b.sha(OUT/'predictions.parquet'))
 (OUT/'run_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in payload.items() if k not in ['all_models','files']}))
if __name__=='__main__':main()
