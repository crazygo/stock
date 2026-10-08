"""Independent numpy/repository formula checks; not a semantic or predictive test."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
source=OUT.parent/'operation_cadence_v1'
P=json.loads((source/'prices.json').read_text())
cases=json.loads((OUT/'verification_cases.json').read_text())
spec=importlib.util.spec_from_file_location('original_traits',OUT.parent/'ai_trend_quadrant_v2/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
count=0;windows=0;five=0
for x in cases:
    rows=[b for b in P['records'][x['code']] if x['start']<=b[0]<=x['end']]
    if x['metrics'] is None:continue
    p=np.array([b[4] for b in rows],float);r=np.log(p[1:]/p[:-1]);m=x['metrics']
    values={'net':p[-1]/p[0]-1,'signedEfficiency':r.sum()/np.abs(r).sum() if np.abs(r).sum()>1e-14 else 0,
            'vol':r.std(ddof=1),'drawdown':float(np.max(1-p/np.maximum.accumulate(p)))}
    pos=np.sort(r[r>0])[::-1]
    values['top3Share']=pos[:3].sum()/pos.sum() if len(pos) else None
    for key,v in values.items():
        if v is None:assert m[key] is None
        else:assert abs(m[key]-v)<1e-10,(x['code'],key,m[key],v)
    measured=old.measure(r)[0]
    for i,v in enumerate(measured):
        if np.isfinite(v):assert abs(x['traits'][i]-v)<1e-10,(x['code'],i,x['traits'][i],v);five+=1
        else:assert x['traits'][i] is None
    for w in x['windows']:
        a=np.array([b[4] for b in rows if w['first']<=b[0]<=w['last']],float)
        assert len(a)==21
        rr=np.log(a[1:]/a[:-1]);net=a[-1]/a[0]-1;eff=rr.sum()/np.abs(rr).sum() if np.abs(rr).sum()>1e-14 else 0
        assert abs(w['net']-net)<1e-10 and abs(w['efficiency']-eff)<1e-10
        assert abs(w['vol']-rr.std(ddof=1))<1e-10
        assert w['up']==bool(net>0 and eff>=.2)
        windows+=1
    if x['shares']:
        for key in ['up','repeated']:
            assert abs(x['shares'][key]-sum(w[key] for w in x['windows'])/len(x['windows']))<1e-12
    count+=1
manifest=json.loads((OUT/'input_manifest.json').read_text())
for f,v in manifest.items():assert hashlib.sha256((ROOT/v['path']).read_bytes()).hexdigest()==v['sha256']
acq=json.loads((source/'acquisition.json').read_text())
intraday=json.loads((source/'intraday.json').read_text())
raw=0
for v in list(acq['datasets'].values())+list(intraday['sources'].values()):
    if not v.get('path'):continue
    assert hashlib.sha256((ROOT/v['path']).read_bytes()).hexdigest()==v['sha256'],v['path']
    raw+=1
report={'status':'passed','independent_engine':'Python numpy + original five_traits_v2',
        'stock_range_checks':count,'historical_window_checks':windows,'original_trait_checks':five,
        'immutable_raw_file_sha_checks':raw,'input_manifest_checks':len(manifest),
        'semantic_alignment':'pending human review','prediction_validation':'out of scope'}
(OUT/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
