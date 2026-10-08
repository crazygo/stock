"""Independent formula and raw-extrema audit; no semantic acceptance inferred."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
P=json.loads((OUT.parent/'operation_cadence_v1/prices.json').read_text())
counts={'paths':0,'complete_waves':0,'up_windows':0}
for x in json.loads((OUT/'verification_cases.json').read_text()):
    rows=[b for b in P['records'][x['code']] if x['start']<=b[0]<=x['end']]
    lookup={b[0]:b[4] for b in rows}
    for w in x['windows']:
        p=np.array([lookup[d[0]] for d in rows if w['first']<=d[0]<=w['last']]);r=np.log(p[1:]/p[:-1])
        e=float(r.sum()/np.abs(r).sum()) if np.abs(r).sum()>1e-14 else 0
        assert w['up']==bool(p[-1]/p[0]>1 and e>=.2)
        counts['up_windows']+=1
    f=x['folds']
    if f is None:continue
    assert f['returns']==len(rows)-1
    assert abs(f['frequency']-len(x['metrics']['moves'][1:])*20/(len(rows)-1))<1e-10
    values=[]
    for w in f['completed']:
        a,b=w['startIndex'],w['endIndex']; segment=[v[4] for v in rows[a:b+1]]
        assert rows[a][0]==w['start'] and rows[b][0]==w['end']
        assert abs(max(segment)/min(segment)-1-w['amplitude'])<1e-10
        if w['direction']>0:assert segment[0]==min(segment) and segment[-1]==max(segment)
        else:assert segment[0]==max(segment) and segment[-1]==min(segment)
        confirm=lookup[w['confirmed']]
        assert confirm/segment[-1]-1<=-.05+1e-10 if w['direction']>0 else confirm/segment[-1]-1>=.05-1e-10
        values.append(max(segment)/min(segment)-1);counts['complete_waves']+=1
    if values:assert abs(float(np.median(values))-f['amplitudeMedian'])<1e-10
    else:assert f['amplitudeMedian'] is None
    assert len({(w['start'],w['end']) for w in f['completed']})==len(f['completed'])
    counts['paths']+=1
manifest=json.loads((OUT/'input_manifest.json').read_text())
for v in manifest.values():assert hashlib.sha256((ROOT/v['path']).read_bytes()).hexdigest()==v['sha256']
report={'status':'passed','engine':'Python numpy + original OHLC extrema','checks':counts,'input_sha_checks':len(manifest),'semantic_alignment':'pending human review'}
(OUT/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report))
