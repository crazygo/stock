"""Math, causality and captured-data checks; no market requests."""
from datetime import date, timedelta
from collections import Counter
import json
import numpy as np
import pandas as pd
from build import OUT, ROOT, V1, USDAYS, HKDAYS, sha, at
from model import measure, boot, KEYS, WINDOWS

checks=[]
def check(name, ok, detail=None):
    checks.append({'name':name,'passed':bool(ok),'detail':detail})
    if not ok: raise AssertionError(name)

def independent(a):
    a=np.asarray(a,float);v=[None]*5
    if len(a)>=120:v[0]=float(sum(a[-120:])/120*252)
    if len(a)>=60:
        b=a[-60:];v[1]=float(np.std(b,ddof=1)*np.sqrt(252))
        if np.std(b[:-1])>1e-10 and np.std(b[1:])>1e-10:
            v[2]=float(np.corrcoef(b[:-1],b[1:])[0,1])
        e=(b-np.mean(b))**2
        if sum(e)>1e-20:v[3]=float(sum(sorted(e)[-6:])/sum(e))
    return v

def direct_four(a):
    """Separate S reference uses only four descriptors, avoiding recursion."""
    a=np.asarray(a,float);b=a[-60:];e=(b-b.mean())**2
    return [a[-120:].mean()*252,b.std(ddof=1)*np.sqrt(252),
            np.corrcoef(b[:-1],b[1:])[0,1],sum(sorted(e)[-6:])/sum(e)]

def reference(a):
    v=independent(a[-120:])[:4] if len(a)>=120 else independent(a)[:4]
    if len(a)<140:return v+[None]
    frames=[direct_four(a[:end]) for end in range(len(a)-20,len(a)+1)]
    f=np.array([[np.tanh(g/.5),np.tanh((h-.35)/.25),m,np.tanh((j-.55)/.20)] for g,h,m,j in frames])
    return v+[float(np.exp(-10*np.abs(np.diff(f,axis=0)).mean()/2))]

def main():
    rng=np.random.default_rng(173);a=rng.normal(.001,.024,140)
    check('five formulas match independent numerical reference',np.allclose(measure(a)[0],reference(a),rtol=1e-11,atol=1e-11))
    check('past-prefix changes outside estimator windows do not change current traits',np.allclose(measure(np.r_[rng.normal(0,.5,100),a])[0],measure(a)[0]))
    for n in [0,59,60,119,120,139,140]:
        vals=measure(a[-n:] if n else np.array([]))[0]
        check(f'estimator coverage boundary {n}',[np.isfinite(x) for x in vals]==[n>=w for w in WINDOWS])
    zero=measure(np.zeros(140))[0]
    check('zero variation remains undefined for M J S',zero[0]==0 and zero[1]==0 and np.isnan(zero[2:]).all())
    check('negative growth is preserved',measure(a-.03)[0,0]<0)
    alternating=np.tile([-.02,.02],70)
    check('alternating returns have reversal coefficient minus one',np.isclose(measure(alternating)[0,2],-1))
    b=boot(a,'fixture','2026-10-06')
    check('bootstrap reproducible',b==boot(a,'fixture','2026-10-06'))
    check('all bootstrap distributions sum to one',all(np.isclose(sum(x['p']),1) and x['replicates']==256 for x in b))
    check('60-return bootstrap retains missing G S',boot(a[-60:],'short','2026-10-06')[0] is None and boot(a[-60:],'short','2026-10-06')[4] is None)

    d=json.loads((OUT/'results.json').read_text());base=json.loads((V1/'results.json').read_text());cut=d['as_of']
    days=[x for x in USDAYS if x<=cut][-141:]
    fixture=pd.DataFrame({'day':days,'close':100*np.exp(np.r_[0,np.cumsum(a)])})
    x,s,_=at(fixture,'US.fixture',cut)
    check('complete official session windows are scored',s==['scored']*5)
    check('price scaling leaves labels invariant',np.allclose(x,at(fixture.assign(close=fixture.close*10),'US.fixture',cut)[0]))
    future=pd.DataFrame({'day':[x for x in USDAYS if x>cut][:2],'close':[1e6,1.]})
    check('future bars cannot alter an earlier cutoff',x==at(pd.concat([fixture,future]),'US.fixture',cut)[0])
    prior=USDAYS[USDAYS.index(days[0])-1]
    extended=pd.concat([pd.DataFrame({'day':[prior],'close':[99.]}),fixture],ignore_index=True)
    hole=extended.drop(extended.index[6]);_,status,seq=at(hole,'US.fixture',cut)
    check('coverage independently rejects S while retaining G V M J',status==['scored']*4+['missing_sessions'] and len(seq)==120)
    _,status,_=at(fixture.drop(fixture.index[-10]),'US.fixture',cut)
    check('recent missing session is not crossed by any label',status==['insufficient_history' if i==4 else 'missing_sessions' for i in range(5)])
    # at() first checks count: the 140 remaining closes cannot support S.

    check('captured members retained exactly once',len(d['records'])==166 and sorted(r['code'] for r in d['records'])==sorted(r['code'] for r in base['records']) and len({r['code'] for r in d['records']})==166)
    check('source v1 results unchanged',d['input_sha256']['v1_results']==sha(V1/'results.json'))
    check('registered source and protocol hashes match',all(d['input_sha256'][k]==sha(OUT/f) for k,f in [('protocol','PROTOCOL.md'),('model','model.py'),('build','build.py')]))
    start=(date.fromisoformat(cut)-timedelta(days=59)).isoformat();numeric_samples=0;hashes=0
    for r in d['records']:
        check(f"value and unknown state agree {r['code']}",all((v is not None)==(s=='scored') for v,s in zip(r['v'],r['status'])))
        check(f"map needs both coordinates {r['code']}",r['mapped']==(r['v'][0] is not None and r['v'][1] is not None))
        check(f"history has no future or synthetic calendar points {r['code']}",all(start<=h['day']<=cut and h['day'] in (USDAYS if r['code'].startswith('US.') else HKDAYS) for h in r['history']) and len({h['day'] for h in r['history']})==len(r['history']))
        for u in r['uncertainty']:
            if u:check(f"valid bootstrap probabilities {r['code']}",np.isclose(sum(u['p']),1) and all(0<=p<=1 for p in u['p']) and u['range'][0]<=u['range'][1])
        path=r['coverage'].get('path')
        if not path:continue
        provenance=next(p for p in d['provenance'] if p['code']==r['code'])
        check(f"raw snapshot unchanged {r['code']}",provenance['sha256']==sha(ROOT/path));hashes+=1
        f=pd.read_parquet(ROOT/path);f['day']=f.time_key.astype(str).str[:10];f=f[f.day<=cut].sort_values('day').drop_duplicates('day',keep='last')
        if r.get('listing_date','')>'1970-01-01':f=f[f.day>=r['listing_date']]
        px=f[['open','high','low','close']].to_numpy(float)
        f=f[np.isfinite(px).all(axis=1)&(px>0).all(axis=1)&(px[:,1]>=np.maximum(px[:,0],px[:,3]))&(px[:,2]<=np.minimum(px[:,0],px[:,3]))]
        samples=[{'day':cut,'v':r['v']}] + [r['history'][i] for i in sorted({0,len(r['history'])//2,len(r['history'])-1})] if r['history'] else [{'day':cut,'v':r['v']}]
        for h in samples:
            closes=f[f.day<=h['day']].close.to_numpy(float)[-141:]
            if len(closes)<2:continue
            expected=reference(np.diff(np.log(closes)))
            check(f"independent point recomputation {r['code']} {h['day']}",all(v is None or np.isclose(v,expected[k],rtol=1e-10,atol=1e-10) for k,v in enumerate(h['v'])));numeric_samples+=1
    check('all raw input hashes checked',hashes==163)
    check('map coverage recomputes',sum(r['mapped'] for r in d['records'])==d['summary']['mapped']==150)
    for k,key in enumerate(KEYS):check('coverage recomputes '+key,dict(Counter(r['status'][k] for r in d['records']))==d['summary']['trait_coverage'][key])
    report={'run_id':d['run_id'],'checks':checks,'errors':[],'numeric_point_samples':numeric_samples,'raw_file_hashes':hashes,'all_passed':all(c['passed'] for c in checks)}
    (OUT/'model_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['run_id','all_passed','numeric_point_samples','raw_file_hashes']}));print(f'{len(checks)} checks passed')

if __name__=='__main__':main()
