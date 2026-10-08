"""Audit the captured run against membership, source hashes and the frozen formula."""
from pathlib import Path
import hashlib,json,unittest
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import pandas as pd
from model import measurements,normalize
from build import HKDAYS

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    d=json.loads((OUT/'results.json').read_text());watch=json.loads((OUT/'watchlist.json').read_text())
    checks=[]
    def check(name,predicate):
        checks.append({'name':name,'passed':bool(predicate)})
        if not predicate:raise AssertionError(name)
    expected={r['code'] for r in watch['groups']['全部'] if r['kind'] in ['STOCK','ETF']}
    check('Every current stock and ETF is preserved exactly once',len(d['records'])==len(expected) and {r['code'] for r in d['records']}==expected)
    check('All non-securities are accounted for',len(d['records'])+len(d['excluded'])==len(watch['groups']['全部']))
    inputs={'watchlist':OUT/'watchlist.json','protocol':OUT/'PROTOCOL.md','quality':ROOT/'analysis/ai_value_chain_map_v1/quality/current.json','hk_calendar':OUT/'hk_calendar_2026.json'}
    for k,p in inputs.items():check('Input hash: '+k,sha(p)==d['source_sha256'][k])
    for k in ['model','build']:check('Code hash: '+k,sha(OUT/(k+'.py'))==d['source_code_sha256'][k])
    check('Immutable snapshot equals current derived results',sha(OUT/'results.json')==sha(OUT/'runs'/d['run_id']/'snapshot.json'))
    for p in d['provenance']:check('Price-source hash: '+p['code'],sha(ROOT/p['path'])==p['sha256'])
    usdays=[r['session_date'] for r in json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']]
    current_rows=0;history_rows=0;max_prob_error=0;scored_codes=set();low_liquidity=[]
    for r in d['records']:
        f=None
        if r['coverage'].get('path'):
            f=pd.read_parquet(ROOT/r['coverage']['path']);f['day']=f.time_key.astype(str).str[:10]
            f=f[f.day<=d['as_of']].sort_values('day').drop_duplicates('day',keep='last')
            listing=r.get('listing_date','')
            if listing>'1970-01-01':f=f[f.day>=listing]
        for key,t in r['windows'].items():
            current_rows+=1
            if t['status']!='scored':
                check('Unscored row has no artificial point '+r['code']+'/'+key,not any(k in t for k in ['p','x','y']))
                continue
            scored_codes.add(r['code'])
            n=int(key);g=f[f.day<=t['as_of']].tail(n+1)
            cal=usdays if r['code'].startswith('US.') else HKDAYS
            check('Complete official session window '+r['code']+'/'+key,list(g.day)==[x for x in cal if g.day.iloc[0]<=x<=t['as_of']] and len(g)==n+1 and t['actual_last_day']==d['as_of'])
            c=np.log(g.close.to_numpy(float));o=np.log(g.open.to_numpy(float));a=np.column_stack([np.diff(c),o[1:]-c[:-1],c[1:]-o[1:],np.log(g.high/g.low).to_numpy()[1:]])
            check('Recomputed eight metrics '+r['code']+'/'+key,np.allclose(measurements(a)[0],t['measurements'],rtol=1e-11,atol=1e-11))
            z=normalize(t['measurements'])
            check('Pugh axes match contribution '+r['code']+'/'+key,np.allclose(z,t['z']) and abs(z[:4]@np.array(d['weight_x'])-t['x'])<1e-12 and abs(z[4:]@np.array(d['weight_y'])-t['y'])<1e-12)
            check('Raw probabilities match all 512 quadrant counts '+r['code']+'/'+key,np.allclose((np.array(t['bootstrap_counts'])+.5)/514,t['raw_p']) and sum(t['bootstrap_counts'])==512)
            if t.get('low_liquidity_observed') and key=='60':low_liquidity.append(r['code'])
        for t in list(r['windows'].values())+r['history']:
            if t['status']!='scored':continue
            max_prob_error=max(max_prob_error,abs(sum(t['p'])-1))
            check('Probability and entropy '+r['code']+'/'+str(t['window'])+'/'+t['as_of'],all(0<p<1 for p in t['p']) and abs(sum(t['p'])-1)<1e-12 and abs(-sum(p*np.log(p) for p in t['p'])/np.log(4)-t['entropy'])<1e-12)
            check('No future cutoff '+r['code']+'/'+t['as_of'],t['as_of']<=d['as_of'] and t['actual_last_day']<=t['as_of'])
        history_rows+=len(r['history'])
    check('Exactly three current windows per security',current_rows==3*len(expected))
    check('Summary recomputed from all records',all(dict(Counter(r['windows'][key]['status'] for r in d['records']))==d['summary']['window_statuses'][key] for key in ['20','60','120']))
    check('No raw daily OHLC series in derived output',all('daily' not in r for r in d['records']))
    from test_model import ModelTests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    check('Meaningful model tests pass',result.wasSuccessful())
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'run_id':d['run_id'],'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),
        'checks':checks,'unit_tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)},'current_rows':current_rows,'history_rows':history_rows,
        'maximum_probability_sum_error':max_prob_error,'low_liquidity_60':low_liquidity,'effect_validation':False,'human_type_probability_calibration':False}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
