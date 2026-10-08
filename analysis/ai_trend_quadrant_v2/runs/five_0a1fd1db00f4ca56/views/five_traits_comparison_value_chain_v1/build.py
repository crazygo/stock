"""Build daily historical descriptor curves from v1's captured local inputs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta,date
from collections import Counter
import hashlib,json,sys
import numpy as np
import pandas as pd
from model import VERSION,KEYS,WINDOWS,NAMES,measure,boot

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;V1=ROOT/'analysis/ai_trend_quadrant_v1'
hk=json.loads((V1/'hk_calendar_2026.json').read_text())
HKDAYS=[d.strftime('%Y-%m-%d') for d in pd.bdate_range(hk['start'],hk['end']) if d.strftime('%Y-%m-%d') not in hk['holidays']]
USDAYS=[d['session_date'] for d in json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def at(f,code,day):
    g=f[f.day<=day].tail(141);values=[None]*5;states=['insufficient_history']*5
    calendar=USDAYS if code.startswith('US.') else HKDAYS if code.startswith('HK.') else None
    if g.empty:return values,states,None
    a=np.diff(np.log(g.close.to_numpy(float)));m=measure(a)[0]
    for k,n in enumerate(WINDOWS):
        if len(g)<n+1:continue
        segment=g.tail(n+1)
        if calendar is None:states[k]='unverified_calendar';continue
        expected=[d for d in calendar if segment.day.iloc[0]<=d<=day]
        if list(segment.day)!=expected or len(expected)!=n+1:states[k]='missing_sessions';continue
        if not np.isfinite(m[k]):states[k]='undefined_zero_variation';continue
        values[k]=float(m[k]);states[k]='scored'
    valid_lengths=[n for k,n in enumerate(WINDOWS) if states[k]=='scored']
    return values,states,a[-max(valid_lengths):] if valid_lengths else None

def main():
    baseline=json.loads((V1/'results.json').read_text());cut=baseline['as_of'];start=(date.fromisoformat(cut)-timedelta(days=59)).isoformat()
    records=[];provenance=[]
    for old in baseline['records']:
        r={k:old.get(k) for k in ['code','name','kind','symbol','favorite','ai_related','quality_grade','structure','listing_date']}
        r['coverage']=old['coverage'];r['history']=[];r['v']=[None]*5;r['status']=['unavailable']*5;r['uncertainty']=[None]*5
        path=old['coverage'].get('path')
        if path:
            p=ROOT/path;f=pd.read_parquet(p);f['day']=f.time_key.astype(str).str[:10];f=f[f.day<=cut].sort_values('day').drop_duplicates('day',keep='last')
            listing=old.get('listing_date','')
            if listing>'1970-01-01':f=f[f.day>=listing]
            px=f[['open','high','low','close']].to_numpy(float)
            valid=np.isfinite(px).all(axis=1)&(px>0).all(axis=1)&(px[:,1]>=np.maximum(px[:,0],px[:,3]))&(px[:,2]<=np.minimum(px[:,0],px[:,3]))
            f=f[valid];r['bars']=len(f)
            r['v'],r['status'],returns=at(f,r['code'],cut)
            raw=boot(returns,r['code'],cut) if returns is not None else [None]*5
            r['uncertainty']=[raw[k] if r['status'][k]=='scored' else None for k in range(5)]
            days=USDAYS if r['code'].startswith('US.') else HKDAYS if r['code'].startswith('HK.') else []
            for day in days:
                if start<=day<=cut:
                    v,status,_=at(f,r['code'],day);r['history'].append({'day':day,'v':v,'status':status})
            recent=f.tail(61);r['zero_volume_warning']=bool(len(recent) and (recent.volume<=0).mean()>.1)
            provenance.append({'code':r['code'],'path':path,'sha256':sha(p)})
        r['mapped']=r['v'][0] is not None and r['v'][1] is not None
        records.append(r)
    summary={'securities':len(records),'mapped':sum(r['mapped'] for r in records),'unmapped':sum(not r['mapped'] for r in records),
        'trait_coverage':{KEYS[k]:dict(Counter(r['status'][k] for r in records)) for k in range(5)}}
    data={'version':VERSION,'as_of':cut,'history_start':start,'generated_at':datetime.now(timezone.utc).isoformat(),
        'watchlist_observed_at':baseline['watchlist_observed_at'],'v1_run_id':baseline['run_id'],'period_kind':'calendar_lookback_not_estimator_window',
        'periods':[14,30,60],'keys':KEYS,'names':NAMES,'windows':WINDOWS,'records':records,'summary':summary,'provenance':provenance,
        'input_sha256':{'v1_results':sha(V1/'results.json'),'protocol':sha(OUT/'PROTOCOL.md'),'model':sha(OUT/'model.py'),'build':sha(OUT/'build.py')}}
    fp=json.dumps({k:data[k] for k in ['version','as_of','input_sha256','provenance']},sort_keys=True)
    data['run_id']='five_'+hashlib.sha256(fp.encode()).hexdigest()[:16]
    (OUT/'results.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
