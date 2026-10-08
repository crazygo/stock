"""Compute a reproducible current-watchlist descriptive map; no training labels."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,sys
import numpy as np
import pandas as pd
from model import VERSION,FEATURE_NAMES,DATUM,SCALE,WX,WY,NAMES,describe,smooth

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
HKCAL=json.loads((OUT/'hk_calendar_2026.json').read_text())
HKDAYS=[x.strftime('%Y-%m-%d') for x in pd.bdate_range(HKCAL['start'],HKCAL['end']) if x.strftime('%Y-%m-%d') not in HKCAL['holidays']]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def model_row(frame,code,cut,window,us_days,sensitivity=False):
    g=frame[frame.day<=cut].tail(window+1)
    if len(g)<window+1:return {'as_of':cut,'window':window,'status':'insufficient_history','available_closes':len(g)}
    verified=code.startswith('US.') or (code.startswith('HK.') and str(g.day.iloc[0])>=HKCAL['start'])
    if verified:
        market_days=us_days if code.startswith('US.') else HKDAYS
        expected=[d for d in market_days if g.day.iloc[0]<=d<=cut]
        if list(g.day)!=expected or len(expected)!=window+1:
            return {'as_of':cut,'window':window,'status':'missing_sessions','missing_days':sorted(set(expected)-set(g.day)),
                'actual_last_day':str(g.day.iloc[-1])}
    c=np.log(g.close.to_numpy(float));o=np.log(g.open.to_numpy(float));hl=np.log(g.high.to_numpy(float)/g.low.to_numpy(float))
    a=np.column_stack([np.diff(c),o[1:]-c[:-1],c[1:]-o[1:],hl[1:]])
    if not np.isfinite(a).all():return {'as_of':cut,'window':window,'status':'invalid_prices'}
    row=describe(a,code,cut,window,sensitivity);row.update(status='scored',start=str(g.day.iloc[0]),actual_last_day=str(g.day.iloc[-1]),
        calendar_verified=verified,low_liquidity_observed=bool((g.volume.to_numpy(float)<=0).mean()>.1))
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--as-of',default='2026-10-06');a=p.parse_args();cut=a.as_of
    watch=json.loads((OUT/'watchlist.json').read_text());acq=json.loads((OUT/'acquisition.json').read_text())
    quality=json.loads((ROOT/'analysis/ai_value_chain_map_v1/quality/current.json').read_text())
    cal=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    us_days=[r['session_date'] for r in cal['sessions'] if r['session_date']<=cut]
    weekly={}
    for day in us_days:
        ts=pd.Timestamp(day);weekly[(ts.isocalendar().year,ts.isocalendar().week)]=day
    ticks=list(weekly.values())[-16:]
    favorites={r['code'] for r in watch['groups'].get('特别关注',[])};etfs={r['code'] for r in watch['groups'].get('ETF',[])}
    records=[];excluded=[];provenance=[];feature_rows=[]
    leveraged={'US.TQQQ':'Nasdaq-100每日3倍做多','US.SQQQ':'Nasdaq-100每日3倍反向','US.SOXL':'半导体每日3倍做多','US.SOXS':'半导体每日3倍反向',
        'US.QLD':'Nasdaq-100每日2倍做多','US.QID':'Nasdaq-100每日2倍反向','US.UPRO':'S&P 500每日3倍做多','US.SPXU':'S&P 500每日3倍反向'}
    seen=set()
    for raw in watch['groups']['全部']:
        code=raw['code']
        if code in seen:continue
        seen.add(code)
        if raw['kind'] not in ['STOCK','ETF']:
            excluded.append(dict(raw,reason='行业列表等非股票/基金实体'));continue
        q=quality['records'].get(code,{})
        grade=q.get('grade')
        current=bool(q.get('expires_at') and q['expires_at']>=watch['observed_at'][:10])
        record=dict(raw,symbol=code.split('.',1)[1],favorite=code in favorites,etf_group=code in etfs,ai_related=code in quality['records'],
            quality_grade=grade if current else None,quality_as_of=q.get('as_of'),quality_pending=not (grade and current),
            structure=leveraged.get(code) or ('名称提示多倍/反向，产品条款未逐只核验' if any(s in raw['name'] for s in ['倍做多','倍做空','杠杆','反向','Ultra','Bull','Bear']) else None),
            windows={},history=[],coverage=acq['entries'].get(code,{}))
        pathstr=record['coverage'].get('path')
        if pathstr:
            path=ROOT/pathstr
            try:
                f=pd.read_parquet(path);f['day']=f.time_key.astype(str).str[:10];f=f[f.day<=cut].sort_values('day').drop_duplicates('day',keep='last')
                px=f[['open','high','low','close']].to_numpy(float)
                good=np.isfinite(px).all(axis=1)&(px>0).all(axis=1)&(px[:,1]>=np.maximum(px[:,0],px[:,3]))&(px[:,2]<=np.minimum(px[:,0],px[:,3]))
                f=f[good]
                listing=raw.get('listing_date','')
                if len(listing)==10 and listing[4]=='-' and listing>'1970-01-01':
                    removed=int((f.day<listing).sum());f=f[f.day>=listing]
                    record['excluded_pre_listing_bars']=removed
                provenance.append({'code':code,'path':pathstr,'sha256':sha(path),'source':record['coverage'].get('source'),'valid_rows':len(f),'last_day':str(f.day.max())})
                for n in [20,60,120]:record['windows'][str(n)]=model_row(f,code,cut,n,us_days,True)
                for tick in ticks:record['history'].append(model_row(f,code,tick,60,us_days))
                smooth(record['history'])
                # The current result is the last causal weekly update. Its sensitivity remains the raw measurement's.
                current60=record['windows']['60']
                if current60.get('status')=='scored' and record['history'][-1].get('status')=='scored':
                    h=record['history'][-1]
                    for field in ['p','dominant','entropy','smoothing_reset','observed_stable_weeks']:current60[field]=h[field]
                    current60['clear_tendency']=bool(current60['clear_tendency'] and max(current60['p'])>=.65 and current60['dominant']==int(np.argmax(current60['raw_p'])))
                if current60.get('status')=='scored':feature_rows.append(current60['measurements'])
            except Exception as e:record['data_error']=type(e).__name__+': '+str(e)[:180]
        for n in [20,60,120]:record['windows'].setdefault(str(n),{'as_of':cut,'window':n,'status':'unavailable'})
        records.append(record)
        if len(records)%30==0:print('Computed',len(records),'/',len(watch['groups']['全部'])-len(excluded),flush=True)
    summary={'raw_watchlist_rows':len(watch['groups']['全部']),'securities':len(records),'excluded_non_securities':len(excluded),
        'window_statuses':{str(n):dict(Counter(r['windows'][str(n)]['status'] for r in records)) for n in [20,60,120]},
        'dominant_60':dict(Counter(NAMES[r['windows']['60']['dominant']] for r in records if r['windows']['60']['status']=='scored')),
        'clear_60':sum(r['windows']['60'].get('clear_tendency',False) for r in records),
        'current_quality_A_or_Aplus':sum(r['quality_grade'] in ['A+','A'] for r in records),
        'unverified_calendar_scored':sum(r['windows']['60']['status']=='scored' and not r['windows']['60']['calendar_verified'] for r in records)}
    bundle={'model_version':VERSION,'generated_at':datetime.now(timezone.utc).isoformat(),'as_of':cut,'watchlist_observed_at':watch['observed_at'],
        'probability_kind':'conditional_Pugh_block_resampling_membership_not_calibrated_true_class_probability',
        'validation':{'human_type_calibration':False,'strategy_effect_validation':False,'prospective_prediction':False},
        'classes':NAMES,'feature_names':FEATURE_NAMES,'datum':DATUM.tolist(),'scale':SCALE.tolist(),'weight_x':WX.tolist(),'weight_y':WY.tolist(),
        'history_dates':ticks,'records':records,'excluded':excluded,'summary':summary,'source_sha256':{'watchlist':sha(OUT/'watchlist.json'),
            'protocol':sha(OUT/'PROTOCOL.md'),'quality':sha(ROOT/'analysis/ai_value_chain_map_v1/quality/current.json'),'hk_calendar':sha(OUT/'hk_calendar_2026.json')},
        'provenance':provenance,'source_code_sha256':{'model':sha(OUT/'model.py'),'build':sha(OUT/'build.py')},
        'feature_correlation':np.nan_to_num(np.corrcoef(np.array(feature_rows).T),nan=0).tolist() if len(feature_rows)>2 else None}
    fingerprint=json.dumps({'sources':bundle['source_sha256'],'code':bundle['source_code_sha256'],'provenance':provenance,'as_of':cut},sort_keys=True)
    run_id='tq_'+hashlib.sha256(fingerprint.encode()).hexdigest()[:16];bundle['run_id']=run_id
    run=OUT/'runs'/run_id;run.mkdir(parents=True,exist_ok=True)
    text=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    snapshot=run/'snapshot.json'
    if snapshot.exists():text=snapshot.read_text().strip()
    else:snapshot.write_text(text+'\n')
    (OUT/'results.json').write_text(text+'\n')
    fields=['code','name','kind','quality_grade','favorite','window','as_of','status','p1','p2','p3','p4','x','y','direction','price_return','annualized_vol','max_drawdown','clear_tendency','observed_stable_weeks']
    with (OUT/'results.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader()
        for r in records:
            for n,t in r['windows'].items():
                row={k:r.get(k) for k in ['code','name','kind','quality_grade','favorite']};row.update({k:t.get(k) for k in ['as_of','status','x','y','direction','price_return','clear_tendency','observed_stable_weeks']});row['window']=n
                if t['status']=='scored':row.update({f'p{i+1}':v for i,v in enumerate(t['p'])});row.update(annualized_vol=t['measurements'][0],max_drawdown=t['measurements'][1])
                w.writerow(row)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
