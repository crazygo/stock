"""Bounded Local -> R2 -> OpenD daily data acquisition. Never uploads."""
from pathlib import Path
import argparse, gzip, json, os, subprocess, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'market_data/trend_quadrant_v1/daily'
sys.path.insert(0,str(ROOT))

def write(d):
    p=OUT/'acquisition.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def valid(f,end):
    if f.empty:return f
    f=f.copy();f['day']=f.time_key.astype(str).str[:10];f=f[f.day<=end].sort_values('day')
    f=f.drop_duplicates('day',keep='last')
    px=f[['open','high','low','close']].to_numpy(float)
    ok=np.isfinite(px).all(axis=1)&(px>0).all(axis=1)&(px[:,1]>=np.maximum(px[:,0],px[:,3]))&(px[:,2]<=np.minimum(px[:,0],px[:,3]))
    return f[ok]

def worker(end,budget):
    started=time.monotonic();deadline=started+budget
    watch=json.loads((OUT/'watchlist.json').read_text())
    entries={r['code']:r for r in watch['groups']['全部'] if r['kind'] in ['STOCK','ETF']}
    favorite={r['code'] for r in watch['groups'].get('特别关注',[])}
    calendar=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
    days=[r['session_date'] for r in calendar if '2025-09-01'<=r['session_date']<=end]
    d={'requested_as_of':end,'started_at':datetime.now(timezone.utc).isoformat(),'budget_seconds':budget,'no_r2_upload':True,'entries':{},'r2_checks':[]}
    # The daily archive can cover many symbols in one read. Raw archive remains immutable.
    latest_local=max(p.parent.name for p in (ROOT/'market_data/us').glob('*/grouped.json.gz'))
    from scripts.r2_client import R2Client
    client=R2Client(timeout=5,deadline=min(deadline,started+35))
    for day in [x for x in days if x>latest_local]:
        p=ROOT/'market_data/us'/day/'grouped.json.gz'
        if p.exists():continue
        try:
            key='market_data/us/'+day+'/grouped.json.gz'
            h=client.head_object(key)
            if h:
                temp=ROOT/'.cache/ai_trend_quadrant_v1'/day/'grouped.json.gz'
                client.get_object(key,temp)
                payload=json.loads(gzip.decompress(temp.read_bytes()))
                if payload.get('adjusted') is True and len(payload.get('results',[]))>1000:
                    p.parent.mkdir(parents=True,exist_ok=True);temp.replace(p)
                    d['r2_checks'].append({'day':day,'status':'downloaded_valid_split_adjusted'})
                else:d['r2_checks'].append({'day':day,'status':'invalid_archive_retained_in_cache'})
            else:d['r2_checks'].append({'day':day,'status':'absent'})
        except Exception as e:d['r2_checks'].append({'day':day,'status':'failed','error':type(e).__name__})
    # Check the dedicated namespace once for all symbols, rather than issuing a HEAD per ticker.
    try:
        objects=client.list_objects('market_data/trend_quadrant_v1/daily/')
        remote={r['key']:r for r in objects}
        d['r2_daily_namespace']={'status':'read','objects':len(remote)}
    except Exception as e:
        remote={};d['r2_daily_namespace']={'status':'failed','error':type(e).__name__}
    # Existing hourly archive is not spliced into a differently adjusted daily series.
    # Its remote modification dates bound whether it could contain newly completed sessions.
    try:
        hourly=client.list_objects('market_data/us_60m/')
        d['r2_hourly_namespace']={'status':'read','objects':len(hourly),
            'possibly_updated_after_local_daily_cutoff':sum(r['last_modified'][:10]>latest_local for r in hourly)}
    except Exception as e:d['r2_hourly_namespace']={'status':'failed','error':type(e).__name__}
    local_rows={code:[] for code in entries if code.startswith('US.')}
    for p in sorted((ROOT/'market_data/us').glob('*/grouped.json.gz')):
        day=p.parent.name
        if not '2025-09-01'<=day<=end:continue
        j=json.loads(gzip.decompress(p.read_bytes()))
        if j.get('adjusted') is not True:continue
        for row in j['results']:
            code='US.'+row['T']
            if code in local_rows:
                local_rows[code].append({'time_key':day+' 00:00:00','open':row['o'],'high':row['h'],'low':row['l'],'close':row['c'],'volume':row['v']})
    pending=[]
    for code in entries:
        path=DATA/(code+'.parquet');f=None
        if path.exists():
            try:f=valid(pd.read_parquet(path),end)
            except Exception:pass
        if f is None and code in local_rows and local_rows[code]:f=valid(pd.DataFrame(local_rows[code]),end)
        if f is not None and not f.empty:
            row={'path':str(path.relative_to(ROOT)) if path.exists() else None,'source':'local_futu_daily' if path.exists() else 'local_massive_grouped',
                'last_day':str(f.day.max()),'bars':len(f),'status':'historical_available'}
            d['entries'][code]=row
            if str(f.day.max())==end and len(f)>=121:
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(path,compression='zstd',compression_level=7,index=False)
                    row['path']=str(path.relative_to(ROOT));row['price_basis']='split_adjusted_Massive'
                row['status']='complete_current';continue
        key='market_data/trend_quadrant_v1/daily/'+code+'.parquet'
        if key in remote:
            try:
                client.get_object(key,path);f=valid(pd.read_parquet(path),end)
                if not f.empty:
                    d['entries'][code]={'path':str(path.relative_to(ROOT)),'source':'r2_dedicated_daily','last_day':str(f.day.max()),'bars':len(f),'status':'historical_available'}
                    if str(f.day.max())==end and len(f)>=121:
                        d['entries'][code]['status']='complete_current';continue
            except Exception as e:d['entries'].setdefault(code,{})['r2_error']=type(e).__name__
        pending.append(code)
    write(d)
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        # Serialize calls with a 1.2s pace; save each successful file atomically.
        order=sorted(pending,key=lambda c:(c not in favorite,not c.startswith('US.'),c))
        for i,code in enumerate(order):
            if time.monotonic()+2>deadline:
                for c in order[i:]:d['entries'].setdefault(c,{})['fetch_status']='budget_exhausted'
                break
            began=time.monotonic()
            try:
                ret,f,key=q.request_history_kline(code,start='2025-09-01',end=end,ktype=ft.KLType.K_DAY,autype=ft.AuType.QFQ,max_count=1000)
                if ret!=ft.RET_OK:
                    d['entries'].setdefault(code,{})['fetch_status']='OpenD_error'
                    d['entries'][code]['fetch_error']=str(f)[:220]
                else:
                    f=valid(f,end)
                    if f.empty:raise ValueError('no_valid_bars')
                    path=DATA/(code+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.parquet.tmp');f.to_parquet(temp,compression='zstd',compression_level=7,index=False);temp.replace(path)
                    d['entries'][code]={'path':str(path.relative_to(ROOT)),'source':'OpenD_daily_QFQ','price_basis':'QFQ_full_series_single_capture',
                        'last_day':str(f.day.max()),'bars':len(f),'received_at':datetime.now(timezone.utc).isoformat(),
                        'status':'complete_current' if str(f.day.max())==end and len(f)>=121 else 'current_short_history' if str(f.day.max())==end else 'historical_available',
                        'market_calendar':'official_Nasdaq' if code.startswith('US.') else 'provider_observed_sessions_not_officially_verified'}
            except Exception as e:d['entries'].setdefault(code,{})['fetch_status']=type(e).__name__
            write(d)
            if i%15==0:print(json.dumps({'progress':i+1,'total':len(order),'code':code,'status':d['entries'][code].get('status',d['entries'][code].get('fetch_status'))}),flush=True)
            time.sleep(max(0,1.2-(time.monotonic()-began)))
    finally:q.close()
    # Preserve locally available historical series for failed current fetches, in a separate file.
    for code,rows in local_rows.items():
        if rows and not d['entries'].get(code,{}).get('path'):
            f=valid(pd.DataFrame(rows),end);path=DATA/(code+'.historical.parquet')
            path.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(path,compression='zstd',compression_level=7,index=False)
            d['entries'].setdefault(code,{}).update(path=str(path.relative_to(ROOT)),source='local_massive_grouped',price_basis='split_adjusted_Massive',last_day=str(f.day.max()),bars=len(f),status='historical_available')
    d['finished_at']=datetime.now(timezone.utc).isoformat();d['elapsed_seconds']=round(time.monotonic()-started,2);write(d)
    from collections import Counter
    print(json.dumps({'statuses':dict(Counter(r.get('status','unavailable') for r in d['entries'].values())),'seconds':d['elapsed_seconds']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--as-of',required=True);p.add_argument('--budget',type=float,default=360);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker(a.as_of,a.budget);return
    try:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--as-of',a.as_of,'--budget',str(a.budget)],timeout=a.budget+5,check=True)
    except subprocess.TimeoutExpired:
        d=json.loads((OUT/'acquisition.json').read_text());d['hard_timeout']=True;write(d);print('Acquisition timed out; completed files and per-symbol evidence preserved.')

if __name__=='__main__':main()
