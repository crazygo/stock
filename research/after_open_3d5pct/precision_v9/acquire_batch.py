"""Resumable, frozen-scope R03 acquisition. No training and no cloud uploads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import futu as ft
import numpy as np
import pandas as pd

from scripts.r2_client import R2Client
from scripts.fetch_research_data import normalize_kline_dataframe
from .acquire_pilot import normalize_sessions
from .evaluate import HERE, ROOT, GROUPS, sha, write

MONTHS=[('2025-08-01','2025-08-31'),('2025-09-01','2025-09-30'),('2025-10-01','2025-10-31'),
        ('2025-11-01','2025-11-30'),('2025-12-01','2025-12-31')]
CLOSED={'2025-09-01','2025-11-27','2025-12-25'}
EARLY={'2025-11-28','2025-12-24'}


def make_calendar():
    out=[]
    for day in pd.bdate_range('2025-08-01','2025-12-31').strftime('%Y-%m-%d'):
        if day in CLOSED: continue
        opened=pd.Timestamp(day+' 09:30',tz='America/New_York')
        closed=pd.Timestamp(day+(' 13:00' if day in EARLY else ' 16:00'),tz='America/New_York')
        out.append({'session_date':day,'open_at':opened.tz_convert('UTC').isoformat(),
                    'close_at':closed.tz_convert('UTC').isoformat(),'duration_minutes':int((closed-opened).total_seconds()/60)})
    old=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    out += [s for s in old['sessions'] if '2026-01-01'<=s['session_date']<='2026-09-24']
    return {'calendar_id':'precision_v9_202508_202609','sessions':out,
            'sources':['https://www.nasdaqtrader.com/content/technicalsupport/2025tradingcalendar.pdf',
                       'https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2025-92',
                       'https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2025-101',
                       'https://www.nasdaq.com/market-activity/stock-market-holiday-schedule']}


def quality(f, start, end, cal):
    f=f[(f.session_date>=start)&(f.session_date<=end)]
    sessions=[s for s in cal['sessions'] if start<=s['session_date']<=end]
    expected=[]
    for s in sessions:
        expected.extend(pd.date_range(s['open_at'],periods=s['duration_minutes']//5,freq='5min'))
    expected=pd.DatetimeIndex(expected)
    actual=pd.to_datetime(f.start_at,utc=True)
    missing=expected.difference(actual)
    v=f[['open','high','low','close']].to_numpy(float)
    bad=(~np.isfinite(v).all(1)|(v[:,0]<=0)|(v[:,2]<=0)|(v[:,1]<np.max(v[:,[0,2,3]],axis=1))|(v[:,2]>np.min(v[:,[0,1,3]],axis=1)))
    basis=f.price_basis.unique().tolist()
    return {'rows':len(f),'expected_rth_bars':len(expected),'missing_rth_bars':len(missing),
            'missing_by_date':pd.Series(missing.tz_convert('America/New_York').strftime('%Y-%m-%d')).value_counts().to_dict(),
            'invalid_ohlc':int(bad.sum()),'duplicates':int(f.start_at.duplicated().sum()),'basis':basis,
            'rth_complete':bool(len(missing)==0 and not bad.any() and not f.start_at.duplicated().any() and basis==['NONE'])}


def normalize(raw, symbol):
    f=normalize_sessions(normalize_kline_dataframe(raw,symbol,'5m','NONE'))
    # Respect official early closes in the new data version.
    start=pd.to_datetime(f.start_at,utc=True).dt.tz_convert('America/New_York')
    minutes=start.dt.hour*60+start.dt.minute
    f.loc[f.session_date.isin(EARLY)&(minutes>=780)&(minutes<1200),'session_type']='post_market'
    return f


def run(output, max_minutes=180):
    source=HERE.parent/'runs/focus_v8_20260927'
    universe=json.loads((source/'inputs/market_data/universe/qqq_retrospective_v1.json').read_text())
    all_symbols={x['symbol'] for x in universe['members'] if x['role']=='candidate'}|{'QQQ'}
    first=set(sum(GROUPS.values(),[]))|{'QQQ'}
    symbols=sorted(first)+sorted(all_symbols-first)
    config={'symbols':symbols,'months':MONTHS,'source':str(source),'no_upload':True,'session':'ALL','basis':'NONE',
            'code_sha':sha(Path(__file__)),'normalizer_sha':sha(HERE/'acquire_pilot.py'),
            'backlog_sha':sha(HERE/'rounds/R03/BACKLOG.md'),'matrix_sha':sha(HERE/'rounds/R03/MATRIX.md')}
    output.mkdir(parents=True,exist_ok=True)
    reg=output/'registration.json'
    if reg.exists():
        saved=json.loads(reg.read_text())
        if saved['config']!=json.loads(json.dumps(config)): raise ValueError('changed frozen job identity')
    else:
        write(reg,{'created_at':datetime.now(timezone.utc).isoformat(),'config':config})
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for p in [Path(__file__),HERE/'acquire_pilot.py']:shutil.copy2(p,snapshot/p.name)
    cal=make_calendar();write(output/'calendar.json',cal)
    r2=R2Client();quote=None;started=time.monotonic();checked=set();remote={};results=[]
    lock=open('/tmp/stock_futu_acquisition.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def context():
        nonlocal quote
        if quote is None:
            ft.SysConfig.enable_proto_encrypt(False);quote=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
            ret,q=quote.get_history_kl_quota(get_detail=False)
            if ret==ft.RET_OK:write(output/f'quota_before_{int(time.time())}.json',{'used':int(q[0]),'remaining':int(q[1])})
        return quote
    try:
        for symbol in symbols:
            for start,end in MONTHS:
                part=output/'parts'/symbol/start[:7];done=part/'result.json'
                if done.exists():
                    result=json.loads(done.read_text());results.append(result)
                    if result['pagination_complete']:continue
                    raise ValueError(f'failed part retained; register recovery before retry: {part}')
                if time.monotonic()-started>max_minutes*60:
                    write(output/'progress.json',{'status':'runtime_checkpoint','completed_parts':len(results),'total_parts':len(symbols)*len(MONTHS),'results':results});return
                part.mkdir(parents=True,exist_ok=False);timer=time.monotonic();audit=[];f=None;tier=None
                local_paths=[ROOT/'market_data/us_5m'/symbol/'2025.parquet',
                    HERE.parent/'runs/precision_v9_20260927/R02_normalized/market_data/us_5m'/symbol/'2025.parquet']
                for p in local_paths:
                    if p.exists():
                        z=pd.read_parquet(p)
                        if quality(z,start,end,cal)['rth_complete']:
                            f=z[(z.session_date>=start)&(z.session_date<=end)].copy();tier='local';audit.append({'source':str(p),'sha':sha(p)});break
                if f is None:
                    if symbol not in checked:
                        key=f'us_5m/{symbol}/2025.parquet';cache=output/'r2_cache'/key
                        try:
                            found=r2.head_object(key);audit.append({'tier':'r2','key':key,'found':found is not None})
                            if found:r2.get_object(key,cache);remote[symbol]=cache
                        except Exception as e:audit.append({'tier':'r2','error_type':type(e).__name__})
                        checked.add(symbol)
                    if symbol in remote:
                        z=pd.read_parquet(remote[symbol])
                        if quality(z,start,end,cal)['rth_complete']:
                            f=z[(z.session_date>=start)&(z.session_date<=end)].copy();tier='r2'
                complete=True;failure=None
                if f is None:
                    tier='futu';pages=[];cursor=None
                    for page in range(30):
                        time.sleep(1.2 if page==0 else .3)
                        for attempt in range(3):
                            sent=datetime.now(timezone.utc).isoformat()
                            ret,chunk,nxt=context().request_history_kline('US.'+symbol,start=start,end=end,
                                ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,extended_time=True,session=ft.Session.ALL,page_req_key=cursor)
                            event={'page':page+1,'attempt':attempt+1,'requested_at':sent,'received_at':datetime.now(timezone.utc).isoformat(),'ok':ret==ft.RET_OK}
                            if ret==ft.RET_OK:
                                raw=part/f'raw_{page+1:03d}.parquet';chunk.to_parquet(raw,index=False,compression='zstd',compression_level=7)
                                event.update(rows=len(chunk),sha=sha(raw),has_more=nxt is not None);audit.append(event);break
                            event['error']=str(chunk)[:250];audit.append(event)
                            if attempt<2:time.sleep(30)
                        write(part/'audit.json',audit)
                        if ret!=ft.RET_OK:complete=False;failure=str(chunk)[:250];break
                        pages.append(chunk);cursor=nxt
                        if cursor is None:break
                    if cursor is not None:complete=False;failure=failure or 'page_cap'
                    if pages:
                        raw=pd.concat(pages,ignore_index=True)
                        if raw.time_key.duplicated().any():complete=False;failure='duplicate_source_times'
                        f=normalize(raw,symbol)
                if f is not None:f.to_parquet(part/'bars.parquet',index=False,compression='zstd',compression_level=7)
                result={'symbol':symbol,'month':start[:7],'tier':tier,'pagination_complete':complete,'failure':failure,
                        'quality':quality(f,start,end,cal) if f is not None else None,'seconds':time.monotonic()-timer,
                        'bars_sha256':sha(part/'bars.parquet') if f is not None else None}
                write(part/'audit.json',audit);write(done,result);results.append(result)
                write(output/'progress.json',{'status':'running','completed_parts':len(results),'total_parts':len(symbols)*len(MONTHS),'results':results})
                print(json.dumps({'symbol':symbol,'month':start[:7],'tier':tier,'complete':complete,'rth_complete':result['quality']['rth_complete'] if f is not None else False,'done':len(results),'total':len(symbols)*len(MONTHS)}),flush=True)
                if not complete:raise RuntimeError('acquisition failed; preserved evidence')
        write(output/'summary.json',{'status':'acquisition_complete','results':results,'seconds':time.monotonic()-started,'no_training':True})
    finally:
        if quote:
            ret,q=quote.get_history_kl_quota(get_detail=False)
            if ret==ft.RET_OK:write(output/f'quota_after_{int(time.time())}.json',{'used':int(q[0]),'remaining':int(q[1])})
            quote.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-minutes',type=int,default=180);args=parser.parse_args()
    run(args.output.resolve(),args.max_minutes)
