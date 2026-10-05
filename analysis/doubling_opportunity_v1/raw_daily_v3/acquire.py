#!/usr/bin/env python3
"""Registered broad-pool NONE daily/rehab snapshot; subscription reads only."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,re,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import futu as ft
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from data import CACHE,ROOT
from scripts.model_history_calendar import calendar
from scripts.r2_client import R2Client

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)
def closed_session():
    now=datetime.now(timezone.utc);day=now.astimezone(ZoneInfo('America/New_York')).date()
    s=calendar((day-timedelta(days=14)).isoformat(),day.isoformat())['sessions']
    return [x['session_date'] for x in s if datetime.fromisoformat(x['close_at'])<=now][-1]

class Rate:
    def __init__(self):self.last=0.
    def call(self,method,*args,**kwargs):
        wait=.65-(time.monotonic()-self.last)
        if wait>0:time.sleep(wait)
        self.last=time.monotonic();return method(*args,**kwargs)

def persist_frame(frame,path):
    tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,compression='zstd',compression_level=7,index=False);tmp.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--symbols',nargs='+',help='Explicit source pilot only; cannot count as complete registered pool.')
    ap.add_argument('--refresh',action='store_true',help='Create a new immutable retrieval snapshot.')
    args=ap.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    lock=(CACHE/'futu_daily_v3_acquisition.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ufile=CACHE/'universe.json';uj=json.loads(ufile.read_text());u=uj['stocks']
    members=list(dict.fromkeys(args.symbols or sorted(u)+['SPY']))
    if any(not re.fullmatch(r'[A-Z0-9.]+',t) for t in members):raise ValueError('Unsupported ticker filename')
    asof=closed_session();cohort=hashlib.sha256(json.dumps([members,uj['metadata']['raw_sha256']],sort_keys=True).encode()).hexdigest()
    pointer=CACHE/'futu_daily_v3_latest.json';existing=json.loads(pointer.read_text()) if pointer.exists() else {}
    if not args.refresh and existing.get('cohort_sha256')==cohort and existing.get('asof')==asof:
        base=Path(existing['path'])
    else:
        base=CACHE/'futu_daily_v3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');base.mkdir(parents=True)
        save(pointer,{'path':str(base.resolve()),'asof':asof,'cohort_sha256':cohort})
    registration=base/'registration.json'
    if not registration.exists():save(registration,{'registered_at':stamp(),'asof':asof,'cohort_sha256':cohort,
        'members':members,'equity_candidates':len(u),'benchmark':'SPY','pilot':bool(args.symbols),
        'universe':uj,'protocol_sha256':sha(HERE/'PROTOCOL.md'),'acquisition_code_sha256':sha(Path(__file__)),
        'source':'Futu get_cur_kline K_DAY NONE; get_rehab',
        'R2_prefix':'market_data/us_daily_none/','no_history_kline_calls':True,'no_R2_upload':True})
    if not (base/'protocol_snapshot.md').exists():(base/'protocol_snapshot.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    (base/'daily').mkdir(exist_ok=True);(base/'rehab').mkdir(exist_ok=True);(base/'metadata').mkdir(exist_ok=True)
    status=base/'acquisition.json';state=json.loads(status.read_text()) if status.exists() else {'started_at':stamp(),'results':{}}
    results=state['results'];rate=Rate();q=None
    def checkpoint(terminal=False,error=None):
        state.update(updated_at=stamp(),terminal=terminal,registered_count=len(members),completed=len(results),
          available=sum(v.get('status')=='available' for v in results.values()),unavailable=sum(v.get('status')!='available' for v in results.values()),
          asof=asof,cohort_sha256=cohort,full_registered_pool=not bool(args.symbols),source_files_immutable=True)
        if error:state.update(error=error,terminal=False)
        else:state.pop('error',None)
        save(status,state);save(CACHE/'futu_daily_v3_acquisition.json',{k:v for k,v in state.items() if k!='results'}|{'path':str(base.resolve())})
    checkpoint()
    try:
        r2=R2Client();objects={x['key']:x for x in r2.list_objects('market_data/us_daily_none/')}
        state['r2_checked_at']=stamp();state['r2_matching_namespace_objects']=len(objects)
        # Only pairs with explicit source metadata can satisfy this same-basis snapshot.
        for t in members:
            dp=base/'daily'/f'{t}.parquet';rp=base/'rehab'/f'{t}.parquet';mp=base/'metadata'/f'{t}.json'
            if dp.exists() and rp.exists() and mp.exists():
                meta=json.loads(mp.read_text())
                if meta.get('asof')==asof and sha(dp)==meta.get('daily_sha256') and sha(rp)==meta.get('rehab_sha256'):
                    results[t]=meta;continue
            prefix=f'market_data/us_daily_none/{t}/'
            keys=[prefix+'latest.parquet',prefix+'rehab.parquet',prefix+'metadata.json']
            if all(k in objects for k in keys):
                incoming=base/'metadata'/f'{t}.r2.json';r2.get_object(keys[2],incoming);meta=json.loads(incoming.read_text())
                if meta.get('asof')==asof and meta.get('price_basis')=='NONE' and meta.get('source')=='Futu get_cur_kline K_DAY':
                    r2.get_object(keys[0],dp);r2.get_object(keys[1],rp)
                    if sha(dp)==meta.get('daily_sha256') and sha(rp)==meta.get('rehab_sha256'):
                        meta['tier']='r2';save(mp,meta);results[t]=meta
        pending=[t for t in members if results.get(t,{}).get('status')!='available']
        ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
        ret,hq=rate.call(q.get_history_kl_quota,get_detail=False)
        state['history_quota_before']={'used':hq[0],'remaining':hq[1]} if ret==ft.RET_OK else {'unavailable':True}
        ret,sq=rate.call(q.query_subscription,is_all_conn=True)
        state['subscription_quota_before']={k:sq[k] for k in ['total_used','own_used','remain']} if ret==ft.RET_OK else {'unavailable':True}
        checkpoint()
        subscribed_at={}
        def subscribe(tickers):
            ret,msg=rate.call(q.subscribe,['US.'+t for t in tickers],[ft.SubType.K_DAY],is_first_push=False,subscribe_push=False)
            if ret==ft.RET_OK:
                subscribed_at.update({t:time.monotonic() for t in tickers});return tickers
            text=str(msg)
            if any(x in text.lower() for x in ['quota','limit','频','额度','断开']):raise RuntimeError('Subscription capacity/rate/connection blocked: '+text[:200])
            if len(tickers)>1:
                mid=len(tickers)//2;return subscribe(tickers[:mid])+subscribe(tickers[mid:])
            results[tickers[0]]={'ticker':tickers[0],'status':'subscription_unavailable','error':text[:300],'received_at':stamp()}
            return []
        for pos in range(0,len(pending),100):
            batch=pending[pos:pos+100];started=time.monotonic();subscribed=subscribe(batch)
            try:
                for t in subscribed:
                    record={'ticker':t,'asof':asof,'tier':'OpenD','source':'Futu get_cur_kline K_DAY','price_basis':'NONE'}
                    ret,frame=rate.call(q.get_cur_kline,'US.'+t,1000,ktype=ft.KLType.K_DAY,autype=ft.AuType.NONE)
                    if ret!=ft.RET_OK:
                        record.update(status='daily_unavailable',error=str(frame)[:300],received_at=stamp());results[t]=record;continue
                    dp=base/'daily'/f'{t}.parquet';persist_frame(frame,dp)
                    record.update(daily_path=str(dp),daily_sha256=sha(dp),raw_rows=len(frame),raw_first=str(frame.time_key.min()),raw_last=str(frame.time_key.max()),daily_received_at=stamp())
                    required={'code','time_key','open','high','low','close','volume'}
                    if not required.issubset(frame.columns) or not (frame.code=='US.'+t).all() or frame.time_key.duplicated().any():
                        record.update(status='daily_identity_or_schema_invalid');results[t]=record;continue
                    complete=frame[frame.time_key.astype(str).str[:10]<=asof]
                    record.update(completed_rows=len(complete),last_completed_date=str(complete.time_key.max())[:10] if len(complete) else None,
                                  current_session_covered=bool(len(complete) and str(complete.time_key.max())[:10]==asof))
                    ret,rehab=rate.call(q.get_rehab,'US.'+t)
                    if ret!=ft.RET_OK:
                        record.update(status='corporate_actions_unavailable',error=str(rehab)[:300],received_at=stamp());results[t]=record;continue
                    rp=base/'rehab'/f'{t}.parquet';persist_frame(rehab,rp)
                    record.update(rehab_path=str(rp),rehab_sha256=sha(rp),rehab_rows=len(rehab),rehab_received_at=stamp(),status='available')
                    save(base/'metadata'/f'{t}.json',record);results[t]=record
                    if len(results)%20==0:
                        checkpoint();print('registered NONE daily',len(results),'/',len(members),'available',state['available'],flush=True)
            finally:
                # Only this connection's explicit subscriptions are cancelled.
                if subscribed:
                    while (wait:=61-(time.monotonic()-max(subscribed_at[t] for t in subscribed)))>0:
                        time.sleep(min(wait,30))
                    ret,msg=rate.call(q.unsubscribe,['US.'+t for t in subscribed],[ft.SubType.K_DAY])
                    if ret!=ft.RET_OK:raise RuntimeError('Own subscription cleanup not confirmed: '+str(msg)[:200])
            checkpoint()
        ret,hq=rate.call(q.get_history_kl_quota,get_detail=False)
        state['history_quota_after']={'used':hq[0],'remaining':hq[1]} if ret==ft.RET_OK else {'unavailable':True}
        ret,sq=rate.call(q.query_subscription,is_all_conn=False)
        state['own_subscription_quota_after']={k:sq[k] for k in ['own_used','remain']} if ret==ft.RET_OK else {'unavailable':True}
        checkpoint(True);print(json.dumps({k:v for k,v in state.items() if k!='results'},indent=2),flush=True)
    except BaseException as exc:
        checkpoint(error={'type':type(exc).__name__,'message':str(exc)[:300]});raise
    finally:
        if q is not None:q.close()
        lock.close()

if __name__=='__main__':main()
