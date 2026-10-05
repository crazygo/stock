#!/usr/bin/env python3
"""Recover empty/stale subscription reads into new files, preserving the originals."""
from __future__ import annotations
import fcntl,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import futu as ft
import pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT));from context import CACHE

def save(path,j):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    lock=(CACHE/'futu_daily_v4_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+240*60
    while True:
        pointer=json.loads((CACHE/'futu_daily_v4_latest.json').read_text());base=Path(pointer['path']);state=json.loads((base/'acquisition.json').read_text())
        if state.get('error'):raise RuntimeError('Full registered source failed; no download-order recovery cohort')
        if state.get('terminal') and state.get('full_registered_pool'):break
        if time.monotonic()>deadline:raise TimeoutError('Full source wait expired')
        print('recovery waiting full registered source',state.get('completed'),flush=True);time.sleep(45)
    asof=state['asof'];out=base/'recovery';out.mkdir(exist_ok=True);(out/'daily').mkdir(exist_ok=True);(out/'rehab').mkdir(exist_ok=True)
    candidates=[t for t,r in state['results'].items() if r.get('status')!='available' or not r.get('current_session_covered') or not r.get('raw_rows')]
    status=out/'acquisition.json';recovery=json.loads(status.read_text()) if status.exists() else {'results':{}}
    recovery.update(asof=asof,source_snapshot_path=str(base),registered_recovery_candidates=candidates,original_sources_preserved=True,terminal=False)
    def checkpoint(terminal=False):
        recovery.update(updated_at=datetime.now(timezone.utc).isoformat(),terminal=terminal,completed=len(recovery['results']),registered_count=len(candidates))
        save(status,recovery);save(CACHE/'futu_daily_v4_recovery_status.json',{k:v for k,v in recovery.items() if k!='results'})
    checkpoint();ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111);last=0.
    def call(method,*a,**kw):
        nonlocal last
        pause=1.2-(time.monotonic()-last)
        if pause>0:time.sleep(pause)
        last=time.monotonic();return method(*a,**kw)
    try:
        def subscribe(tickers):
            ret,err=call(q.subscribe,['US.'+t for t in tickers],[ft.SubType.K_DAY],is_first_push=True,subscribe_push=False)
            if ret==ft.RET_OK:return tickers
            if any(s in str(err).lower() for s in ['quota','limit','频','额度','断开']):raise RuntimeError('Recovery subscription rate/capacity/connection failure')
            if len(tickers)>1:
                mid=len(tickers)//2;return subscribe(tickers[:mid])+subscribe(tickers[mid:])
            t=tickers[0];recovery['results'][t]={'ticker':t,'status':'recovery_subscription_unavailable','error':str(err)[:200]};return []
        pending=[t for t in candidates if recovery['results'].get(t,{}).get('status')!='available']
        for start in range(0,len(pending),100):
            batch=subscribe(pending[start:start+100])
            if not batch:checkpoint();continue
            for _ in range(3):time.sleep(21)
            try:
                for t in batch:
                    record={'ticker':t,'asof':asof,'source':'Futu get_cur_kline K_DAY','price_basis':'NONE','tier':'OpenD_recovery',
                            'original_source_record':state['results'][t],'subscription_minimum_wait_seconds':63}
                    ret,frame=call(q.get_cur_kline,'US.'+t,1000,ktype=ft.KLType.K_DAY,autype=ft.AuType.NONE)
                    valid=ret==ft.RET_OK and len(frame)>0 and {'code','time_key','open','high','low','close','volume'}.issubset(frame.columns)
                    if not valid or not (frame.code=='US.'+t).all() or frame.time_key.duplicated().any():
                        record.update(status='recovery_daily_unavailable');recovery['results'][t]=record;continue
                    dp=out/'daily'/f'{t}.parquet';frame.to_parquet(dp,compression='zstd',compression_level=7,index=False)
                    complete=frame[frame.time_key.astype(str).str[:10]<=asof]
                    record.update(daily_path=str(dp.resolve()),daily_sha256=sha(dp),raw_rows=len(frame),raw_first=str(frame.time_key.min()),raw_last=str(frame.time_key.max()),
                                  current_session_covered=bool(len(complete) and str(complete.time_key.max())[:10]==asof),daily_received_at=datetime.now(timezone.utc).isoformat())
                    ret,events=call(q.get_rehab,'US.'+t)
                    if ret!=ft.RET_OK:record.update(status='recovery_corporate_actions_unavailable')
                    else:
                        rp=out/'rehab'/f'{t}.parquet';events.to_parquet(rp,compression='zstd',compression_level=7,index=False)
                        record.update(rehab_path=str(rp.resolve()),rehab_sha256=sha(rp),rehab_rows=len(events),status='available',rehab_received_at=datetime.now(timezone.utc).isoformat())
                    recovery['results'][t]=record;checkpoint();print('source recovery',t,record['status'],record.get('current_session_covered'),flush=True)
            finally:
                ret,msg=call(q.unsubscribe,['US.'+t for t in batch],[ft.SubType.K_DAY])
                if ret!=ft.RET_OK:raise RuntimeError('Own recovery subscription cleanup unconfirmed')
        checkpoint(True)
    finally:q.close()

if __name__=='__main__':main()
