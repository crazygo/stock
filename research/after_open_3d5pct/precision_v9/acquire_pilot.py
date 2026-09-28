"""Bounded Local -> R2 -> OpenD read-only data pilot with immutable raw pages."""
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

from scripts.fetch_research_data import normalize_kline_dataframe
from scripts.r2_client import R2Client
from .evaluate import HERE, ROOT, sha, write

SYMBOLS = ['AMD', 'COHR', 'MU', 'SNDK', 'QQQ']
START, END = '2025-08-01', '2025-08-29'
CALENDAR_SOURCE = 'https://www.nasdaqtrader.com/content/technicalsupport/2025tradingcalendar.pdf'


def normalize_sessions(frame):
    """Keep midnight-crossing bars in overnight; never discard observed bars."""
    out = frame.copy()
    start = pd.to_datetime(out.start_at, utc=True).dt.tz_convert('America/New_York')
    end = pd.to_datetime(out.end_at, utc=True).dt.tz_convert('America/New_York')
    same_day = start.dt.date == end.dt.date
    a = start.dt.hour*60 + start.dt.minute
    b = end.dt.hour*60 + end.dt.minute
    out['session_type'] = np.select([same_day & (a>=570) & (b<=960),
                                    same_day & (a>=240) & (b<=570),
                                    same_day & (a>=960) & (b<=1200)],
                                    ['regular','pre_market','post_market'], default='overnight')
    return out


def quality(frame):
    f = frame[(frame.session_date >= START) & (frame.session_date <= END)].copy()
    required = pd.bdate_range(START, END).strftime('%Y-%m-%d').tolist()
    full, missing, extra = [], {}, {}
    for d in required:
        expected = pd.date_range(d+' 09:30', periods=78, freq='5min', tz='America/New_York').tz_convert('UTC')
        actual = pd.to_datetime(f.loc[f.session_date == d, 'start_at'], utc=True)
        absent = expected.difference(actual)
        if len(absent): missing[d] = len(absent)
        else: full.append(d)
    vals = f[['open','high','low','close']].to_numpy(float)
    bad = (~np.isfinite(vals).all(1) | (vals[:,0] <= 0) | (vals[:,2] <= 0)
           | (vals[:,1] < np.max(vals[:,[0,2,3]], axis=1)) | (vals[:,2] > np.min(vals[:,[0,1,3]], axis=1)))
    basis = f.price_basis.unique().tolist() if 'price_basis' in f else []
    duplicate = int(f.start_at.duplicated().sum())
    return {'rows': len(f), 'calendar_source': CALENDAR_SOURCE, 'required_rth_sessions': len(required),
            'complete_rth_sessions': len(full), 'missing_rth_bars': missing,
            'duplicates': duplicate, 'invalid_ohlc_rows': int(bad.sum()), 'price_basis': basis,
            'session_counts': {k:int(v) for k,v in f.session_type.value_counts().items()},
            'rth_price_coverage_pass': bool(len(full)==len(required) and duplicate==0 and not bad.any() and basis==['NONE']),
            'availability_quality': 'historical_assumed_bar_end_plus_1s',
            'extended_session_completeness': 'observed_counts_only_not_certified'}


def acquire(output):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'registration.json', {'created_at': datetime.now(timezone.utc).isoformat(),
          'symbols': SYMBOLS, 'start':START, 'end':END, 'price_basis':'NONE', 'interval':'5m', 'session':'ALL',
          'backlog':sha(HERE/'rounds/R02/BACKLOG.md'), 'matrix':sha(HERE/'rounds/R02/MATRIX.md'),
          'code':sha(Path(__file__)), 'calendar_source':CALENDAR_SOURCE, 'no_upload':True})
    lock = open('/tmp/stock_futu_acquisition.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    r2 = R2Client(); quote = None; results=[]
    def context():
        nonlocal quote
        if quote is None:
            ft.SysConfig.enable_proto_encrypt(False)
            quote = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
            ret, q = quote.get_history_kl_quota(get_detail=False)
            if ret == ft.RET_OK: write(output/'quota_before.json', {'used':int(q[0]), 'remaining':int(q[1])})
        return quote
    try:
        for symbol in SYMBOLS:
            started=time.monotonic(); dest=output/'market_data/us_5m'/symbol/'2025.parquet'
            dest.parent.mkdir(parents=True)
            local=ROOT/'market_data/us_5m'/symbol/'2025.parquet'
            f=None; source=None; audit=[]
            if local.exists():
                candidate=pd.read_parquet(local)
                if quality(candidate)['rth_price_coverage_pass']:
                    f=candidate; source='local'; shutil.copy2(local,dest)
            if f is None:
                key=f'us_5m/{symbol}/2025.parquet'
                try:
                    hit=r2.head_object(key)
                    audit.append({'tier':'r2','key':key,'found':hit is not None})
                    if hit:
                        cache=output/'r2_cache'/key
                        r2.get_object(key,cache)
                        candidate=pd.read_parquet(cache)
                        if quality(candidate)['rth_price_coverage_pass']:
                            f=candidate; source='r2'; shutil.copy2(cache,dest)
                except Exception as e:
                    audit.append({'tier':'r2','status':'lookup_failed','error_type':type(e).__name__})
            complete=True; failure=None
            if f is None:
                source='futu'; pages=[]; cursor=None; page=0
                while True:
                    time.sleep(1.2 if page==0 else .3)
                    ret=None
                    for attempt in range(3):
                        sent=datetime.now(timezone.utc).isoformat()
                        ret, chunk, nxt=context().request_history_kline('US.'+symbol,start=START,end=END,
                                ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,
                                extended_time=True,session=ft.Session.ALL,page_req_key=cursor)
                        received=datetime.now(timezone.utc).isoformat()
                        event={'tier':'futu','page':page+1,'attempt':attempt+1,'requested_at':sent,
                               'received_at':received,'status':'ok' if ret==ft.RET_OK else 'error'}
                        if ret==ft.RET_OK:
                            raw=output/'raw_pages'/symbol/f'{page+1:03d}.parquet'; raw.parent.mkdir(parents=True,exist_ok=True)
                            chunk.to_parquet(raw,index=False,compression='zstd',compression_level=7)
                            event.update(rows=len(chunk),sha256=sha(raw),has_more=nxt is not None)
                            audit.append(event); break
                        event['error']=str(chunk)[:250]; audit.append(event)
                        if attempt<2: time.sleep(30)
                    if ret!=ft.RET_OK:
                        complete=False; failure=str(chunk)[:250]; break
                    pages.append(chunk); page+=1; cursor=nxt
                    if cursor is None: break
                    if page>=30:
                        complete=False; failure='registered_30_page_cap'; break
                if pages:
                    raw=pd.concat(pages,ignore_index=True)
                    # Reject duplicate raw timestamps before the legacy normalizer can remove them.
                    if raw.time_key.duplicated().any():
                        complete=False; failure='duplicate_source_timestamp'
                    f=normalize_sessions(normalize_kline_dataframe(raw,symbol,'5m','NONE'))
                    f.to_parquet(dest,index=False,compression='zstd',compression_level=7)
            ca_dest=output/'market_data/corporate_actions'/f'{symbol}.parquet'; ca_dest.parent.mkdir(parents=True,exist_ok=True)
            ca_local=ROOT/'market_data/corporate_actions'/f'{symbol}.parquet'
            ca_source=None
            if ca_local.exists():
                shutil.copy2(ca_local,ca_dest); ca_source='local'
            else:
                try:
                    key=f'corporate_actions/{symbol}.parquet'
                    if r2.head_object(key): r2.get_object(key,ca_dest); ca_source='r2'
                except Exception as e:
                    audit.append({'tier':'r2_corporate_actions','error_type':type(e).__name__})
                if not ca_dest.exists():
                    time.sleep(1.2)
                    ret,ca=context().get_rehab('US.'+symbol)
                    if ret==ft.RET_OK:
                        ca.to_parquet(ca_dest,index=False,compression='zstd',compression_level=7); ca_source='futu'
            result={'symbol':symbol,'source':source,'pagination_complete':complete,'failure':failure,
                    'seconds':time.monotonic()-started,'quality':quality(f) if f is not None else None,
                    'bar_sha256':sha(dest) if dest.exists() else None,'corporate_actions_source':ca_source,
                    'corporate_actions_sha256':sha(ca_dest) if ca_dest.exists() else None}
            write(output/'audit'/f'{symbol}.json',audit); results.append(result)
            write(output/'progress.json',results)
            print(json.dumps(result,ensure_ascii=False),flush=True)
        write(output/'summary.json',results)
    finally:
        if quote:
            ret,q=quote.get_history_kl_quota(get_detail=False)
            if ret==ft.RET_OK: write(output/'quota_after.json',{'used':int(q[0]),'remaining':int(q[1])})
            quote.close()
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    acquire(parser.parse_args().output.resolve())
