"""Resumable whole-watchlist/QQQ minute preparation, Local -> R2 -> OpenD."""
import argparse,fcntl,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from common import OUT,OLD,ROOT,write,save,sha,now,journal

def normalize_futu(frame):
    end=pd.to_datetime(frame.time_key)
    f=pd.DataFrame({'start':end-pd.Timedelta(minutes=5),'end':end,'available':end+pd.Timedelta(seconds=1)})
    for c in ['open','high','low','close','volume']:f[c]=pd.to_numeric(frame[c],errors='raise').to_numpy()
    f['day']=f.start.dt.strftime('%Y-%m-%d');f['minute']=f.start.dt.hour*60+f.start.dt.minute
    return f.drop_duplicates('start').sort_values('start').reset_index(drop=True)

def normalize_archive(f):
    if {'start','end','available','day','minute'}.issubset(f):return f
    if not {'start_at_et','end_at_et','price_basis'}.issubset(f) or set(f.price_basis.dropna())!={'NONE'}:raise ValueError('archive not explicit NONE')
    n=f[['open','high','low','close','volume']].copy()
    for dst,src in [('start','start_at_et'),('end','end_at_et')]:n[dst]=pd.to_datetime(f[src],utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    n['available']=n.end+pd.Timedelta(seconds=1);n['day']=n.start.dt.strftime('%Y-%m-%d');n['minute']=n.start.dt.hour*60+n.start.dt.minute
    return n.drop_duplicates('start').sort_values('start').reset_index(drop=True)

def universe():
    old=json.loads((OLD/'universe.json').read_text());watch=json.loads((OUT/'watchlist_snapshot.json').read_text())
    rows={s:dict(symbol=s,role='ETF_member',groups=m.get('groups',[])) for s,m in old['members'].items()}
    for snapshot in watch['snapshots']:
        if not isinstance(snapshot['members'],list):continue
        for r in snapshot['members']:
            code=r['code']
            if not code.startswith('US.') or r['stock_type'] not in ['STOCK','ETF']:continue
            s=code[3:];item=rows.setdefault(s,dict(symbol=s,role=r['stock_type'],groups=[]))
            item['stock_type']=r['stock_type'];item['listing_date']=r.get('listing_date')
            if snapshot['group'] not in item['groups']:item['groups'].append(snapshot['group'])
    def priority(r):
        g=r['groups'];return (0 if '特别关注' in g else 1 if '全部' in g else 2 if 'QQQ' in g else 3,r['symbol'])
    rows=sorted(rows.values(),key=priority)
    write(OUT/'universe.json',dict(observed_at=now(),members=rows,membership='current_snapshot_retrospective_not_PIT',legacy_funds=old['funds']))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=120);ap.add_argument('--start',default='2024-10-04');ap.add_argument('--end',default='2026-10-02');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    rows=universe(); inventory=json.loads((OUT/'cache/r2_inventory.json').read_text());objects=[]
    for source in inventory['sources']:objects+=source.get('objects',[])
    from scripts.r2_client import R2Client
    import futu as ft
    client=R2Client();ft.SysConfig.enable_proto_encrypt(False)
    q=None;history_requests=0;report=[]
    lock=(OUT/'cache/acquire.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX if a.wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for r in rows:
            symbol=r['symbol'];meta=OUT/'cache/acquired'/f'{symbol}.json';path=meta.with_suffix('.parquet')
            if meta.exists():report.append(json.loads(meta.read_text()));continue
            known=OLD/'raw'/f'{symbol}.parquet'
            if known.exists():
                f=pd.read_parquet(known,columns=['start','end']);item=dict(**r,status='local_available',source=str(known.relative_to(ROOT)),first=str(f.start.min()),last=str(f.end.max()),bars=len(f),requested=[a.start,a.end],received_at=None)
                write(meta,item);report.append(item);continue
            frames=[];sources=[]
            for obj in objects:
                key=obj['key']
                if not key.startswith(f'us_5m/{symbol}/') or not key.endswith('.parquet'):continue
                local=ROOT/'market_data'/key
                if not local.exists():
                    local=OUT/'cache/r2'/key;local.parent.mkdir(parents=True,exist_ok=True)
                    try:client.get_object(key,local)
                    except Exception as e:sources.append('R2_unavailable:'+key+':'+type(e).__name__);continue
                try:frames.append(normalize_archive(pd.read_parquet(local)));sources.append('R2:'+key)
                except ValueError:pass
            f=pd.concat(frames,ignore_index=True).drop_duplicates('start').sort_values('start') if frames else pd.DataFrame()
            if len(f) and str(f.end.max())[:10]>=a.end:
                save(f,path);item=dict(**r,status='r2_available',sources=sources,first=str(f.start.min()),last=str(f.end.max()),bars=len(f),requested=[a.start,a.end],received_at=now());write(meta,item);report.append(item);continue
            if history_requests>=a.limit:report.append(dict(**r,status='queued_free_history',reason='per-run conservative new-symbol cap'));continue
            if q is None:q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
            pages=[];page=None;failed=None;pagecount=0
            # Each new symbol consumes existing history quota; no paid quota purchase.
            history_requests+=1
            start=a.start
            if len(f):start=max(a.start,str(f.end.max())[:10])
            while True:
                ret,bars,page=q.request_history_kline('US.'+symbol,start=start,end=a.end,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,page_req_key=page,extended_time=True,session=ft.Session.ALL)
                if ret!=ft.RET_OK:failed=str(bars)[:180];break
                pages.append(normalize_futu(bars));pagecount+=1
                if pagecount%20==0:print(json.dumps(dict(symbol=symbol,pages=pagecount)),flush=True)
                if not page:break
                time.sleep(.4)
            if pages:
                merged=pd.concat(([f] if len(f) else [])+pages,ignore_index=True).drop_duplicates('start',keep='last').sort_values('start');save(merged,path)
                item=dict(**r,status='futu_available' if failed is None else 'partial',sources=sources+['OpenD_ALL_NONE_5m'],first=str(merged.start.min()),last=str(merged.end.max()),bars=len(merged),pages=pagecount,requested=[a.start,a.end],received_at=now(),sha256=sha(path),error=failed)
            else:item=dict(**r,status='unavailable',requested=[a.start,a.end],received_at=now(),error=failed)
            write(meta,item);report.append(item);journal('new_minute_history',item);print(json.dumps(dict(symbol=symbol,status=item['status'],pages=pagecount)),flush=True)
            time.sleep(1.2)
            write(OUT/'history_acquisition.json',dict(at=now(),requested=[a.start,a.end],records=report,complete=False))
            if failed and ('配额' in failed or 'quota' in failed.lower()):break
    finally:
        if q is not None:q.close()
        lock.close()
    seen={r['symbol'] for r in report};report += [dict(**r,status='queued_free_history') for r in rows if r['symbol'] not in seen]
    write(OUT/'history_acquisition.json',dict(at=now(),requested=[a.start,a.end],records=report,complete=all(r['status'] in ['local_available','r2_available','futu_available'] for r in report)))
if __name__=='__main__':main()
