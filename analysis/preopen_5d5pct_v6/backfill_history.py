"""Priority preregistered history prefix for existing 124 stocks.

Runs sequentially after the whole-watchlist downloader releases its lock.
Existing v5 prices take precedence. No writes outside v6, no cloud upload.
"""
import argparse,fcntl,json,time
import numpy as np
import pandas as pd
from common import OUT,OLD,ROOT,write,save,sha,now
from acquire_history import normalize_archive,normalize_futu
FIVE=['ALAB','AMD','MRVL','TER','TXG']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',default='2024-10-04');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
 lock=(OUT/'cache/acquire.lock').open('a');print(json.dumps(dict(status='waiting_for_sequential_data_lock' if a.wait else 'acquiring_data_lock')),flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX if a.wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
 import futu as ft
 from scripts.r2_client import R2Client
 ft.SysConfig.enable_proto_encrypt(False);q=None;client=R2Client();report=[]
 inv=json.loads((OUT/'cache/r2_inventory.json').read_text());objects=[o for source in inv['sources'] for o in source.get('objects',[])]
 favorites={r['code'].removeprefix('US.') for snap in json.loads((OUT/'watchlist_snapshot.json').read_text())['snapshots'] if snap['group']=='特别关注' for r in snap['members'] if isinstance(snap['members'],list) and r['code'].startswith('US.')}
 paths=sorted((OLD/'raw').glob('*.parquet'),key=lambda p:(0 if p.stem in FIVE else 1 if p.stem in favorites else 2,p.stem))
 try:
  for original in paths:
   symbol=original.stem;dest=OUT/'cache/backfill'/f'{symbol}.parquet';meta=dest.with_suffix('.json')
   if meta.exists():report.append(json.loads(meta.read_text()));continue
   old=pd.read_parquet(original);first=old.start.min();end=str(first.date());sources=[];frames=[]
   for p in sorted((ROOT/'market_data/us_5m'/symbol).glob('*.parquet')):
    try:frames.append(normalize_archive(pd.read_parquet(p)));sources.append('Local:'+str(p.relative_to(ROOT)))
    except ValueError:continue
   for o in objects if str(first.date())>a.start else []:
    key=o['key']
    if not key.startswith(f'us_5m/{symbol}/') or not key.endswith('.parquet'):continue
    p=OUT/'cache/r2'/key
    if not p.exists():
     p.parent.mkdir(parents=True,exist_ok=True)
     try:client.get_object(key,p)
     except Exception as e:
      # A failed cloud read must not abort the resumable local/OpenD route.
      sources.append('R2_unavailable:'+key+':'+type(e).__name__)
      continue
    try:frames.append(normalize_archive(pd.read_parquet(p)));sources.append('R2:'+key)
    except ValueError:continue
   local=pd.concat(frames,ignore_index=True).drop_duplicates('start').sort_values('start') if frames else old.iloc[:0]
   local=local[local.start<first];failed=None;pagecount=0
   enough=len(local) and str(local.start.min().date())<=a.start
   if str(first.date())>a.start and not enough:
    if q is None:q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    page=None;parts=[]
    while True:
     ret,bars,page=q.request_history_kline('US.'+symbol,start=str((pd.Timestamp(a.start)-pd.Timedelta(days=1)).date()),end=end,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,page_req_key=page,extended_time=True,session=ft.Session.ALL)
     if ret!=ft.RET_OK:failed=str(bars)[:160];break
     parts.append(normalize_futu(bars));pagecount+=1
     if pagecount%30==0:print(json.dumps(dict(symbol=symbol,prefix_pages=pagecount)),flush=True)
     if not page:break
     time.sleep(.4)
    if parts:local=pd.concat([local]+parts,ignore_index=True).drop_duplicates('start',keep='last').sort_values('start');local=local[local.start<first];sources.append('OpenD_ALL_NONE_5m')
   # All existing timestamps and prices are immutable, even if a provider
   # re-query might return revised overlapping bars.
   merged=pd.concat([local,old],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start');save(merged,dest)
   bound=merged.merge(old,on='start',suffixes=('_new','_old'),validate='one_to_one')
   assert len(bound)==len(old)
   assert np.array_equal(bound[[c+'_new' for c in ['open','high','low','close','volume']]].to_numpy(),bound[[c+'_old' for c in ['open','high','low','close','volume']]].to_numpy(),equal_nan=True)
   item=dict(symbol=symbol,status='partial' if failed else 'acquired_not_quality_approved',requested=[a.start,'2026-09-30'],prefix_request_end=end,first=str(merged.start.min()),last=str(merged.end.max()),bars=len(merged),added_prefix_bars=len(local),pages=pagecount,sources=sources,source_old_sha256=sha(original),sha256=sha(dest),error=failed,received_at=now())
   write(meta,item);report.append(item);write(OUT/'prefix_acquisition.json',dict(at=now(),status='running',records=report));print(json.dumps(dict(symbol=symbol,prefix_bars=len(local),status=item['status'])),flush=True);time.sleep(1.2)
   if failed and ('配额'in failed or 'quota'in failed.lower()):break
 finally:
  if q is not None:q.close()
  lock.close()
 write(OUT/'prefix_acquisition.json',dict(at=now(),status='acquisition_finished_quality_pending',records=report,requested=['2024-10-04','2026-09-30']))
if __name__=='__main__':main()
