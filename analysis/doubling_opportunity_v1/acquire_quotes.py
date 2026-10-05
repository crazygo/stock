#!/usr/bin/env python3
"""Whole-pool read-only market snapshots, with time and field lineage retained."""
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import futu as ft
from data import CACHE,universe

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true');ap.add_argument('--refresh-universe',action='store_true');args=ap.parse_args()
u=universe(True)[0] if args.refresh_universe else json.loads((CACHE/'universe.json').read_text())['stocks'];p=CACHE/'latest_market_snapshots.json'
if p.exists() and not args.refresh:
 print(json.dumps({k:v for k,v in json.loads(p.read_text()).items() if k not in ['stocks','unavailable']},indent=2));sys.exit(0)
ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
rows={};failed={};calls=0;last=0.

def fetch(codes):
 global calls,last
 delay=1.05-(time.monotonic()-last)
 if delay>0:time.sleep(delay)
 last=time.monotonic();calls+=1
 ret,data=q.get_market_snapshot(codes)
 if ret==ft.RET_OK:
  for r in json.loads(data.to_json(orient='records')):
   t=r['code'].removeprefix('US.');rows[t]=r
  returned=set(rows)
  for c in codes:
   t=c.removeprefix('US.')
   if t not in returned:failed[t]={'reason':'not_returned'}
  return
 msg=str(data)
 if len(codes)>1 and not any(s in msg.lower() for s in ['permission','频率','频控','权限','limit']):
  middle=len(codes)//2;fetch(codes[:middle]);fetch(codes[middle:]);return
 for c in codes:failed[c.removeprefix('US.')]={'reason':'provider_unavailable','error':msg[:400]}

start=datetime.now(timezone.utc).isoformat()
try:
 stocks=list(u)
 for i in range(0,len(stocks),400):
  fetch(['US.'+t for t in stocks[i:i+400]])
  print('snapshots',min(i+400,len(stocks)),'/',len(stocks),'returned',len(rows),'unavailable',len(failed),flush=True)
finally:q.close()
payload={'source':'Futu OpenD get_market_snapshot','requested_at':start,'received_at':datetime.now(timezone.utc).isoformat(),
         'requested_count':len(u),'returned_count':len(rows),'unavailable_count':len(failed),'calls':calls,'stocks':rows,'unavailable':failed,
         'quote_timestamp_zone':'America/New_York for US update_time','historical_probability_refresh':False,
         'warning':'Latest quotes do not supply the missing intermediate daily bars; old-anchor model probabilities remain old-anchor estimates.'}
p.write_text(json.dumps(payload,ensure_ascii=False))
print(json.dumps({k:v for k,v in payload.items() if k not in ['stocks','unavailable']},indent=2))
