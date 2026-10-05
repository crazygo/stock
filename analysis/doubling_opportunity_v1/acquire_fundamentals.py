#!/usr/bin/env python3
"""Resumable, rate-limited primary SEC facts acquisition for the full issuer pool."""
import argparse,json,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from data import CACHE,get_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);args=p.parse_args()
u=json.loads((CACHE/'universe.json').read_text())['stocks']
issuers={}
for t,m in u.items():
 if 'cik' in m:issuers.setdefault(m['cik'],[]).append(t)
items=list(issuers.items())
if args.limit:items=items[:args.limit]
lock=threading.Lock();last=[0.];results=[]
status=CACHE/'fundamentals_acquisition.json'

def fetch(item):
 cik,tickers=item;path=CACHE/'companyfacts'/f'CIK{cik:010d}.json'
 if path.exists():
  try:
   j=json.loads(path.read_text())
   if int(j.get('cik',0))==cik and j.get('facts'):
    return {'cik':cik,'tickers':tickers,'status':'local','bytes':path.stat().st_size}
  except Exception:pass
 for attempt in range(3):
  with lock:
   delay=.4-(time.monotonic()-last[0])
   if delay>0:time.sleep(delay)
   last[0]=time.monotonic()
  try:
   incoming=CACHE/'companyfacts_incoming'/path.name
   j=get_json(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',incoming,True)
   if int(j.get('cik',0))!=cik or not j.get('facts'):raise ValueError('CIK or facts mismatch')
   path.parent.mkdir(parents=True,exist_ok=True)
   temp=path.with_suffix('.valid.tmp');temp.write_bytes(incoming.read_bytes());temp.replace(path)
   return {'cik':cik,'tickers':tickers,'status':'downloaded','bytes':path.stat().st_size,'received_at':datetime.now(timezone.utc).isoformat()}
  except Exception as e:
   if attempt==2:return {'cik':cik,'tickers':tickers,'status':'unavailable','error_type':type(e).__name__}
   time.sleep(1+attempt)

def save(terminal=False):
 payload={'started_at':start,'updated_at':datetime.now(timezone.utc).isoformat(),'terminal':terminal,
          'registered_issuers':len(items),'completed':len(results),'available':sum(x['status']!='unavailable' for x in results),
          'unavailable':sum(x['status']=='unavailable' for x in results),'results':results}
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(status)
start=datetime.now(timezone.utc).isoformat();save()
with ThreadPoolExecutor(max_workers=max(1,min(3,args.workers))) as pool:
 futures=[pool.submit(fetch,i) for i in items]
 for f in as_completed(futures):
  results.append(f.result())
  if len(results)%20==0 or len(results)==len(items):
   save();print('SEC facts',len(results),'/',len(items),'unavailable',sum(x['status']=='unavailable' for x in results),flush=True)
save(True)
