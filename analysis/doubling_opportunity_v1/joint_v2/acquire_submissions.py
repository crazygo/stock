#!/usr/bin/env python3
"""Current SEC metadata and primary filing dates; resumable, bounded download rate."""
import argparse,json,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
PARENT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PARENT))
from data import CACHE,get_json
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--limit',type=int);args=ap.parse_args()
u=json.loads((CACHE/'universe.json').read_text())['stocks'];ids={}
for t,m in u.items():
 if 'cik' in m:ids.setdefault(m['cik'],[]).append(t)
items=list(ids.items())[:args.limit];lock=threading.Lock();last=[0.];result=[]
start=datetime.now(timezone.utc).isoformat();output=CACHE/'submissions_acquisition.json'

def fetch(item):
 cik,registered=item;p=CACHE/'submissions'/f'CIK{cik:010d}.json';status='local' if p.exists() else 'downloaded'
 if not p.exists():
  with lock:
   delay=.4-(time.monotonic()-last[0])
   if delay>0:time.sleep(delay)
   last[0]=time.monotonic()
 try:
  j=get_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',p)
  if int(j.get('cik',0))!=cik:raise ValueError('CIK mismatch')
  live=set(t.replace('-','.') for t in j.get('tickers',[]))
  return {'cik':cik,'registered_tickers':registered,'status':status,'sic':j.get('sic'),'sic_description':j.get('sicDescription'),
          'entity_name':j.get('name'),'current_sec_tickers':sorted(live),'registered_tickers_confirmed':sorted(set(registered)&live),
          'membership_mode':'current_SEC_metadata_not_historical_PIT','received_at':datetime.now(timezone.utc).isoformat(),'bytes':p.stat().st_size}
 except Exception as e:return {'cik':cik,'registered_tickers':registered,'status':'unavailable','error_type':type(e).__name__}

def save(terminal=False):
 j={'started_at':start,'updated_at':datetime.now(timezone.utc).isoformat(),'terminal':terminal,'registered_issuers':len(items),
    'completed':len(result),'available':sum(x['status']!='unavailable' for x in result),'unavailable':sum(x['status']=='unavailable' for x in result),'results':result}
 temp=output.with_suffix('.tmp');temp.write_text(json.dumps(j));temp.replace(output)
save()
with ThreadPoolExecutor(max_workers=3) as pool:
 for f in as_completed([pool.submit(fetch,i) for i in items]):
  result.append(f.result())
  if len(result)%50==0 or len(result)==len(items):save();print('SEC submissions',len(result),'/',len(items),'missing',sum(x['status']=='unavailable' for x in result),flush=True)
save(True)
