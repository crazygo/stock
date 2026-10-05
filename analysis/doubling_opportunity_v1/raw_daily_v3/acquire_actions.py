#!/usr/bin/env python3
"""Download only actual registered public Nasdaq corporate-action article links."""
from __future__ import annotations
import concurrent.futures,fcntl,hashlib,json,sys,threading,time,urllib.error,urllib.request
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT));from data import CACHE

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def main():
    root=CACHE/'corporate_actions_v3';root.mkdir(exist_ok=True);lock=(root/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=HERE/'NASDAQ_ARCHIVE_REGISTRATION.json';registration=json.loads(source.read_text());items=[]
    for year in registration['years']:
        ids=[('eca' if n in year['lowercase'] else 'ECA')+str(year['year'])+'-'+str(n) for a,b in year['ranges'] for n in range(a,b+1)]+year['other']
        if len(ids)!=year['count']:raise ValueError('Visible archive registration count mismatch')
        items.extend({'archive_year':year['year'],'id':id,'url':registration['article_template'].format(id=id)} for id in ids)
    registration['articles']=items;registration['registered_count']=len(items);registration['registration_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    registration['protocol_sha256']=hashlib.sha256((HERE/'ACTION_AUDIT_PROTOCOL.md').read_bytes()).hexdigest();save(root/'registration.json',registration)
    (root/'html').mkdir(exist_ok=True);status=root/'acquisition.json';state=json.loads(status.read_text()) if status.exists() else {'results':{},'started_at':datetime.now(timezone.utc).isoformat()}
    limiter=threading.Lock();last=[0.]
    def fetch(item):
        key=hashlib.sha256(item['url'].encode()).hexdigest()[:16];target=root/'html'/(key+'.html');old=state['results'].get(item['url'],{})
        if target.exists() and old.get('sha256')==hashlib.sha256(target.read_bytes()).hexdigest():return old
        last_error='unknown'
        for attempt in range(3):
            try:
                with limiter:
                    wait=.6-(time.monotonic()-last[0])
                    if wait>0:time.sleep(wait)
                    last[0]=time.monotonic()
                request=urllib.request.Request(item['url'],headers={'User-Agent':'stock-research corporate-actions public-source audit','Accept':'text/html'})
                with urllib.request.urlopen(request,timeout=30) as r:raw=r.read()
                if b'Equity Corporate Actions Alert' not in raw:raise ValueError('Unexpected public article document')
                temp=target.with_suffix('.incoming');temp.write_bytes(raw);temp.replace(target)
                return {**item,'status':'available','path':str(target.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'received_at':datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                last_error=type(exc).__name__
                if isinstance(exc,urllib.error.HTTPError) and exc.code in [403,404]:break
                if attempt<2:time.sleep(2+attempt)
        return {**item,'status':'unavailable','error_type':last_error,'received_at':datetime.now(timezone.utc).isoformat()}
    def checkpoint(terminal=False):
        state.update(updated_at=datetime.now(timezone.utc).isoformat(),registered_count=len(items),completed=len(state['results']),
                     available=sum(r['status']=='available' for r in state['results'].values()),terminal=terminal,source='actual registered Nasdaq public corporate-action articles')
        save(status,state);save(CACHE/'corporate_actions_v3_acquisition.json',{k:v for k,v in state.items() if k!='results'})
    checkpoint()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(fetch,items):
            state['results'][result['url']]=result
            if len(state['results'])%20==0:checkpoint();print('registered action articles',state['completed'],'/',len(items),'available',state['available'],flush=True)
    checkpoint(True);print(json.dumps({k:v for k,v in state.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
