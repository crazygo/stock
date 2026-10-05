#!/usr/bin/env python3
"""Full registered issuer coverage, isolated cache, one task-wide SEC rate limit."""
from __future__ import annotations
import fcntl,hashlib,json,os,shutil,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parent))
from context import CACHE,OLD_CACHE,ROOT,HERE

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2));tmp.replace(p)
def validate(raw,cik,kind):
    j=json.loads(raw)
    if int(j.get('cik',0))!=cik:raise ValueError('CIK identity mismatch')
    if kind=='companyfacts' and not isinstance(j.get('facts'),dict):raise ValueError('Missing facts structure')
    if kind=='submissions' and not isinstance(j.get('tickers'),list):raise ValueError('Missing issuer ticker structure')
    return j

def main():
    CACHE.mkdir(parents=True,exist_ok=True);lockfile=(CACHE/'sec_acquisition.lock').open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    up=CACHE/'universe.json';u=json.loads(up.read_text());issuers={}
    for t,m in u['stocks'].items():
        if m.get('cik'):issuers.setdefault(m['cik'],[]).append(t)
    registration_path=CACHE/'sec_registration.json'
    current_code_sha256=sha(Path(__file__))
    registration={'catalog_sha256':sha(up),'catalog_key':json.loads((CACHE/'catalog_latest.json').read_text())['key'],
        'issuers':{str(c):sorted(ts) for c,ts in sorted(issuers.items())},'protocol_sha256':sha(HERE/'PROTOCOL.md'),
        'code_sha256':sha(Path(__file__)),'kinds':['companyfacts','submissions'],'maximum_workers':3,'task_request_interval_seconds':.4}
    if registration_path.exists():
        old=json.loads(registration_path.read_text());old.pop('registered_at')
        # Resume the same issuer/protocol registration; preserve its original
        # implementation and tag any new retrievals with the actual resumed code.
        registration['code_sha256']=old['code_sha256']
        if old!=registration:raise ValueError('Registered source inputs changed; create a new namespace')
    else:save(registration_path,registration|{'registered_at':stamp()})
    snapshot=CACHE/'registered_code/acquire_sec.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():
        if current_code_sha256!=registration['code_sha256']:raise ValueError('Original registered code snapshot missing')
        snapshot.write_bytes(Path(__file__).read_bytes())
    if sha(snapshot)!=registration['code_sha256']:raise ValueError('Registered SEC implementation snapshot mismatch')
    actual_code=CACHE/'registered_code'/f'acquire_sec_{current_code_sha256}.py'
    if not actual_code.exists():actual_code.write_bytes(Path(__file__).read_bytes())
    contact=subprocess.run(['git','config','user.email'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    ua=os.environ.get('SEC_USER_AGENT') or f'StockDoublingResearch/1.0 {contact}'
    rate_lock=threading.Lock();last=[0.];started=stamp();results={'companyfacts':{},'submissions':{}}
    for kind,name in [('companyfacts','fundamentals'),('submissions','submissions')]:
        state_path=CACHE/(name+'_acquisition.json')
        if state_path.exists():
            prior=json.loads(state_path.read_text())
            if prior.get('registration_sha256')!=sha(registration_path):raise ValueError('Resume registration mismatch')
            results[kind]={int(c):v for c,v in prior.get('results',{}).items()}
    def checkpoint(terminal=False):
        for kind,name in [('companyfacts','fundamentals'),('submissions','submissions')]:
            rs=results[kind]
            save(CACHE/(name+'_acquisition.json'),{'started_at':started,'updated_at':stamp(),'terminal':terminal,
                'full_registered_pool':True,'registered_issuers':len(issuers),'completed':len(rs),
                'available':sum(x['status']=='available' for x in rs.values()),'unavailable':sum(x['status']!='available' for x in rs.values()),
                'registration_sha256':sha(registration_path),'results':rs})
    def fetch(cik,kind):
        p=CACHE/kind/f'CIK{cik:010d}.json';origin=OLD_CACHE/kind/p.name
        url=f'https://data.sec.gov/'+(f'api/xbrl/companyfacts/{p.name}' if kind=='companyfacts' else f'submissions/{p.name}')
        record={'cik':cik,'registered_tickers':issuers[cik],'source_url':url,
                'actual_acquisition_code_sha256':current_code_sha256,'actual_acquisition_code_path':str(actual_code.resolve())}
        # New-day/new-directory cohorts refresh filings instead of silently
        # reusing the old experiment's financial snapshot.
        origins=[p] if os.environ.get('DOUBLING_SEC_FRESH')=='1' else [p,origin]
        for candidate in origins:
            if not candidate.exists():continue
            try:
                raw=candidate.read_bytes();j=validate(raw,cik,kind)
                if candidate!=p:
                    p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,p)
                record.update(status='available',tier='Local',path=str(p.resolve()),sha256=sha(p),bytes=len(raw),
                    original_path=str(candidate.resolve()),original_sha256=hashlib.sha256(raw).hexdigest(),
                    copied_or_verified_at=stamp(),original_received_at_unknown=True)
                break
            except (ValueError,TypeError,json.JSONDecodeError):continue
        else:
            for attempt in range(3):
                with rate_lock:
                    wait=.4-(time.monotonic()-last[0])
                    if wait>0:time.sleep(wait)
                    last[0]=time.monotonic()
                try:
                    with urlopen(Request(url,headers={'User-Agent':ua}),timeout=40) as response:raw=response.read()
                    j=validate(raw,cik,kind);p.parent.mkdir(parents=True,exist_ok=True)
                    tmp=p.with_suffix('.incoming');tmp.write_bytes(raw);tmp.replace(p)
                    record.update(status='available',tier='SEC',path=str(p.resolve()),sha256=sha(p),bytes=len(raw),received_at=stamp());break
                except Exception as exc:
                    code=exc.code if isinstance(exc,HTTPError) else None
                    if attempt==2 or code in [400,403,404]:
                        return record|{'status':'unavailable','error_type':type(exc).__name__,'http_status':code,'received_at':stamp(),'attempts':attempt+1}
                    time.sleep(2*(attempt+1))
        if kind=='submissions':
            tickers=sorted(t.replace('-','.') for t in j['tickers'])
            record.update(current_sec_tickers=tickers,registered_tickers_confirmed=sorted(set(issuers[cik])&set(tickers)),
                          sic=j.get('sic'),sic_description=j.get('sicDescription'),entity_name=j.get('name'),membership_mode='current_not_PIT')
        return record
    checkpoint()
    tasks=[(c,k) for c in sorted(issuers) for k in results if c not in results[k]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fetch,c,k):(c,k) for c,k in tasks}
        for number,f in enumerate(as_completed(pending),1):
            c,k=pending[f];results[k][c]=f.result()
            if number%30==0 or number==len(tasks):
                checkpoint();print('registered SEC',sum(map(len,results.values())),'/',2*len(issuers),
                    'available',sum(x['status']=='available' for rs in results.values() for x in rs.values()),flush=True)
    if any(set(rs)!=set(issuers) for rs in results.values()):raise ValueError('Incomplete registered issuer outcomes')
    checkpoint(True);lockfile.close();print('registered SEC complete',len(issuers),'issuers',flush=True)

if __name__=='__main__':main()
