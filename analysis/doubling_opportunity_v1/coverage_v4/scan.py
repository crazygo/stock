#!/usr/bin/env python3
"""One whole-pool run: complete source cohorts, fixed evaluation, report and debug."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from context import CACHE,COVERAGE_ROOT,HERE
from acquire_daily import closed_session

def state(p):return json.loads(p.read_text()) if p.exists() else {}
def run(script,env,*args):subprocess.run([sys.executable,str(HERE/script),*args],env=env,check=True)
def ensure(script,lock_path,state_path,env):
    deadline=time.monotonic()+240*60
    while True:
        s=state(state_path)
        if s.get('terminal') or s.get('status')=='complete_source_audit':return
        lock_path.parent.mkdir(parents=True,exist_ok=True);lock=lock_path.open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            if time.monotonic()>deadline:raise TimeoutError('Existing full source still incomplete: '+script)
            print('waiting registered source',script,s.get('completed'),s.get('registered_count',s.get('registered_issuers')),flush=True)
            time.sleep(45);continue
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();run(script,env);return

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--top',type=int,default=30)
    p.add_argument('--debug',action='store_true');p.add_argument('--refresh',action='store_true');p.add_argument('--status',action='store_true');args=p.parse_args()
    if args.top<1:p.error('--top must be positive')
    if args.status:
        for name in ['futu_daily_v4_acquisition','fundamentals_acquisition','submissions_acquisition','futu_daily_v4_recovery_status','coverage_v4_action_audit_status','coverage_v4_training_status']:
            j=state(CACHE/(name+'.json'));j.pop('results',None);print(json.dumps({name:j or {'status':'not_started'}},indent=2))
        print(json.dumps({'active_source_cache':str(CACHE),'latest_report':state(HERE/'latest_run.json')},indent=2));return
    COVERAGE_ROOT.mkdir(parents=True,exist_ok=True)
    full_lock=(COVERAGE_ROOT/'scan.lock').open('a');fcntl.flock(full_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    asof=closed_session();base=CACHE;env=dict(os.environ)
    current=state(CACHE/'futu_daily_v4_acquisition.json')
    if args.refresh or current and current.get('asof')!=asof:
        # Finish any live immutable cohort before switching the active source.
        # Old results remain available, but never supply a new day's SEC snapshot.
        for script,lock,state_name in [('acquire_daily.py','futu_daily_v4_acquisition.lock','futu_daily_v4_acquisition.json'),
             ('acquire_sec.py','sec_acquisition.lock','submissions_acquisition.json')]:
            if state(CACHE/state_name) and not state(CACHE/state_name).get('terminal'):
                ensure(script,CACHE/lock,CACHE/state_name,env)
        base=COVERAGE_ROOT/'source_cohorts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        base.mkdir(parents=True);env['DOUBLING_COVERAGE_CACHE']=str(base.resolve());env['DOUBLING_SEC_FRESH']='1'
        run('catalog.py',env,'--refresh-directory')
    else:
        env['DOUBLING_COVERAGE_CACHE']=str(base.resolve())
        if not (base/'universe.json').exists():run('catalog.py',env,'--refresh-directory')
    run('verify_catalog.py',env)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(ensure,'acquire_daily.py',base/'futu_daily_v4_acquisition.lock',base/'futu_daily_v4_acquisition.json',env),
              pool.submit(ensure,'acquire_sec.py',base/'sec_acquisition.lock',base/'submissions_acquisition.json',env)]
        for job in jobs:job.result()
    source=Path(state(base/'futu_daily_v4_latest.json')['path'])
    ensure('recover_sources.py',base/'futu_daily_v4_recovery.lock',source/'recovery/acquisition.json',env)
    # An auditor can be live even after its source has completed.
    ensure('prepare_actions.py',base/'coverage_v4_action_audit.lock',base/'coverage_v4_action_audit_status.json',env)
    run('prepare_actions.py',env)
    if args.debug:run('compare_native_minutes.py',env)
    run('train.py',env,'--reuse-if-unchanged')
    if args.top!=30:run('run.py',env,'--top',str(args.top));run('verify.py',env)
    if args.debug:run('build_debug.py',env)
    active=COVERAGE_ROOT/'active_source_cache.json';tmp=active.with_suffix('.tmp')
    tmp.write_text(json.dumps({'path':str(base.resolve()),'asof':asof},indent=2));tmp.replace(active)

if __name__=='__main__':main()
