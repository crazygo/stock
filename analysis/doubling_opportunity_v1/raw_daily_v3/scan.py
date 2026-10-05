#!/usr/bin/env python3
"""Repeat whole-pool v3 acquisition, unchanged-model reuse and one report."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT));from data import CACHE

def run(path,*args):subprocess.run([sys.executable,str(path),*args],check=True)

def ensure_job(script,lock_path,state_path):
    """Continue a source only if no live process owns it; a terminal source is reused."""
    deadline=time.monotonic()+240*60
    while True:
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('terminal'):return
        lock=lock_path.open('a')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            if time.monotonic()>=deadline:raise TimeoutError('Existing registered source job still incomplete: '+script.name)
            print('waiting existing registered source',script.name,state.get('completed'),state.get('registered_count'),flush=True);time.sleep(45);continue
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();run(script);return

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh',action='store_true')
    p.add_argument('--top',type=int,default=30);p.add_argument('--debug',action='store_true')
    p.add_argument('--status',action='store_true',help='Inspect actual acquisition/training state without starting another job.')
    args=p.parse_args()
    if args.top<1:p.error('--top must be positive')
    if args.status:
        for name in ['futu_daily_v3_acquisition','corporate_actions_v3_acquisition','futu_daily_v3_recovery_status','raw_daily_v3_action_audit_status','raw_daily_v3_training_status']:
            q=CACHE/(name+'.json');print(json.dumps({name:json.loads(q.read_text()) if q.exists() else {'status':'not_started'}},indent=2))
        return
    # A live registered job owns its source snapshot. Wait instead of starting a duplicate.
    state=CACHE/'futu_daily_v3_acquisition.json';deadline=time.monotonic()+240*60
    while state.exists() and not (j:=json.loads(state.read_text())).get('terminal'):
        if j.get('error'):raise RuntimeError('Existing registered source job failed; inspect --status and its acquisition log')
        if time.monotonic()>=deadline:raise TimeoutError('Registered source wait expired; no subset report was substituted')
        print('existing source job',j.get('completed'),'/',j.get('registered_count'),flush=True);time.sleep(45)
    if args.refresh:run(PARENT/'acquire_quotes.py','--refresh','--refresh-universe')
    from acquire import closed_session
    uj=json.loads((CACHE/'universe.json').read_text());members=sorted(uj['stocks'])+['SPY']
    cohort=hashlib.sha256(json.dumps([members,uj['metadata']['raw_sha256']],sort_keys=True).encode()).hexdigest()
    current=json.loads(state.read_text()) if state.exists() else {}
    if args.refresh or not (current.get('terminal') and current.get('asof')==closed_session() and current.get('cohort_sha256')==cohort):
        run(HERE/'acquire.py',*(['--refresh'] if args.refresh else []))
    ensure_job(HERE/'acquire_actions.py',CACHE/'corporate_actions_v3/lock',CACHE/'corporate_actions_v3/acquisition.json')
    run(HERE/'acquire_action_corrections.py')
    base=Path(json.loads((CACHE/'futu_daily_v3_latest.json').read_text())['path'])
    ensure_job(HERE/'recover_sources.py',CACHE/'futu_daily_v3_recovery.lock',base/'recovery/acquisition.json')
    audit_state=CACHE/'raw_daily_v3_action_audit_status.json'
    # Wait for a live auditor; otherwise validate or recompute this exact source snapshot.
    lock=(CACHE/'raw_daily_v3_action_audit.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();deadline=time.monotonic()+240*60
        while True:
            ready_file=HERE/'ACTION_AUDIT_READY.json';ready=json.loads(ready_file.read_text()) if ready_file.exists() else {}
            if ready.get('ready_for_first_training') and ready.get('source_snapshot_path')==str(base.resolve()):break
            if time.monotonic()>deadline:raise TimeoutError('Action audit still incomplete')
            print('waiting existing action auditor',flush=True);time.sleep(45)
    else:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();run(HERE/'prepare_actions.py')
    if args.debug:run(HERE/'compare_native_minutes.py')
    run(HERE/'train.py','--reuse-if-unchanged')
    if args.top!=30:
        run(HERE/'run.py','--top',str(args.top))
        run(HERE/'verify.py')
    if args.debug:run(HERE/'build_debug.py')

if __name__=='__main__':main()
