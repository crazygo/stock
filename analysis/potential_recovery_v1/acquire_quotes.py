#!/usr/bin/env python3
"""Bound one report's quote supplementation in a killable child; keep atomic snapshots."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from context import CACHE

def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(j,ensure_ascii=False,allow_nan=False));tmp.replace(p)

def worker(deadline,priority):
    import futu as ft
    u=json.loads((CACHE/'universe.json').read_text())['stocks'];priority=[t for t in dict.fromkeys(priority) if t in u]
    members=priority+[t for t in u if t not in set(priority)];rows={};failed={};calls=0;last=0.;q=None
    requested=stamp();p=CACHE/'latest_market_snapshots.json'
    def remaining():return max(0,deadline-time.monotonic())
    def fetch(codes):
        nonlocal calls,last
        if remaining()<=0:return
        delay=max(0,1.05-(time.monotonic()-last))
        if delay>=remaining():return
        if delay:time.sleep(delay)
        last=time.monotonic();calls+=1
        ret,data=q.get_market_snapshot(codes)
        if ret==ft.RET_OK:
            returned=set();requested_tickers={c.removeprefix('US.') for c in codes}
            for r in json.loads(data.to_json(orient='records')):
                t=r['code'].removeprefix('US.')
                if t in requested_tickers:rows[t]=r;returned.add(t)
            for t in requested_tickers-returned:failed[t]={'reason':'not_returned'}
            return
        msg=str(data)
        if len(codes)>1 and not any(s in msg.lower() for s in ['permission','频率','频控','权限','limit']):
            mid=len(codes)//2;fetch(codes[:mid]);fetch(codes[mid:]);return
        for c in codes:failed[c.removeprefix('US.')]={'reason':'provider_unavailable','error':msg[:250]}
    def checkpoint(terminal=False):
        missing={t:failed.get(t,{'reason':'not_requested_within_current_budget'}) for t in members if t not in rows}
        payload={'source':'Futu OpenD get_market_snapshot','requested_at':requested,'received_at':stamp(),
            'requested_count':len(members),'returned_count':len(rows),'unavailable_count':len(missing),'calls':calls,
            'stocks':rows,'unavailable':missing,'terminal':terminal,'partial_snapshot':len(rows)<len(members),
            'quote_priority_tickers':priority,'quote_priority_is_signal_or_model_selection':False,
            'quote_timestamp_zone':'America/New_York for US update_time','historical_probability_refresh':False,
            'warning':'Quote update_time is not verified per-session trade time; snapshot is a descriptive reference only.'}
        save(p,payload)
    try:
        ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
        for i in range(0,len(members),400):
            if remaining()<=0:break
            fetch(['US.'+t for t in members[i:i+400]]);checkpoint()
            print('quote stage',min(i+400,len(members)),'/',len(members),'returned',len(rows),'remaining_seconds',round(remaining(),1),flush=True)
        checkpoint(True)
    finally:
        if q is not None:q.close()

def supervise(budget,command=None,priority=()):
    """Public helper also permits a real hung-child verification without network."""
    started=time.monotonic();deadline=started+budget
    cmd=command or [sys.executable,str(Path(__file__).resolve()),'--worker','--deadline',str(deadline),'--priority-symbols',*priority]
    proc=subprocess.Popen(cmd,start_new_session=True)
    try:
        code=proc.wait(timeout=max(.001,deadline-time.monotonic()));return {'timed_out':False,'exit_code':code,'elapsed_seconds':time.monotonic()-started}
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
        return {'timed_out':True,'exit_code':proc.returncode,'elapsed_seconds':time.monotonic()-started}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true')
    ap.add_argument('--budget-seconds',type=float,default=30);ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    ap.add_argument('--priority-symbols',nargs='*',default=[],help='Quote report candidates first; all registered securities remain in coverage.')
    args=ap.parse_args()
    if args.budget_seconds<=0:ap.error('--budget-seconds must be positive')
    if args.worker:worker(args.deadline,args.priority_symbols);return
    p=CACHE/'latest_market_snapshots.json'
    if p.exists() and not args.refresh:
        j=json.loads(p.read_text());print(json.dumps({k:v for k,v in j.items() if k not in ['stocks','unavailable']},indent=2));return
    requested=stamp();before=sha(p) if p.exists() else None;old_snapshot=None
    if before:
        old_snapshot=CACHE/'quote_snapshots'/f'{before}.json';old_snapshot.parent.mkdir(exist_ok=True)
        if not old_snapshot.exists():old_snapshot.write_bytes(p.read_bytes())
    outcome=supervise(args.budget_seconds,priority=args.priority_symbols);after=sha(p) if p.exists() else None
    attempt={'requested_at':requested,'finished_at':stamp(),'budget_seconds':args.budget_seconds,**outcome,
        'prior_snapshot_sha256':before,'result_snapshot_sha256':after,'prior_complete_file_preserved':str(old_snapshot) if old_snapshot else None,
        'mode':'no_snapshot_missing' if after is None else 'supplementation_failed_or_timed_out_prior_cache_retained' if before==after else 'new_atomic_snapshot_available',
        'model_or_threshold_changed':False}
    save(CACHE/'latest_quote_refresh_attempt.json',attempt);print(json.dumps(attempt,indent=2),flush=True)
    # Missing/failed quotes do not prevent the frozen daily reference report.

if __name__=='__main__':main()
