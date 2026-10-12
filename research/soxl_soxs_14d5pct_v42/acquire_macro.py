"""Snapshot six daily NONE histories after local and both R2 namespace checks."""
from pathlib import Path
import sys,json,time,subprocess,datetime,hashlib
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'.cache/soxl_soxs_14d5pct_v42'
sys.path.insert(0,str(ROOT/'analysis/stock_traits_daily_v1'))
from sources import OpenDLayer

def acquire():
    quota=json.loads((C/'quota_preflight.json').read_text());assert quota['ok'] and quota['remain']>=12
    pre=json.loads((C/'r2_preflight.json').read_text());assert all('error_type' not in x for x in pre['r2'])
    assert not any(x['objects'] for x in pre['r2'] if '/us_day/' in '/'+x['prefix'])
    out=C/'data/capture';out.mkdir(parents=True,exist_ok=True);manifest=[]
    q=OpenDLayer('127.0.0.1',11111,15)
    for s in ['SPY','IWM','TLT','SHY','HYG','LQD']:
        daily=out/(s+'_daily_NONE.parquet');actions=out/(s+'_actions.parquet');assert not daily.exists() and not actions.exists()
        begin=datetime.datetime.now(datetime.timezone.utc).isoformat()
        result=q.fetch_day('US.'+s,'2024-09-01','2026-09-30',daily,autype='NONE')
        end=datetime.datetime.now(datetime.timezone.utc).isoformat();assert result.get('ok'),result
        time.sleep(3.2)
        cp=subprocess.run([sys.executable,str(Path(__file__).with_name('action_worker.py')),'--symbol',s,'--output',str(actions)],capture_output=True,text=True,timeout=30)
        assert cp.returncode==0,(s,cp.stderr[-300:]);raw=next(x.split('@@MACRO_ACTION@@ ',1)[1] for x in cp.stdout.splitlines() if '@@MACRO_ACTION@@ ' in x);action=json.loads(raw)
        manifest.append({'symbol':s,'daily':{'path':str(daily.relative_to(ROOT)),'sha256':hashlib.sha256(daily.read_bytes()).hexdigest(),'requested_at':begin,'received_at':end,'source':'OpenD K_DAY NONE regular','receipt':result},
                         'actions':{'path':str(actions.relative_to(ROOT)),'sha256':hashlib.sha256(actions.read_bytes()).hexdigest(),'source':'OpenD get_rehab','receipt':action}})
        (C/'data/acquisition.json').write_text(json.dumps({'captured_at':end,'symbols':manifest,'uploads':0,'partial':len(manifest)<6},indent=2)+'\n')
        print(s,'daily rows',result['rows'],'action rows',action['rows'],flush=True);time.sleep(3.2)
    after=q.quota_now();(C/'quota_after.json').write_text(json.dumps(after,indent=2)+'\n')

if __name__=='__main__':acquire()
