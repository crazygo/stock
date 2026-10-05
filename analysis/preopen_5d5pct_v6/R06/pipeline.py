"""Finish the finite requested audit/training, without an unfinished protocol."""
import sys,json,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from common import write,now

def run(script,args=[]):
    print(json.dumps(dict(at=now(),stage=script)),flush=True)
    with (P/'logs'/(Path(script).stem+'_pipeline.log')).open('a') as stream:subprocess.run([sys.executable,'-u',str(P/script)]+args,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    started=time.monotonic()
    while json.loads((P/'acquisition.json').read_text())['status']!='completed':
        if time.monotonic()-started>5400:raise RuntimeError('First serial acquisition exceeded bounded 90-minute budget')
        time.sleep(10)
    # Resume the handful of mixed-N/A serialization failures once, retaining
    # their original failure logs. Genuine unavailable responses remain missing.
    run('collect_history.py');run('quality.py');run('features.py');run('training_coverage.py');run('train.py',['--fit','--workers','2']);run('finish.py')

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(P/'pipeline_status.json',dict(at=now(),status='failed',error=repr(e),independent_pass=False));raise
