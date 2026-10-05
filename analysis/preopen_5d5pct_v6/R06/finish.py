"""Complete evidence and close the finite R06 matrix; no outer retuning."""
import sys,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from common import OUT,write,now

def main():
    metas=[json.loads(p.read_text()) for p in (P/'cache/runs').glob('*/meta.json')]
    assert len(metas)==40 and all(m['status']=='completed_development' for m in metas),'All 40 actual models must finish'
    steps=[(P.parent/'calibration_audit.py',['--round','R06']),(P.parent/'favorite_report.py',['--round','R06']),(P.parent/'verify.py',['--round','R06']),(P/'verify_control.py',[]),(P/'export.py',[]),(P/'stability.py',[]),(P/'audit.py',[]),(P/'review.py',[])]
    for script,args in steps:
        print(json.dumps(dict(at=now(),stage=script.name)),flush=True)
        with (P/'logs'/(script.stem+'_finish.log')).open('a') as stream:subprocess.run([sys.executable,'-u',str(script)]+args,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(P/'pipeline_status.json',dict(at=now(),status='completed',models=40,scored_variants=80,independent_pass=False,dispositions_closed=True))

if __name__=='__main__':main()
