"""Finalize the finite R05 matrix; incomplete jobs cannot masquerade as done."""
import json,subprocess,sys
from common import OUT,write,now

def main():
    metas=[json.loads(p.read_text()) for p in (OUT/'R05/cache/runs').glob('*/meta.json')]
    assert len(metas)==10 and all(m['status']=='completed_development' for m in metas)
    steps=[('calibration_audit.py',['--round','R05'],'R05_calibration.log'),('favorite_report.py',[],'R05_favorites.log'),
        ('verify.py',['--round','R05'],'R05_verify.log'),('verify_r05_control.py',[],'R05_control.log'),
        ('export_models.py',[],'export_models_R05.log'),('input_stability_audit.py',[],'input_stability_final.log')]
    for script,args,log in steps:
        print(json.dumps(dict(at=now(),stage=script)),flush=True)
        with (OUT/'logs'/log).open('a') as stream:subprocess.run([sys.executable,'-u',str(OUT/script)]+args,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(OUT/'R05/pipeline_status.json',dict(at=now(),status='completed',independent_pass=False))
if __name__=='__main__':main()
