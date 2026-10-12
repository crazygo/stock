"""Recompute statistics and execute the failed-primary guard before any case output."""
import argparse,subprocess,sys
from adaptive_penalty import *
from audit_statistics import audit_statistics

def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');reg=read(run/'registration.json')
    assert reg['phase']=='development' and v['passed'] and v['results_sha256']==sha(run/'results.json')
    n=audit_statistics(run);gate=development_gate(r['primary'],r['coverage']['complete'],reg['preregistration']['protocol'])
    assert r['development_gate']==gate and r['primary_name']=='adaptive_band' and r['primary']==r['routes']['adaptive_band']
    blocked=False
    if not gate['passed']:
        target=run.parent/'blocked_cases_guard_check';assert not target.exists()
        result=subprocess.run([sys.executable,str(SOURCE/'run_penalty.py'),'--output',str(target),'--case-after',str(run)],text=True,capture_output=True)
        assert result.returncode!=0 and 'Development primary failed; new case replay is forbidden' in result.stderr and not target.exists();blocked=True
    receipt={'passed':True,'independent_statistic_assertions':n,'primary_development_passed':gate['passed'],'failed_primary_guard_verified':blocked,
        'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'case_output_created':False}
    dump(run/'development_audit.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
