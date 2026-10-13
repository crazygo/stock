"""Independent gate and actual CLI rejection before new case output."""
import argparse,subprocess
from path_learning import *
from audit_statistics import audit_statistics


def audit(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');p=read(run/'registration.json')['preregistration']['protocol']
    assert r['phase']=='development' and v['passed'] and v['results_sha256']==sha(run/'results.json');n=audit_statistics(run)
    gate=development_gate(r['primary'],r['coverage']['complete'],p);assert r['primary_name']=='dynamic_point' and r['primary']==r['routes']['dynamic_point'] and r['development_gate']==gate
    blocked=False
    if not gate['passed']:
        out=run.parent/'blocked_cases_guard_check';assert not out.exists();result=subprocess.run([sys.executable,str(SOURCE/'run_paths.py'),'--output',str(out),'--case-after',str(run)],text=True,capture_output=True)
        assert result.returncode!=0 and 'Development primary failed; new case replay is forbidden' in result.stderr and not out.exists();blocked=True
    receipt={'passed':True,'statistics':n,'primary_development_passed':gate['passed'],'failed_primary_guard_verified':blocked,'case_output_created':False,'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json')}
    dump(run/'development_audit.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
