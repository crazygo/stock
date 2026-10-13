"""Package verified aggregate evidence and hashes; exclude raw data and fitted models."""
import argparse
import shutil
from datetime import datetime,timezone
from history_learning import *


def package(run):
    run=Path(run).resolve();v=read(run/'verification.json');d=read(run/'diagnostic_audit.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert d['passed'] and d['diagnostics_sha256']==sha(run/'diagnostics.json')
    for f,h in read(SOURCE/'evidence/preregistration.json')['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert read(run/'input_audit.json')['passed'] and read(SOURCE/'evidence/source_sensitivity.json')['passed']
    assert read(run/'execution_sensitivity.json')['passed']
    files=['results.json','verification.json','registration.json','input_registration.json','input_manifest.json','input_audit.json',
           'diagnostics.json','diagnostic_audit.json','diagnostic_rows.csv','actual_entry_forecasts.csv','sample_mass.csv','case_guard.json','execution_sensitivity.json']
    files += [k+'_'+suffix for k in ROUTES for suffix in ['equity.csv','trades.csv','decisions.csv']]
    for f in files:shutil.copyfile(run/f,SOURCE/'evidence'/f)
    shutil.copyfile(CACHE/'tests.log',SOURCE/'evidence/tests.log')
    audit={'passed':True,'created_at':datetime.now(timezone.utc).isoformat(),'registered_source_files':len(read(SOURCE/'evidence/preregistration.json')['code_sha256']),
        'parent_delivery_sha256':verify_parents(),'copied_run_evidence_files':files,'no_raw_data_or_models_copied':True,
        'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'goal_complete':False}
    dump(SOURCE/'evidence/final_verification.json',audit)
    files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
    cache={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()}
    dump(SOURCE/'evidence/delivery_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'cache_root':str(run.relative_to(ROOT)),'delivery_files_sha256':files,'cache_files_sha256':cache,
        'input_root':str((CACHE/'inputs/development').relative_to(ROOT)),
        'input_files_sha256':{p.name:sha(p) for p in (CACHE/'inputs/development').iterdir() if p.is_file()},
        'goal_complete':False,'new_buy_eligible':False})
    print({'delivery_files':len(files),'cache_files':len(cache)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);package(p.parse_args().run)
