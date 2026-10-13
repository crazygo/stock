"""Package verified research outputs; keep raw prices and fitted models in cache."""
import argparse,shutil
from datetime import datetime,timezone
from class_learning import *


def package(run):
    run=Path(run).resolve();v=read(run/'verification.json');d=read(run/'diagnostic_audit.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert d['passed'] and d['diagnostics_sha256']==sha(run/'diagnostics.json')
    entries=read(run/'entry_diagnostics.json')
    assert entries['results_sha256']==sha(run/'results.json') and entries['source_sha256']==sha(SOURCE/'entry_diagnostics.py')
    assert not entries['changes_policy_or_gate'] and not read(run/'case_guard.json')['authorized']
    for f,h in read(SOURCE/'evidence/preregistration.json')['code_sha256'].items():assert sha(ROOT/f)==h,f
    files=['results.json','verification.json','registration.json','input_registration.json','input_manifest.json','input_audit.json',
        'diagnostics.json','diagnostic_audit.json','diagnostic_rows.csv','actual_entry_forecasts.csv','mixture_weights.csv','case_guard.json',
        'entry_diagnostics.json','matched_entry_forecasts.csv']
    files += [k+'_'+suffix for k in ROUTES for suffix in ['equity.csv','trades.csv','decisions.csv']]
    for f in files:shutil.copyfile(run/f,SOURCE/'evidence'/f)
    for f in ['tests.log','tests_attempt1.log','preflight_attempt1.log','preflight_attempt2.log','newton_diagnostics.log']:
        shutil.copyfile(CACHE/f,SOURCE/'evidence'/f)
    dump(SOURCE/'evidence/final_verification.json',{'passed':True,'created_at':datetime.now(timezone.utc).isoformat(),
        'parent_delivery_sha256':verify_parent(),'results_sha256':sha(run/'results.json'),
        'verification_sha256':sha(run/'verification.json'),'no_raw_prices_or_fitted_models_copied':True,'goal_complete':False})
    files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
    dump(SOURCE/'evidence/delivery_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'delivery_files_sha256':files,
        'cache_root':str(run.relative_to(ROOT)),'cache_files_sha256':{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()},
        'input_root':str((CACHE/'inputs/development').relative_to(ROOT)),
        'input_files_sha256':{p.name:sha(p) for p in (CACHE/'inputs/development').iterdir() if p.is_file()},
        'additional_cache_files_sha256':{str(p.relative_to(CACHE)):sha(p) for p in (CACHE/'input_sources').glob('*.py')},
        'goal_complete':False,'new_buy_eligible':False})
    print({'delivery_files':len(files)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);package(p.parse_args().run)
