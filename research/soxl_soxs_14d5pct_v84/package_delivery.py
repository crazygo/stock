"""Package verified code, frozen protocol and derived results without raw market data."""
import argparse,shutil
from datetime import datetime,timezone
from state_calibration import *


def package(run):
    run=Path(run).resolve();v=read(run/'verification.json');d=read(run/'diagnostic_audit.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert d['passed'] and d['diagnostics_sha256']==sha(run/'diagnostics.json')
    assert not read(run/'case_guard.json')['authorized']
    for f,h in read(SOURCE/'evidence/preregistration.json')['code_sha256'].items():assert sha(ROOT/f)==h,f
    files=['results.json','verification.json','registration.json','diagnostics.json','diagnostic_audit.json','diagnostic_rows.csv',
        'event_features.csv','actual_entry_forecasts.csv','case_guard.json','trade_differences.json']
    files += [k+'_'+s for k in ROUTES for s in ['equity.csv','trades.csv','decisions.csv']]
    for f in files:shutil.copyfile(run/f,SOURCE/'evidence'/f)
    shutil.copyfile(CACHE/'tests.log',SOURCE/'evidence/tests.log')
    dump(SOURCE/'evidence/final_verification.json',{'passed':True,'created_at':datetime.now(timezone.utc).isoformat(),
        'parent_delivery_sha256':verify_parent(),'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),
        'no_raw_market_or_fitted_models_copied':True,'goal_complete':False})
    files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
    dump(SOURCE/'evidence/delivery_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'delivery_files_sha256':files,
        'cache_root':str(run.relative_to(ROOT)),'cache_files_sha256':{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()},
        'input_root':str(PARENT.relative_to(ROOT)),'input_files_sha256':read(SOURCE/'evidence/preregistration.json')['input_files_sha256'],
        'additional_cache_files_sha256':{f:sha(CACHE/f) for f in ['source_initial_network_failure.log','source_retry_network_failure.log','source_audit.log','calendar_build.log','preflight.log']},
        'event_source_cache_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (CACHE/'sources').rglob('*') if p.is_file()},
        'goal_complete':False,'new_buy_eligible':False});print({'delivery_files':len(files)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);package(p.parse_args().run)
