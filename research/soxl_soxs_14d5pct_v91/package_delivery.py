"""Package verified derived evidence, preserving raw inputs and fitted models in cache."""
import argparse,shutil,gzip
from datetime import datetime,timezone
from calibration91 import *
from registration91 import verify_registration


def package(run):
    run=Path(run).resolve();v=read(run/'verification.json');d=read(run/'diagnostic_audit.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert d['passed'] and d['diagnostics_sha256']==sha(run/'diagnostics.json')
    pre=read(SOURCE/'evidence/preregistration.json');verify_registration(pre)
    gate=read(run/'case_guard.json');r=read(run/'results.json')
    assert gate['authorized']==r['development_gate']['passed']
    files=['results.json','verification.json','registration.json','diagnostics.json','diagnostic_audit.json','diagnostic_rows.csv',
        'actual_entry_forecasts.csv','case_guard.json','trade_differences.json']
    files += [k+'_'+s for k in ROUTES for s in ['equity.csv','trades.csv','decisions.csv']]
    for f in files:shutil.copyfile(run/f,SOURCE/'evidence'/f)
    encodings={}
    for name in ['event_features.csv','fundamental_features.csv']:
        raw=(run/name).read_bytes();target=SOURCE/'evidence'/(name+'.gz')
        target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(target.read_bytes())==raw
        encodings['evidence/'+name+'.gz']=dict(decompressed_sha256=sha(run/name),decompressed_bytes=len(raw),compressed_bytes=target.stat().st_size,encoding='gzip',roundtrip_verified=True)
    shutil.copyfile(CACHE/'tests.log',SOURCE/'evidence/tests.log')
    dump(SOURCE/'evidence/final_verification.json',dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),
        parent_delivery_sha256=verify_parent(),results_sha256=sha(run/'results.json'),verification_sha256=sha(run/'verification.json'),
        no_raw_market_or_fitted_models_copied=True,goal_complete=False))
    files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json'}
    dump(SOURCE/'evidence/delivery_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),delivery_files_sha256=files,
        cache_root=str(run.relative_to(ROOT)),cache_files_sha256={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()},
        input_root=str(PARENT.relative_to(ROOT)),input_files_sha256=pre['input_files_sha256'],
        additional_cache_files_sha256={f:sha(CACHE/f) for f in ['fundamental_verification.log','preflight.log','tests.log']},
        source_manifest_sha256=sha(ROOT/'research/soxl_soxs_14d5pct_v90/evidence/delivery_manifest.json'),
        lossless_delivery_encodings=encodings,goal_complete=False,new_buy_eligible=False))
    print(dict(delivery_files=len(files),capital_books=len(ROUTES),lossless_feature_tables=len(encodings)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);package(p.parse_args().run)
