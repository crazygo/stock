"""Freeze this protocol and transitive source allowlist before capital evaluation."""
from datetime import datetime,timezone
from shift_learning import *


if __name__=='__main__':
    target=SOURCE/'evidence/preregistration.json';assert not target.exists()
    old=read(ROOT/'research/soxl_soxs_14d5pct_v79/evidence/preregistration.json')
    for f,h in old['code_sha256'].items():assert sha(ROOT/f)==h,f
    code=dict(old['code_sha256'])
    for folder in [SOURCE,ROOT/'research/soxl_soxs_14d5pct_v79']:
        code.update({str(f.relative_to(ROOT)):sha(f) for f in folder.glob('*.py')})
    preflight=read(SOURCE/'evidence/model_preflight.json');assert preflight['passed']
    assert '\nOK\n' in (CACHE/'tests.log').read_text()
    dump(target,{'created_at':datetime.now(timezone.utc).isoformat(),'protocol':read(SOURCE/'protocol.json'),
        'protocol_sha256':sha(SOURCE/'protocol.json'),'method_sha256':sha(SOURCE/'METHOD.md'),
        'model_preflight_sha256':sha(SOURCE/'evidence/model_preflight.json'),'tests_sha256':sha(CACHE/'tests.log'),
        'parent_delivery_sha256':verify_parent(),'code_sha256':code,
        'input_files_sha256':{f:sha(PARENT/f) for f in ['semiconductor_panel.parquet','features.parquet','baseline_labels.json','stress_labels.json']},
        'new_capital_outcomes_seen':False,'historical_exposure':True,'goal_complete':False})
    print({'registered_sources':len(code),'preregistration_sha256':sha(target)},flush=True)
