"""Freeze and verify transitive code, numerical preflight and reviewed inputs."""
from datetime import datetime,timezone
from calibration91 import *


def verify_registration(pre):
    for field,file in [('protocol_sha256','protocol.json'),('method_sha256','METHOD.md'),
                       ('model_preflight_sha256','evidence/model_preflight.json'),
                       ('fundamental_audit_sha256','evidence/fundamental_verification.json')]:
        assert pre[field]==sha(SOURCE/file),file
    assert pre['parent_delivery_sha256']==verify_parent()
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    for f,h in pre['input_files_sha256'].items():assert sha(PARENT/f)==h,f
    assert pre['tests_sha256']==sha(CACHE/'tests.log')


def register():
    target=SOURCE/'evidence/preregistration.json';assert not target.exists()
    prior=read(upstream.SOURCE/'evidence/preregistration.json');code=dict(prior['code_sha256'])
    for f,h in code.items():assert sha(ROOT/f)==h,f
    for folder in [SOURCE,upstream.SOURCE,ROOT/'research/soxl_soxs_14d5pct_v90']:
        code.update({str(f.relative_to(ROOT)):sha(f) for f in folder.glob('*.py')})
    assert read(SOURCE/'evidence/model_preflight.json')['passed']
    assert read(SOURCE/'evidence/fundamental_verification.json')['passed']
    assert '\nOK\n' in (CACHE/'tests.log').read_text()
    value=dict(created_at=datetime.now(timezone.utc).isoformat(),protocol=read(SOURCE/'protocol.json'),
        protocol_sha256=sha(SOURCE/'protocol.json'),method_sha256=sha(SOURCE/'METHOD.md'),
        model_preflight_sha256=sha(SOURCE/'evidence/model_preflight.json'),
        fundamental_audit_sha256=sha(SOURCE/'evidence/fundamental_verification.json'),
        tests_sha256=sha(CACHE/'tests.log'),parent_delivery_sha256=verify_parent(),code_sha256=code,
        input_files_sha256={f:sha(PARENT/f) for f in ['semiconductor_panel.parquet','features.parquet','baseline_labels.json','stress_labels.json']},
        historical_exposure=True,new_capital_outcomes_seen=False,goal_complete=False)
    verify_registration(value);dump(target,value)
    print(dict(registered_sources=len(code),preregistration_sha256=sha(target)),flush=True)


if __name__=='__main__':register()
