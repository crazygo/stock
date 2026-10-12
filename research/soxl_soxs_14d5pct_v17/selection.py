"""Freeze a development-selected candidate before any new case replay."""
from pathlib import Path
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v16'))
from anchored_allocation import core,development_gate
from verify_continuation import read

SOURCE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selection():
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    source=core.ROOT/p['selection_run'];result=read(source/'results.json');verification=read(source/'verification.json');selection=prereg['development_selection']
    assert result['phase']=='development' and verification['passed']
    assert verification['results_sha256']==sha(source/'results.json')==selection['results_sha256']
    assert sha(source/'verification.json')==selection['verification_sha256']
    assert p['primary']=='anchor_local_full'==selection['selected_route']
    assert sha(SOURCE.parent/'soxl_soxs_14d5pct_v16/anchored_allocation.py')==selection['policy_sha256']
    assert result['primary_name']!='anchor_local_full' and not result['development_gate']['passed']
    gate=development_gate(result['routes'][p['primary']],result['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Selected development candidate did not pass the frozen pre-case screen')
    return {'version':'v17','selected_route':p['primary'],'selection_run':str(source.resolve()),
            'results_sha256':sha(source/'results.json'),'gate':gate,'selection_used_new_case_results':False,
            'v16_primary_remains_failed':True}
