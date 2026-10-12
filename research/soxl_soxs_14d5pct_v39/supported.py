"""Freeze the exposed v38 all-supported diagnostic as a separate v39 candidate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v38'))
from eligible_risk import *
import eligible_risk as parent
SOURCE=Path(__file__).resolve().parent
ROUTES=['all_supported','eligible_supported','all_gate','source_v35','cash','double_cost']


def load_frozen():
    p=read(SOURCE/'protocol.json');r=read(SOURCE/'evidence/preregistration.json')
    assert r['protocol_sha256']==sha(SOURCE/'protocol.json') and r['protocol']==p
    for f,h in r['code_sha256'].items():assert sha(core.ROOT/f)==h
    return p,r


def authorize(path):
    p,_=load_frozen();run=Path(path);reg=read(run/'registration.json');r=read(run/'results.json');v=read(run/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert r['primary_name']==p['primary']=='all_supported' and r['primary']==r['routes']['all_supported']
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert v['registered_primary_verified'] and v['primary_development_passed']
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Registered v39 primary development failed')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'gate':gate}


def selected_events(issued):
    out={k:issued[k] for k in ROUTES if k!='double_cost'}
    out['double_cost']=dict(out['all_supported'])
    return out


def history_source():
    p,_=load_frozen();root=core.ROOT/p['selected_development_source'];d=read(core.ROOT/'research/soxl_soxs_14d5pct_v38/evidence/delivery_manifest.json')
    for key,folder in [('cache_outputs_sha256',''),('daily_records_sha256','daily'),('training_records_sha256','training')]:
        for f,h in d[key].items():assert sha(root/folder/f)==h
    assert read(root/'verification.json')['passed']
    return root,d


def case_source(path,authorization):
    root=Path(path);reg=read(root/'registration.json');v=read(root/'verification.json');receipt=read(root/'extension_receipt.json')
    assert reg['phase']=='cases' and reg['consumer_authorization']==authorization
    assert v['passed'] and v['results_sha256']==sha(root/'results.json') and v['case_prefix_exact']
    assert receipt['verification_sha256']==sha(root/'verification.json') and receipt['consumer_only']
    for f,h in receipt['files_sha256'].items():assert sha(root/f)==h
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    return root,records,receipt['files_sha256']
