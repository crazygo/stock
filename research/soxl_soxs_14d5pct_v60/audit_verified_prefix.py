"""Reuse completed development checks only after exact artifact and causal-input proof."""
import gzip,pickle
from path_learning import *
from audit_case_prefix import audit_prefix


def exact_model(a,b):
    assert type(a) is type(b)
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact_model(a[k],b[k])
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,strict=True)
    else:assert a==b


def certify_prefix(dev,run):
    v=read(dev/'verification.json');r=read(dev/'results.json');dr=read(dev/'registration.json');rr=read(run/'registration.json')
    assert v['passed'] and v['phase']=='development' and v['results_sha256']==sha(dev/'results.json')
    assert v['audit_amendment_sha256']==sha(SOURCE/'evidence/audit_amendment.json')
    assert r['development_gate']['passed'] and rr['authorization']['verification_sha256']==sha(dev/'verification.json')
    assert dr['protocol_sha256']==rr['protocol_sha256']==sha(SOURCE/'protocol.json')
    matured=audit_prefix(dev,run);old=pd.read_parquet(dev/'activity_panel.parquet');new=pd.read_parquet(run/'activity_panel.parquet');assert list(new.index[:len(old)])==list(old.index)
    for k,es in read(dev/'predictions.json').items():assert read(run/'predictions.json')[k][:len(es)]==es
    files=0
    for folder in ['training','daily','source_daily']:
        for f in (dev/folder).glob('*.json'):
            assert sha(f)==sha(run/folder/f.name),(folder,f.name);files+=1
    a=dict(np.load(dev/'path_bank.npz'));b=dict(np.load(run/'path_bank.npz'))
    for k,x in a.items():np.testing.assert_array_equal(x,b[k][:len(x)],strict=True)
    rows=read(dev/'bank_records.json');assert read(run/'bank_records.json')[:len(rows)]==rows
    models=0
    for at in old.index:
        for k in VARIANTS:
            f=dev/'models'/(k+'_'+tag(at)+'.pkl.gz');g=run/'models'/f.name;assert f.exists()==g.exists()
            if f.exists():
                with gzip.open(f,'rb') as x,gzip.open(g,'rb') as y:exact_model(pickle.load(x),pickle.load(y))
                models+=1
        for k in ['v56','v55']:assert not (run/'baseline_models'/(k+'_'+tag(at)+'.pkl.gz')).exists()
    return {'passed':True,'reused_queries':len(old),'exact_json_files':files,'exact_models':models,'newly_matured_outcomes':matured,'development_verification_sha256':sha(dev/'verification.json'),'no_reuse_for_new_case_queries':True}
