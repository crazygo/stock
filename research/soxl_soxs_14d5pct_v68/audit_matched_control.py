"""Exact v67 fresh development control; cases retain that immutable prefix."""
import argparse
from path_learning import *


def verified_manifest(version):
    directory=core.ROOT/('research/soxl_soxs_14d5pct_'+version);m=read(directory/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(directory/f)==h,f
    return m


def matched_bank_source(phase='development'):
    m=verified_manifest('v66');run=Path(m['local_run' if phase=='development' else 'case_run'])
    v=read(run/'verification.json');expected=m['results_sha256' if phase=='development' else 'case_results_sha256']
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')==expected
    for f,h in read(run/'input_manifest.json')['derived_files_sha256'].items():assert sha(run/f)==h
    return run


def matched_source(phase='development'):
    m=verified_manifest('v67');run=Path(m['local_run']);v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')==m['results_sha256']
    for f,h in read(run/'input_manifest.json')['derived_files_sha256'].items():assert sha(run/f)==h
    return run


def compare_record(saved,old):
    for key in ['fits','raws','estimates']:
        if key in saved:assert saved[key]['fresh']==old[key]['fresh'],key
    if 'events' in saved:
        for name in ['fresh_point','fresh_band','source_v56','source_v55','cash']:assert saved['events'][name]==old['events'][name],name


def audit(run):
    run=Path(run);phase=read(run/'results.json')['phase'];old=matched_source();queries=days=0
    bank=dict(np.load(run/'path_bank.npz'));prior=dict(np.load(matched_bank_source(phase)/'path_bank.npz'));assert set(bank)==set(prior)
    for name in bank:np.testing.assert_array_equal(bank[name],prior[name])
    for f in sorted((old/'training').glob('*.json')):
        a,b=read(run/'training'/f.name),read(f);compare_record(a,b)
        for key in ['at','date','clock','cutoff','baselines']:assert a[key]==b[key]
        queries+=1
    for f in (old/'daily').glob('*.json'):
        assert read(run/'daily'/f.name)['fresh']==read(f)['fresh'];days+=1
        assert read(run/'source_daily'/f.name)==read(old/'source_daily'/f.name)
    a,b=read(run/'predictions.json'),read(old/'predictions.json');files=[]
    for name in ['fresh_point','fresh_band','source_v56','source_v55','cash']:
        assert a[name][:len(b[name])]==b[name]
        if phase=='development':
            assert len(a[name])==len(b[name])
            for suffix in ['equity','trades','orders','decisions']:
                f=name+'_'+suffix+'.csv';assert sha(run/f)==sha(old/f);files.append(f)
    result={'passed':True,'phase':phase,'queries':queries,'daily_records':days,'unchanged_bank_arrays':len(bank),'exact_control_csv':files,'case_extension_independently_verified_not_preexisting_case':phase=='cases','source_run':str(old),'source_results_sha256':sha(old/'results.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'matched_control_audit.json',result);print(result,flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
