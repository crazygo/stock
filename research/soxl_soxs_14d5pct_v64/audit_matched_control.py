"""Require exact v63 fixed/unconditional histories and executable controls."""
import argparse
from path_learning import *


def matched_source():
    directory=core.ROOT/'research/soxl_soxs_14d5pct_v63';manifest=read(directory/'evidence/delivery_manifest.json')
    for f,h in manifest['delivery_files_sha256'].items():assert sha(directory/f)==h,f
    run=Path(manifest['local_run']);v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')==manifest['results_sha256']
    for f,h in read(run/'input_manifest.json')['derived_files_sha256'].items():assert sha(run/f)==h,f
    return run


def compare_record(saved,old):
    for kind,prior_kind in [('ordinary','conditional'),('unconditional','unconditional')]:
        for key,value in old['fits'][prior_kind].items():assert saved['fits'][kind][key]==value,(kind,key)
        assert saved['raws'][kind]==old['raws'][prior_kind]
        if 'estimates' in saved:assert saved['estimates'][kind]==old['estimates'][prior_kind]
        if 'events' in saved:assert saved['events'][kind+'_band']==old['events'][prior_kind+'_band']


def audit(run):
    run=Path(run);old=matched_source();assert read(run/'results.json')['phase']=='development';n=0;days=0
    bank=dict(np.load(run/'path_bank.npz'));prior=dict(np.load(old/'path_bank.npz'))
    for f,a in prior.items():np.testing.assert_array_equal(bank[f],a)
    for file in sorted((run/'training').glob('*.json')):
        compare_record(read(file),read(old/'training'/file.name));n+=1
    for file in sorted((run/'daily').glob('*.json')):
        a,b=read(file),read(old/'daily'/file.name)
        for k,oldk in [('ordinary','conditional'),('unconditional','unconditional')]:assert a[k]==b[oldk]
        days+=1
    x,y=read(run/'predictions.json'),read(old/'predictions.json');files=[]
    for name,oldname in [('ordinary_band','conditional_band'),('unconditional_band','unconditional_band')]:
        assert x[name]==y[oldname]
        for suffix in ['equity','trades','orders','decisions']:
            f=name+'_'+suffix+'.csv';assert sha(run/f)==sha(old/(oldname+'_'+suffix+'.csv'));files.append(f)
    result={'passed':True,'queries':n,'daily_records':days,'unchanged_bank_arrays':len(prior),'exact_control_csv':files,'new_results_sha256':sha(run/'results.json'),'source_run':str(old),'source_results_sha256':sha(old/'results.json')}
    dump(run/'matched_control_audit.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
