"""Require identical v65 predictive functions and independently replayed primary."""
import argparse,ast,json
from path_learning import *
from run_planning import json_ready
from verify_financial import check_action


def source():
    directory=core.ROOT/'research/soxl_soxs_14d5pct_v65'
    m=read(directory/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(directory/f)==h,f
    run=Path(m['local_run']);v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')==m['results_sha256']
    old=ast.parse((directory/'path_learning.py').read_text());new=ast.parse((SOURCE/'path_learning.py').read_text())
    assert len(old.body)==len(new.body)
    for a,b in zip(old.body,new.body):
        if isinstance(a,ast.FunctionDef) and a.name=='events':continue
        assert ast.dump(a)==ast.dump(b)
    return directory,run


def preflight():
    directory,old=source();checks=0;queries=0
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    for f in sorted((old/'training').glob('*.json')):
        rec=read(f);at=pd.Timestamp(rec['at']);date=pd.Timestamp(rec['date']);clock=rec['clock'];daily=read(old/'daily'/(str(date.date())+'.json'))
        emitted=canonical(events(at,date,clock,rec['estimates'],daily,rec['events']['source_v56'],rec['events']['source_v55']))
        for name in ROUTES:
            e=emitted[name]
            if name in ['premarket_only','double_cost']:
                if name=='premarket_only' and clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0;continue
                assert e==rec['events']['conditional_point']
            else:assert e==rec['events'][name]
            if name in ['conditional_point','premarket_only','double_cost']:
                check_action(rec['estimates']['conditional'],daily['conditional']['feedback']['expired_unresolved_prefix'],False,e);checks+=1
        queries+=1
    out={'passed':True,'queries':queries,'independent_point_action_checks':checks,'predictive_functions_AST_identical':True,'source_results_sha256':sha(old/'results.json'),'source_verification_sha256':sha(old/'verification.json'),'source_delivery_sha256':sha(directory/'evidence/delivery_manifest.json'),'new_capital_replayed':False,'development_selection_disclosed':True,'case_outcomes_used_for_selection':False}
    dump(SOURCE/'evidence/policy_preflight.json',out);print(out,flush=True)


def audit(run):
    directory,old=source();run=Path(run);queries=days=0;assert read(run/'results.json')['phase']=='development'
    a,b=dict(np.load(run/'path_bank.npz')),dict(np.load(old/'path_bank.npz'));assert set(a)==set(b);array_count=len(a)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    for f in sorted((run/'training').glob('*.json')):
        a,b=read(f),read(old/'training'/f.name)
        for key in a:
            if key=='events':continue
            assert a[key]==b[key],(f.name,key)
        for name in ROUTES:
            if name in ['premarket_only','double_cost']:continue
            assert a['events'][name]==b['events'][name]
        queries+=1
    for folder in ['daily','source_daily']:
        for f in (run/folder).glob('*.json'):assert read(f)==read(old/folder/f.name);days+=1
    x,y=read(run/'predictions.json'),read(old/'predictions.json');files=[]
    for name in ROUTES:
        if name in ['premarket_only','double_cost']:continue
        assert x[name]==y[name]
        for suffix in ['equity','trades','orders','decisions']:
            f=name+'_'+suffix+'.csv';assert sha(run/f)==sha(old/f);files.append(f)
    r=read(run/'results.json');prior=read(old/'results.json')
    assert r['primary_name']=='conditional_point' and r['primary']==prior['routes']['conditional_point']
    out={'passed':True,'queries':queries,'daily_and_source_records':days,'unchanged_bank_arrays':array_count,'exact_csv':files,'primary_is_exact_v65_point_control':True,'source_results_sha256':sha(old/'results.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'policy_identity_audit.json',out);print(out,flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);args=p.parse_args()
    audit(args.run) if args.run else preflight()
