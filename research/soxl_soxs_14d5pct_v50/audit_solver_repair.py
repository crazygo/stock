"""Verify preserved failed prefix and unchanged KKT gate after numerical repair."""
import argparse
from confidence_shrinkage_v2 import *


def audit(run):
    run=Path(run);reg=read(run/'registration.json');add=reg['preregistration']['numerical_solver_addendum'];old=Path(add['failed_run'])
    assert not (old/'results.json').exists() and add['capital_outcomes_seen'] is False and add['no_threshold_relaxation']
    v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    prefix={}
    for directory in ['daily','training']:
        fs=sorted((old/directory).glob('*.json'))
        for f in fs:assert sha(f)==sha(run/directory/f.name),str(f)
        prefix[directory]=len(fs)
    refined=[];models=0;non_success=0;max_residual_fraction=0.
    for f in sorted((run/'daily').glob('*.json')):
        daily=read(f)
        for kind,group in daily['offsets'].items():
            for b,m in enumerate([group['point_model'],*group['bootstrap_models']]):
                models+=1;non_success+=int(not m['solver_success']);limit=1e-8*max(1,m['n_dates'])
                assert m['projected_gradient_max']<=limit;max_residual_fraction=max(max_residual_fraction,m['projected_gradient_max']/limit)
                if 'numerical_refinement' in m:
                    r=m['numerical_refinement'];assert r['same_objective_and_bounds'] and 1<=r['steps']<=25 and r['initial_projected_gradient_max']>limit
                    refined.append({'date':f.stem,'variant':kind,'replicate':'point' if b==0 else b-1,'steps':r['steps'],
                        'initial_residual':r['initial_projected_gradient_max'],'final_residual':m['projected_gradient_max'],'unchanged_limit':limit})
    receipt={'passed':True,'original_complete_prefix_byte_exact':prefix,'old_capital_results_created':False,
        'direction_models_checked':models,'solver_non_success_but_KKT_certified':non_success,
        'max_residual_fraction_of_unchanged_limit':max_residual_fraction,'refined_models':len(refined),'refinements':refined,
        'verification_sha256':sha(run/'verification.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'solver_repair_audit.json',receipt);print({k:v for k,v in receipt.items() if k!='refinements'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
