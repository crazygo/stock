"""Same-input IID versus RQMC development/case evidence; never selects a route."""
import argparse, gzip, pickle
from path_learning import *


def compare(run):
    run=Path(run);result=read(run/'results.json');verification=read(run/'verification.json')
    assert verification['passed'] and verification['results_sha256']==sha(run/'results.json')
    phase=result['phase'];old=core.ROOT/f'.cache/soxl_soxs_14d5pct_v60/runs/{phase}_20261013'
    old_result=read(old/'results.json');old_verify=read(old/'verification.json')
    assert old_verify['passed'] and old_verify['results_sha256']==sha(old/'results.json')
    manifest=read(core.ROOT/'research/soxl_soxs_14d5pct_v60/evidence/delivery_manifest.json')['runs'][old.name]
    for f in ['results.json','verification.json','diagnostics.json']:
        assert sha(old/f)==manifest['top_level_sha256'][f]
    for f in ['features.parquet','financial_panel.parquet','activity_panel.parquet','baseline_labels.json','stress_labels.json']:
        assert sha(run/f)==sha(old/f)
    new_bank=np.load(run/'path_bank.npz');old_bank=np.load(old/'path_bank.npz')
    for k in old_bank.files:np.testing.assert_array_equal(new_bank[k],old_bank[k])
    rows=[];panel=pd.read_parquet(run/'activity_panel.parquet');p=read(SOURCE/'protocol.json')[phase]
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        records={'rqmc':read(run/'training'/(tag(at)+'.json')),'iid':read(old/'training'/(tag(at)+'.json'))}
        for kind in VARIANTS:
            for method,folder in [('rqmc',run),('iid',old)]:
                rec=records[method];fit=rec['fits'][kind]
                if not fit['ready']:continue
                if method=='rqmc':cov=np.array(fit['integration']['mean_covariance'])
                else:
                    modelname=kind+'_'+tag(at)+'.pkl.gz'
                    assert sha(old/'models'/modelname)==manifest['base_models_sha256'][modelname]
                    with gzip.open(folder/'models'/modelname,'rb') as f:model=pickle.load(f)
                    cov=np.cov(model['result']['log_returns'],rowvar=False)/2048
                day=read(folder/'daily'/(str(row.date.date())+'.json'))[kind]
                if not day['ready']:continue
                c=CLOCKS.index(row.clock);model=day['model']['point_model']
                # Rows are output dimensions; columns are raw pair input dimensions.
                J=np.eye(2)+np.array(model['coefficients'][c]).T/np.array(model['input_scale'][c])[None,:]
                after=J@cov@J.T
                rows.append({'at':at,'date':row.date,'kind':kind,'method':method,
                    'raw_mean_se':float(np.sqrt(np.diag(cov)).mean()),
                    'calibrated_mean_se':float(np.sqrt(np.diag(after)).mean()),
                    'point':rec['estimates'][kind]['point'],'lower':rec['estimates'][kind]['lower']})
    frame=pd.DataFrame(rows);summaries={}
    for (kind,method),g in frame.groupby(['kind','method']):
        means=g.groupby('date')[['raw_mean_se','calibrated_mean_se']].mean().mean()
        summaries[kind+'_'+method]={'queries':len(g),'dates':int(g.date.nunique()),**means.to_dict()}
    out={'phase':phase,'same_inputs_and_contract':True,'v60_results_sha256':sha(old/'results.json'),
        'v61_results_sha256':sha(run/'results.json'),'v60_primary':old_result['primary'],
        'v61_primary':result['primary'],'numerical_uncertainty':summaries,
        'diagnostics':{'v60':read(old/'diagnostics.json')['variants'],'v61':read(run/'diagnostics.json')['variants']},
        'numerical_error_is_not_market_uncertainty':True,'rows':rows}
    dump(run/'iid_comparison.json',out)
    print({'numerical_uncertainty':summaries,'v60_mean':old_result['primary']['primary_average'],'v61_mean':result['primary']['primary_average']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);compare(p.parse_args().run)
