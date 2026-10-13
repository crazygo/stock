"""Read-only matched-path attribution; no selection or parameter changes."""
import argparse,gzip,pickle
from path_learning import *


def inspect(run):
    run=Path(run);result=read(run/'results.json');verify=read(run/'verification.json')
    assert verify['passed'] and verify['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');p=read(SOURCE/'protocol.json')[result['phase']];rows=[];same=0
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'))
        assert rec['fits']['conditional']['ready']==rec['fits']['fixed']['ready']
        if not rec['fits']['conditional']['ready']:continue
        models=[]
        for kind in ['conditional','fixed']:
            with gzip.open(run/'models'/(kind+'_'+tag(at)+'.pkl.gz'),'rb') as f:models.append(pickle.load(f))
        a,b=models;np.testing.assert_array_equal(a['selected_blocks'][:,0],b['selected_blocks'][:,0])
        x,y=a['result'],b['result'];survived=x['exit_day']>=2
        np.testing.assert_array_equal(survived,y['exit_day']>=2)
        np.testing.assert_array_equal(x['log_returns'][~survived],y['log_returns'][~survived]);same+=int((~survived).sum())
        delta=x['log_returns']-y['log_returns'];replicate_means=delta.reshape(8,256,2).mean(axis=1)
        se=np.std(replicate_means,axis=0,ddof=1)/np.sqrt(8)
        rows.append({'at':at,'date':row.date,'survives_two_sessions_fraction':survived.mean(axis=0),
            'expected_log_return_difference':delta.mean(axis=0),'paired_numerical_se':se,
            'same_proposed_action':rec['events']['conditional_band']['action']==rec['events']['fixed_band']['action']})
    frame=pd.DataFrame(rows)
    mean=lambda field:np.mean([np.mean(np.stack(g[field]),axis=0) for _,g in frame.groupby('date')],axis=0)
    out={'phase':result['phase'],'queries':len(rows),'dates':frame.date.nunique(),
         'first_block_exits_log_returns_identical':same,
         'date_equal_survival_fraction':mean('survives_two_sessions_fraction'),
         'date_equal_mean_log_return_difference':mean('expected_log_return_difference'),
         'date_equal_paired_integration_se':mean('paired_numerical_se'),
         'different_action_queries':int((~frame.same_proposed_action).sum()),
         'numerical_difference_is_not_market_uncertainty':True,
         'results_sha256':sha(run/'results.json'),'records':rows}
    dump(run/'path_effect.json',out);print({k:v for k,v in out.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);inspect(p.parse_args().run)
