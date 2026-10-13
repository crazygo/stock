"""Registered matched first-block attribution; no strategy selection."""
import argparse,gzip,pickle
from path_learning import *


def report(run):
    run=Path(run);result=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');bank=dict(np.load(run/'path_bank.npz'));p=read(SOURCE/'protocol.json')[result['phase']];records=[]
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));a,b=rec['fits']['conditional'],rec['fits']['fixed'];assert a['ready']==b['ready']
        if not a['ready']:continue
        assert a['members']==b['members'];ids=bank['blocks'][a['members'],0];c=CLOCKS.index(row.clock);states=bank['absolute_states'][ids,c]
        w=np.array(a['kernels']['first']['probability']);z=np.array(b['kernels']['first']['probability'])
        models=[]
        for k in ['conditional','fixed']:
            with gzip.open(run/'models'/(k+'_'+tag(at)+'.pkl.gz'),'rb') as file:models.append(pickle.load(file))
        x,y=models;np.testing.assert_array_equal(x['selected_blocks'][:,1:],y['selected_blocks'][:,1:])
        diff=x['result']['log_returns']-y['result']['log_returns'];means=diff.reshape(8,256,2).mean(axis=1)
        records.append({'at':at,'date':row.date,'first_weight_total_variation':float(np.abs(w-z).sum()/2),
            'new_weighted_absolute_state':float(w@states),'old_weighted_absolute_state':float(z@states),'current_absolute_state':float(row[ABSOLUTE]),
            'new_absolute_alignment_error':float(abs(w@states-row[ABSOLUTE])),'old_absolute_alignment_error':float(abs(z@states-row[ABSOLUTE])),
            'new_first_ess':a['kernels']['first']['effective_dates'],'old_first_ess':b['kernels']['first']['effective_dates'],
            'same_first_block_fraction':float(np.mean(x['selected_blocks'][:,0]==y['selected_blocks'][:,0])),
            'paired_mean_log_return_difference':diff.mean(axis=0),'paired_integration_se':means.std(axis=0,ddof=1)/np.sqrt(8),
            'same_proposed_action':rec['events']['conditional_band']['action']==rec['events']['fixed_band']['action']})
    frame=pd.DataFrame(records);bydate=frame.groupby('date');cols=['first_weight_total_variation','new_absolute_alignment_error','old_absolute_alignment_error','new_first_ess','old_first_ess','same_first_block_fraction'];averages=bydate[cols].mean().mean().to_dict()
    for field in ['paired_mean_log_return_difference','paired_integration_se']:averages[field]=np.mean([np.stack(g[field]).mean(axis=0) for _,g in bydate],axis=0)
    out={'phase':result['phase'],'queries':len(frame),'dates':frame.date.nunique(),'date_equal_averages':averages,'later_source_blocks_identical':True,'different_proposed_action_queries':int((~frame.same_proposed_action).sum()),'alignment_is_not_return_acceptance':True,'results_sha256':sha(run/'results.json'),'records':records}
    dump(run/'first_match_diagnostics.json',out);print({k:z for k,z in out.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
