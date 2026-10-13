"""Registered mechanism diagnostics; no decisions or thresholds are changed."""
import argparse,gzip,pickle
from path_learning import *


def report(run):
    run=Path(run);results=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'][results['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');bank=dict(np.load(run/'path_bank.npz'));rows=[];same_resolved=0;matched_tail=0
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));a,b=rec['fits']['conditional'],rec['fits']['fixed']
        if not a['ready']:continue
        assert b['ready'] and a['members']==b['members'] and a['kernels']['later']==b['kernels']['later'];wa=np.array(a['kernels']['first']['probability']);wb=np.array(b['kernels']['first']['probability'])
        members=np.array(a['members']);values=bank['activity_states'][bank['blocks'][members,0],CLOCKS.index(row.clock)];target=float(row[ACTIVITY]);models=[]
        for k in ['conditional','fixed']:
            with gzip.open(run/'models'/(k+'_'+tag(at)+'.pkl.gz'),'rb') as f:models.append(pickle.load(f))
        am,bm=models;np.testing.assert_array_equal(am['resolved'],bm['resolved'])
        for key,val in am['partition_result'].items():np.testing.assert_array_equal(val,bm['partition_result'][key])
        same_resolved+=int(2*sum(am['resolved']));np.testing.assert_array_equal(am['selected_blocks'][:,1:],bm['selected_blocks'][:,1:])
        identical=am['selected_blocks'][:,0]==bm['selected_blocks'][:,0]
        if len(identical):
            for key in am['result']:np.testing.assert_array_equal(am['result'][key][identical],bm['result'][key][identical])
            matched_tail+=int(2*sum(identical))
        rows.append({'at':at,'date':row.date,'first_weight_total_variation':float(sum(abs(wa-wb))/2),'new_activity_alignment_error':float(abs(wa@values-target)),
                     'old_activity_alignment_error':float(abs(wb@values-target)),'new_first_ess':a['kernels']['first']['effective_dates'],'old_first_ess':b['kernels']['first']['effective_dates'],
                     'new_resolved_probability':1-a['partition']['unresolved_probability'],'old_resolved_probability':1-b['partition']['unresolved_probability'],
                     'same_unresolved_tail_first_fraction':float(np.mean(identical)) if len(identical) else None,'action_changed':rec['events']['conditional_band']['action']!=rec['events']['fixed_band']['action'],
                     'raw_difference':np.array(rec['raws']['conditional'])-np.array(rec['raws']['fixed'])})
    frame=pd.DataFrame(rows);numeric=[c for c in frame.columns if c not in ['at','date','raw_difference','action_changed']]
    summary=frame.groupby('date')[numeric].mean().mean().to_dict()
    out={'passed':True,'queries':len(rows),'dates':int(frame.date.nunique()),'date_equal_averages':summary,'different_proposed_action_queries':int(frame.action_changed.sum()),'unweighted_resolved_contract_values_exact':same_resolved,
         'identical_tail_path_contract_values_exact':matched_tail,'later_source_blocks_identical':True,'raw_difference_mean':np.mean(frame.raw_difference.to_list(),axis=0),'rows':rows,
         'results_sha256':sha(run/'results.json'),'scope':'Alignment and changed weights/actions do not establish predictive or capital value. Tail matching concerns U-conditioned samples, not full first-block probability mass.'}
    dump(run/'activity_diagnostics.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
